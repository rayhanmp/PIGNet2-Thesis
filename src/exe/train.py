# CUDA devices should be recognized first.
# isort: off
import set_cuda

# isort: on
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# isort: off
import path
import utils
from data import ComplexDataModule
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import io
import numpy as np
import math

def log_plot_to_mlflow(fig, artifact_name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    mlflow.log_image(buf, artifact_file=artifact_name)
    plt.close(fig)


def _energy_component_names(config):
    names = [
        "vdw",
        "hbond",
        "metal_ligand",
        "hydrophobic",
    ]
    if getattr(config.model, "include_ionic", False):
        names.append("ionic")
    if getattr(config.model, "include_gb", False):
        # Pairwise GB and GB self-energy are appended in the model forward
        names.append("gb_pairwise")
        names.append("gb_self")
    return names


def log_energy_distribution_to_mlflow(model, config, stage, epoch):
    """Aggregate per-sample energy components and log histogram plots + stats to MLflow."""
    # Collect values per component across all tasks/keys
    component_names = _energy_component_names(config)
    values_per_component = [[] for _ in component_names]

    # model.predictions[task][key] -> list[energy_components]
    for task in model.predictions:
        for _, energies in model.predictions[task].items():
            # Guard against unexpected length differences
            upto = min(len(energies), len(values_per_component))
            for i in range(upto):
                values_per_component[i].append(float(energies[i]))

    total_count = sum(len(v) for v in values_per_component)
    if total_count == 0:
        return  # Nothing to log this epoch (e.g., empty predictions)

    # Plot histograms for individual components
    num_components = len(component_names)
    cols = min(4, num_components)
    rows = int(math.ceil(num_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, name in enumerate(component_names):
        data = values_per_component[i]
        ax = axes_flat[i]
        if len(data) > 0:
            ax.hist(data, bins=50, color="steelblue", alpha=0.9)
        ax.set_title(f"{name} (n={len(data)})")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Count")

    # Hide any unused subplots
    for j in range(num_components, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    # Log plot image
    log_plot_to_mlflow(fig, f"energy_distribution/{stage}_epoch_{epoch}.png")

    # Log basic statistics per component for quick numeric tracking
    try:
        for i, name in enumerate(component_names):
            data = values_per_component[i]
            if len(data) == 0:
                continue
            arr = np.asarray(data, dtype=float)
            mlflow.log_metric(f"energy/{stage}/{name}_mean", float(arr.mean()), step=epoch)
            mlflow.log_metric(f"energy/{stage}/{name}_std", float(arr.std(ddof=0)), step=epoch)
            mlflow.log_metric(f"energy/{stage}/{name}_min", float(arr.min()), step=epoch)
            mlflow.log_metric(f"energy/{stage}/{name}_max", float(arr.max()), step=epoch)
    except Exception as e:
        print(f"Error logging energy distribution stats to MLflow: {e}")

    # Also log total energy distribution across samples
    try:
        totals = []
        for task in model.predictions:
            for _, energies in model.predictions[task].items():
                totals.append(float(sum(energies)))
        if len(totals) > 0:
            fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3))
            ax2.hist(totals, bins=50, color="darkorange", alpha=0.9)
            ax2.set_title(f"total_energy (n={len(totals)})")
            ax2.set_xlabel("Energy")
            ax2.set_ylabel("Count")
            plt.tight_layout()
            log_plot_to_mlflow(fig2, f"energy_distribution/{stage}_total_epoch_{epoch}.png")

            arr = np.asarray(totals, dtype=float)
            mlflow.log_metric(f"energy/{stage}/total_mean", float(arr.mean()), step=epoch)
            mlflow.log_metric(f"energy/{stage}/total_std", float(arr.std(ddof=0)), step=epoch)
            mlflow.log_metric(f"energy/{stage}/total_min", float(arr.min()), step=epoch)
            mlflow.log_metric(f"energy/{stage}/total_max", float(arr.max()), step=epoch)
    except Exception as e:
        print(f"Error logging total energy distribution to MLflow: {e}")


def run(
    model: torch.nn.Module,
    data: ComplexDataModule,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train: bool,
    use_profiler: bool = False,
):
    if train:
        model.train()
        loaders = data.train_dataloader()
    else:
        model.eval()
        loaders = data.val_dataloader()

    tasks = list(loaders.keys())
    
    with record_function("data_loading_and_processing"):
        for batch in tqdm(zip(*(loaders[task] for task in tasks))):
            with record_function("batch_preparation"):
                batch = dict(zip(tasks, batch))
                batch = {task: batch[task].to(device) for task in batch}

            if train:
                with record_function("training_step"):
                    model.zero_grad()
                    with record_function("forward_pass"):
                        loss_total = model.training_step(batch)
                    with record_function("backward_pass"):
                        loss_total.backward()
                    with record_function("optimizer_step"):
                        optimizer.step()
            else:
                with record_function("validation_step"):
                    with torch.no_grad():
                        model.validation_step(batch)


@hydra.main(version_base=None, config_path="../config", config_name="config_train")
def main(config: DictConfig):
    logger = utils.initialize_logger(config.run.log_file)
    logger.info(f"Current working directory: {os.getcwd()}")

    os.makedirs(config.run.checkpoint_dir, exist_ok=True)
    os.makedirs(config.run.tensorboard_dir, exist_ok=True)

    # Set GPUs.
    gpu_idx = utils.cuda_visible_devices(config.run.ngpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint if exists.
    if config.run.restart_file:
        checkpoint = torch.load(config.run.restart_file, map_location=device)
        config = utils.merge_configs(checkpoint["config"], config)
        logger.info(f"Restart from: {os.path.realpath(config.run.restart_file)}")
    else:
        checkpoint = None

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"device: {repr(device)}, gpu_idx: {gpu_idx}")

    # Set a seed for reproducibility.
    if config.run.seed is not None:
        utils.seed(config.run.seed)
        logger.warning("WARNING: Currently, manual seeding does not guarantee reproducibility!")

    data = ComplexDataModule(config)
    model, last_epoch = utils.initialize_state(device, checkpoint, config, data.num_features)
    optimizer = model.configure_optimizers()
    if checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state from checkpoint: {e}. Using freshly initialized optimizer.")

    for task in data.tasks:
        if dir_path := config.data[task].processed_data_dir:
            logger.info(f"Using processed data for '{task}' from: {os.path.realpath(dir_path)}")

    logger.info("Number of data: training | test")
    for task, (len_train, len_test) in data.size.items():
        msg = f"\t'{task}': {len_train} | {len_test}"
        if (n_samples := getattr(config.data[task], "n_samples", 0)) > 0:
            len_actual_train, len_actual_test = data.approximate_size(task)
            msg += f" Sampled {n_samples} per PDB"
            msg += f" -> Approximately {len_actual_train} | {len_actual_test}"
        logger.info(msg)

    logger.info(f"Number of parameters: {model.size[0]}")
    writer = SummaryWriter(config.run.tensorboard_dir)

    # MLflow setup
    try:
        mlflow.set_tracking_uri(config.run.mlflow_uri)
        mlflow.set_experiment(config.run.experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow: {e}")

    history = {
    "epoch": [],
    "train_loss_total": [],
    "test_loss_total": [],
    "train_r2": [],
    "test_r2": [],
    }

    with mlflow.start_run(run_name=config.run.run_name):
        try:
            mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        except Exception as e:
            print(f"Error logging parameters to MLflow: {e}")

        # PyTorch Profiler setup
        profiler_enabled = getattr(config.run, 'enable_profiler', False)
        profiler_epochs = getattr(config.run, 'profiler_epochs', [1, 2])  # Profile first 2 epochs by default
        profiler_output_dir = getattr(config.run, 'profiler_output_dir', './profiler_output')
        
        os.makedirs(profiler_output_dir, exist_ok=True)
        
        # Configure profiler
        profiler_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            profiler_activities.append(ProfilerActivity.CUDA)
            
        profiler_context = None
        if profiler_enabled:
            logger.info(f"Profiler enabled for epochs: {profiler_epochs}")
            logger.info(f"Profiler output directory: {profiler_output_dir}")

        # Track best model based on test loss for the scoring task (or first available task)
        best_epoch = -1
        best_metric = float("inf")
        best_ckpt_path = None
        best_metric_task = None

        for epoch in range(last_epoch + 1, config.run.num_epochs + 1):
            start_time = time.time()
            data.sample_keys()

            # Start profiler for specific epochs
            should_profile = profiler_enabled and epoch in profiler_epochs
            if should_profile:
                profiler_context = profile(
                    activities=profiler_activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    on_trace_ready=lambda trace: trace.export_chrome_trace(
                        os.path.join(profiler_output_dir, f"epoch_{epoch}_trace.json")
                    )
                )
                profiler_context.__enter__()
                logger.info(f"Starting profiler for epoch {epoch}")

            model.reset_log()
            run(model, data, device, optimizer, True, should_profile)
            train_losses = utils.get_losses(model)

            task_name = "scoring"
            if task_name not in model.predictions:
                task_name = list(model.predictions.keys())[0]
            train_r, train_r2, train_tau = utils.get_stats(model, task_name)
            utils.write_predictions(model, config, True)
            # Log energy distribution for training predictions
            try:
                log_energy_distribution_to_mlflow(model, config, stage="train", epoch=epoch)
            except Exception as e:
                print(f"Error logging train energy distribution: {e}")

            model.reset_log()
            run(model, data, device, optimizer, False, should_profile)
            test_losses = utils.get_losses(model)
            test_r, test_r2, test_tau = utils.get_stats(model, task_name)
            utils.write_predictions(model, config, False)
            # Log energy distribution for validation/test predictions
            try:
                log_energy_distribution_to_mlflow(model, config, stage="test", epoch=epoch)
            except Exception as e:
                print(f"Error logging test energy distribution: {e}")

            # Update best checkpoint based on test loss of the scoring task (fallback to first task)
            metric_task = "scoring" if "scoring" in test_losses else None
            if metric_task is None:
                for k in test_losses.keys():
                    if k != "dvdw":
                        metric_task = k
                        break
            if metric_task is not None:
                current_metric = test_losses[metric_task]
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_metric_task = metric_task
                    best_ckpt_path = os.path.join(config.run.checkpoint_dir, "save_best.pt")
                    utils.save_state(best_ckpt_path, epoch, model, optimizer)
                    try:
                        mlflow.log_metric(f"best_test_loss_{metric_task}", best_metric, step=epoch)
                        mlflow.log_metric("best_epoch_so_far", best_epoch, step=epoch)
                    except Exception as e:
                        print(f"Error logging best-so-far metrics to MLflow: {e}")

            # Stop profiler and save results
            if should_profile and profiler_context:
                profiler_context.__exit__(None, None, None)
                
                # Export additional profiler outputs
                try:
                    # Save detailed profiler table
                    profile_table = profiler_context.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                    with open(os.path.join(profiler_output_dir, f"epoch_{epoch}_profile_table.txt"), "w") as f:
                        f.write(profile_table)
                    
                    # Save profiler statistics by input shapes
                    profile_shapes = profiler_context.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20)
                    with open(os.path.join(profiler_output_dir, f"epoch_{epoch}_profile_shapes.txt"), "w") as f:
                        f.write(profile_shapes)
                        
                    logger.info(f"Profiler data saved to {profiler_output_dir}/epoch_{epoch}_*")
                    print(f"\nTop operations by CUDA time for epoch {epoch}:")
                    print(profiler_context.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    
                except Exception as e:
                    logger.error(f"Error saving profiler data: {e}")

            end_time = time.time()

            if epoch == last_epoch + 1:
                logger.info(utils.get_log_line(data.tasks, title=True))

            log_elements = [
                str(epoch),
                utils.get_log_line(data.tasks, train_losses),
                utils.get_log_line(data.tasks, test_losses),
                f"{train_r:.3f}",
                f"{test_r:.3f}",
                f"{train_tau:.3f}",
                f"{test_tau:.3f}",
                f"{end_time - start_time:.3f}",
            ]
            logger.info("\t".join(log_elements))

            writer.add_scalars("training loss", train_losses, epoch)
            writer.add_scalars("test loss", test_losses, epoch)
            writer.add_scalar("R2/train", train_r2, epoch)
            writer.add_scalar("R2/test", test_r2, epoch)
            writer.add_scalar("R/train", train_r, epoch)
            writer.add_scalar("R/test", test_r, epoch)
            writer.add_scalar("tau/train", train_tau, epoch)
            writer.add_scalar("tau/test", test_tau, epoch)

            # Log to MLflow
            try:
                for k, v in train_losses.items():
                    mlflow.log_metric(f"train_loss_{k}", v, step=epoch)
                for k, v in test_losses.items():
                    mlflow.log_metric(f"test_loss_{k}", v, step=epoch)
                mlflow.log_metric("R2/train", train_r2, step=epoch)
                mlflow.log_metric("R2/test", test_r2, step=epoch)
                mlflow.log_metric("R/train", train_r, step=epoch)
                mlflow.log_metric("R/test", test_r, step=epoch)
                mlflow.log_metric("tau/train", train_tau, step=epoch)
                mlflow.log_metric("tau/test", test_tau, step=epoch)
                mlflow.log_metric("epoch_time", end_time - start_time, step=epoch)
            except Exception as e:
                print(f"Error logging to MLflow: {e}")

            history["epoch"].append(epoch)
            history["train_loss_total"].append(train_losses.get("total", 0))
            history["test_loss_total"].append(test_losses.get("total", 0))
            history["train_r2"].append(train_r2)
            history["test_r2"].append(test_r2)

            if epoch == 1 or epoch % 50 == 0:
                save_path = os.path.join(config.run.checkpoint_dir, f"save_{epoch}.pt")
                utils.save_state(save_path, epoch, model, optimizer)
    
        try:
            # Log best checkpoint and metrics at the end
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                try:
                    mlflow.log_metric("best_epoch", best_epoch)
                    if best_metric_task is not None:
                        mlflow.log_metric(f"best_test_loss_{best_metric_task}", best_metric)
                    mlflow.log_artifact(best_ckpt_path, artifact_path="best_checkpoint")
                except Exception as e:
                    print(f"Error logging best checkpoint to MLflow: {e}")

            mlflow.pytorch.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"Error logging model to MLflow: {e}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()

