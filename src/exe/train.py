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

# isort: off
import path
import utils
from data import ComplexDataModule
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import io

def log_plot_to_mlflow(fig, artifact_name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    mlflow.log_image(buf, artifact_file=artifact_name)
    plt.close(fig)


def run(
    model: torch.nn.Module,
    data: ComplexDataModule,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train: bool,
):
    if train:
        model.train()
        loaders = data.train_dataloader()
    else:
        model.eval()
        loaders = data.val_dataloader()

    tasks = list(loaders.keys())
    for batch in tqdm(zip(*(loaders[task] for task in tasks))):
        batch = dict(zip(tasks, batch))
        batch = {task: batch[task].to(device) for task in batch}

        if train:
            model.zero_grad()
            loss_total = model.training_step(batch)
            loss_total.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.validation_step(batch)


@hydra.main(config_path="../config", config_name="config_train")
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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
    mlflow.set_tracking_uri(config.run.mlflow_uri)
    mlflow.set_experiment(config.run.experiment_name)

    history = {
    "epoch": [],
    "train_loss_total": [],
    "test_loss_total": [],
    "train_r2": [],
    "test_r2": [],
    }

    with mlflow.start_run(run_name=config.run.run_name):
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))

        for epoch in range(last_epoch + 1, config.run.num_epochs + 1):
            start_time = time.time()
            data.sample_keys()

            model.reset_log()
            run(model, data, device, optimizer, True)
            train_losses = utils.get_losses(model)

            task_name = "scoring"
            if task_name not in model.predictions:
                task_name = list(model.predictions.keys())[0]
            train_r, train_r2, train_tau = utils.get_stats(model, task_name)
            utils.write_predictions(model, config, True)

            model.reset_log()
            run(model, data, device, optimizer, False)
            test_losses = utils.get_losses(model)
            test_r, test_r2, test_tau = utils.get_stats(model, task_name)
            utils.write_predictions(model, config, False)

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

            history["epoch"].append(epoch)
            history["train_loss_total"].append(train_losses.get("total", 0))
            history["test_loss_total"].append(test_losses.get("total", 0))
            history["train_r2"].append(train_r2)
            history["test_r2"].append(test_r2)

            if config.run.save_every:
                if epoch == 1 or epoch % config.run.save_every == 0:
                    save_path = os.path.join(config.run.checkpoint_dir, f"save_{epoch}.pt")
                    utils.save_state(save_path, epoch, model, optimizer)
                    mlflow.pytorch.log_model(model, artifact_path="model")
        
        # Plot loss curves
        fig, ax = plt.subplots()
        ax.plot(history["epoch"], history["train_loss_total"], label="Train Loss")
        ax.plot(history["epoch"], history["test_loss_total"], label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        log_plot_to_mlflow(fig, "loss_curve.png")

        # Plot R2 curve
        fig, ax = plt.subplots()
        ax.plot(history["epoch"], history["train_r2"], label="Train R²")
        ax.plot(history["epoch"], history["test_r2"], label="Test R²")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title("R² Score")
        ax.legend()
        log_plot_to_mlflow(fig, "r2_score_curve.png")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
