echo "[SETUP] Installing Python requirements"; \
conda run -n pignet2 pip install --no-cache-dir -r requirements.txt && echo "[SETUP] Requirements installed"; \
echo "[SETUP] Downloading raw PDBbind refined set"; \
bash dataset/preprocess/download_raw.sh && echo "[SETUP] Raw download completed"; \
echo "[SETUP] Generating complexes from raw (with PQR charges)"; \
conda run -n pignet2 python dataset/preprocess/script_pdbbind.py && echo "[SETUP] Complex generation completed"; \
echo "[SETUP] Preprocessing generated data (feature caching)"; \
conda run -n pignet2 python preprocess_data.py && echo "[SETUP] Preprocessing completed"; \
bash experiments/training_scripts/baseline.sh && echo "[SETUP] Training completed";

