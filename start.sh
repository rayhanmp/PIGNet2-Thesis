echo "[SETUP] Installing Python requirements"; \
conda run -n pignet2 pip install --no-cache-dir -r requirements.txt && echo "[SETUP] Requirements installed"; \
echo "[SETUP] Downloading dataset"; \
bash dataset/download.sh && echo "[SETUP] Download completed"; \
bash dataset/untar.sh && echo "[SETUP] Decompression completed"; \
bash experiments/training_scripts/pda_nda.sh && echo "[SETUP] Training completed";

