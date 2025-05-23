echo "[SETUP] Installing Python requirements"; \
conda run -n pignet2 pip install --no-cache-dir -r /PIGNet2/requirements.txt && echo "[SETUP] Requirements installed"; \
echo "[SETUP] Downloading dataset"; \
bash /PIGNet2/dataset/download.sh && echo "[SETUP] Download completed"; \
bash /PIGNet2/dataset/untar.sh && echo "[SETUP] Decompression completed"
