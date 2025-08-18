echo "[SETUP] Installing Python requirements"; \
conda run -n pignet2 pip install --no-cache-dir -r requirements.txt && echo "[SETUP] Requirements installed"; \
echo "[SETUP] Downloading raw PDBbind refined set"; \
pushd dataset/preprocess >/dev/null; \
bash download_raw.sh; status=$?; popd >/dev/null; \
if [ $status -ne 0 ]; then echo "[ERROR] Raw download failed"; exit 1; fi; echo "[SETUP] Raw download completed"; \
echo "[SETUP] Generating complexes from raw (with PQR charges)"; \
pushd dataset/preprocess >/dev/null; \
conda run -n pignet2 python script_pdbbind.py; status=$?; popd >/dev/null; \
if [ $status -ne 0 ]; then echo "[ERROR] Complex generation failed"; exit 1; fi; echo "[SETUP] Complex generation completed"; \
echo "[SETUP] Preprocessing generated data (feature caching)"; \
conda run -n pignet2 python preprocess_data.py && echo "[SETUP] Preprocessing completed"; \
bash experiments/training_scripts/baseline.sh && echo "[SETUP] Training completed";

