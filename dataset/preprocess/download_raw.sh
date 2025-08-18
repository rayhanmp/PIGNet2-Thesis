#!/usr/bin/env bash
set -euo pipefail

URL="https://thesis.rayhan.id/PDBbind_v2020_refined.tar.gz"
FILE="PDBbind_v2020_refined.tar.gz"
OUTPUT_DIR="PDBbind_v2020_refined"

echo "Downloading ${FILE} from ${URL}…"
if [[ -f "$FILE" ]]; then
  echo "File already exists: $FILE (skipping download)"
else
  curl -L "$URL" -o "$FILE" || { echo "Download failed."; exit 1; }
fi
echo "Download complete."

echo "Preparing output dir: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

echo "Extracting ${FILE} → ${OUTPUT_DIR}"
# Strip the top-level folder from the tar to avoid double nesting
tar -xzf "$FILE" -C "$OUTPUT_DIR" --strip-components=1 || { echo "Extraction failed."; exit 1; }

# Sanity checks
if [[ ! -d "${OUTPUT_DIR}/refined-set" ]]; then
  echo "ERROR: refined-set not found in ${OUTPUT_DIR}. The archive layout may have changed."
  echo "Tar contents (top 10) for inspection:"
  tar -tzf "$FILE" | head -n 10
  exit 1
fi

echo "Extraction complete. Files are in '$(realpath "$OUTPUT_DIR")/'"

# Optional: quick visibility check
echo "Example entries:"
ls -d "${OUTPUT_DIR}/refined-set/"* | head -n 5 || true
