#!/bin/bash

URL="https://thesis.rayhan.id/thesis/PDBbind_v2020_refined.tar.gz"
FILE="PDBbind_v2020_refined.tar.gz"
OUTPUT_DIR="PDBbind_v2020_refined"

echo "Downloading $FILE from $URL..."
curl -L "$URL" -o "$FILE" || { echo "Download failed."; exit 1; }
echo "Download complete."

echo "Extracting $FILE..."
mkdir -p "$OUTPUT_DIR"
tar -xzf "$FILE" -C "$OUTPUT_DIR" || { echo "Extraction failed."; exit 1; }

# ✨ New line: show the full path to the extracted folder
echo "Extraction complete. Files are in '$(realpath "$OUTPUT_DIR")/'"
