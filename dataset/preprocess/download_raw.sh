#!/bin/bash

URL="https://thesis.rayhan.id/thesis/PDBbind_v2020_refined.tar.gz"
FILE="PDBbind_v2020_refined.tar.gz"
OUTPUT_DIR="PDBbind_v2020_refined"

echo "Downloading $FILE from $URL..."
curl -L "$URL" -o "$FILE"

if [ $? -ne 0 ]; then
    echo "Download failed."
    exit 1
fi

echo "Download complete."

echo "Extracting $FILE..."
mkdir -p "$OUTPUT_DIR"
tar -xzf "$FILE" -C "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Extraction failed."
    exit 1
fi

echo "Extraction complete. Files are in '$OUTPUT_DIR/'"