# Steganography Studio

Live demo: https://stega-senior.onrender.com/

Flask web app for hiding and revealing data inside images. Supports three algorithms (LSB, PVD, DCT), keyed embedding/decoding, file payloads, an algorithm recommender, and a heuristic steganalysis check.

## Features
- Encode/decode with LSB, PVD, or DCT; optional key for shuffling positions.
- Embed text or arbitrary files; length-prefixed to preserve binary data.
- Capacity check and simple quality metrics (PSNR, SSIM).
- Auto-recommend algorithm based on payload size and image capacity.
- Image library picker with thumbnails and drag/drop upload.
- Steganalysis page with heuristic risk signals (LSB chi-square/RS, DCT asymmetry, residual energy).

## Setup
```bash
git clone https://github.com/bobxali/stega-senior.git
cd stega-senior
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```
Open the shown localhost URL in your browser.

## Usage
- Encode tab: choose payload type (text or file), pick/upload a cover image, optional key, and Encode. Use Recommend/Capacity in Advanced options if needed.
- Decode tab: upload a stego image, optional key, and decode (auto-detect or pick an algorithm).
- Steganalysis tab: upload any image to get heuristic “Low/Medium/High” risk signals (not proof; expect false positives/negatives).
- Library images: add PNG/BMP/JPG files to `images/` and select them from the library list; uploads go to `uploads/`.

## Notes
- Use lossless PNG/BMP for best capacity/quality.
- Keep the same key for encoding/decoding when used.
- Steganalysis is heuristic and cannot guarantee presence/absence of hidden data.
