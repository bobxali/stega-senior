import numpy as np
import cv2
from PIL import Image

# Simple, heuristic steganalysis. These signals are not definitive; they only estimate risk.


def _chi_square_lsb(img: np.ndarray) -> float:
    """Chi-square statistic on LSBs of all channels, normalized to [0,1]."""
    lsb = img & 1
    zeros = np.sum(lsb == 0)
    ones = np.sum(lsb == 1)
    total = zeros + ones
    if total == 0:
        return 0.0
    expected = total / 2.0
    chi = ((zeros - expected) ** 2 + (ones - expected) ** 2) / expected
    # Normalize: small chi => uniform => could be suspicious; we invert and clamp
    norm = 1.0 - np.exp(-chi / 10.0)
    return float(np.clip(norm, 0.0, 1.0))


def _rs_analysis(img: np.ndarray) -> float:
    """
    Very lightweight RS estimate on green channel.
    Returns a score in [0,1]; higher means more suspicious.
    """
    g = img[:, :, 1].astype(np.int32)
    h, w = g.shape
    # Take 2x2 blocks
    g = g[: h - (h % 2), : w - (w % 2)]
    diffs = []
    for dy in (0, 1):
        for dx in (0, 1):
            if dy == 0 and dx == 0:
                continue
            diffs.append(np.abs(g[dy::2, dx::2] - g[0::2, 0::2]))
    diff = np.stack(diffs, axis=0)
    mean_diff = diff.mean(axis=0)
    # Flip mask on reference and recompute
    flipped = g.copy()
    flipped[0::2, 0::2] ^= 1
    diffs_f = []
    for dy in (0, 1):
        for dx in (0, 1):
            if dy == 0 and dx == 0:
                continue
            diffs_f.append(np.abs(flipped[dy::2, dx::2] - flipped[0::2, 0::2]))
    diff_f = np.stack(diffs_f, axis=0)
    mean_diff_f = diff_f.mean(axis=0)
    reg = (mean_diff > mean_diff_f).sum()
    sing = (mean_diff < mean_diff_f).sum()
    total = reg + sing + 1e-9
    imbalance = abs(reg - sing) / total
    return float(np.clip(imbalance, 0.0, 1.0))


def _dct_asymmetry(img: np.ndarray) -> float:
    """
    Check sign imbalance of mid-frequency DCT coefficients on luminance.
    Returns [0,1]; higher means more imbalance (potential DCT embedding).
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    h, w = y.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    y = y[:h8, :w8]
    blocks = y.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)
    signs = []
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            b = blocks[by, bx]
            c = cv2.dct(b)
            # mid-band coefficients (skip DC and very low/high)
            coeffs = [c[1, 2], c[2, 1], c[2, 2], c[3, 1], c[1, 3]]
            for v in coeffs:
                if abs(v) < 1e-3:
                    continue
                signs.append(1 if v > 0 else -1)
    if not signs:
        return 0.0
    signs = np.array(signs)
    pos = np.sum(signs > 0)
    neg = np.sum(signs < 0)
    total = pos + neg
    if total == 0:
        return 0.0
    imbalance = abs(pos - neg) / total
    return float(np.clip(imbalance, 0.0, 1.0))


def _residual_energy(img: np.ndarray) -> float:
    """
    Laplacian residual energy normalized; higher can indicate tampering/noise.
    Returns [0,1] after a cap.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = float(np.mean(lap ** 2))
    # Soft normalization: most natural images fall below ~0.002
    norm = energy / 0.005
    return float(np.clip(norm, 0.0, 1.0))


def _risk_label(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Medium"
    return "High"


def analyze_pil(img: Image.Image) -> dict:
    """
    Run heuristic steganalysis on a PIL image.
    Returns per-test scores and an overall risk label.
    """
    rgb = np.array(img.convert("RGB"))
    scores = {
        "lsb_chi": _chi_square_lsb(rgb),
        "rs": _rs_analysis(rgb),
        "dct_asym": _dct_asymmetry(rgb),
        "residual": _residual_energy(rgb),
    }
    # Overall score: average of signals (clamped)
    overall = float(np.clip(np.mean(list(scores.values())), 0.0, 1.0))
    return {
        "scores": scores,
        "labels": {k: _risk_label(v) for k, v in scores.items()},
        "overall": overall,
        "label": _risk_label(overall),
    }
