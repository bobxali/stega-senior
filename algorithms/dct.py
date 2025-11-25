
import numpy as np
from PIL import Image
import cv2
from algorithms.capacity import StegoCapacity

BLOCK_SIZE = 8
REDUNDANCY = 9  # blocks per bit (majority vote) - higher for more robustness
C1_POS = (1, 2)
C2_POS = (2, 1)
MARGIN = 15.0


def _to_bits(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def _from_bits(bitstr: str) -> bytes:
    return bytes(int(bitstr[i : i + 8], 2) for i in range(0, len(bitstr), 8))


def _majority(chunk):
    ones = sum(chunk)
    zeros = len(chunk) - ones
    return 1 if ones >= zeros else 0


def encode_pil(img: Image.Image, payload: bytes) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    available = StegoCapacity.dct_bytes(img, redundancy=REDUNDANCY)
    if len(payload) > available:
        raise ValueError(f"Payload too large for image capacity ({available} bytes available)")

    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    data = len(payload).to_bytes(4, "big") + payload
    bits = _to_bits(data)
    needed_blocks = len(bits) * REDUNDANCY
    blocks_available = (h // BLOCK_SIZE) * (w // BLOCK_SIZE)
    if blocks_available < needed_blocks:
        raise ValueError("Payload too large after redundancy is applied")

    block_idx = 0
    h8 = (h // BLOCK_SIZE) * BLOCK_SIZE
    w8 = (w // BLOCK_SIZE) * BLOCK_SIZE
    for by in range(0, h8, BLOCK_SIZE):
        for bx in range(0, w8, BLOCK_SIZE):
            bit_idx = block_idx // REDUNDANCY
            if bit_idx >= len(bits):
                break

            block = y[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE]
            dct = cv2.dct(block)

            c1y, c1x = C1_POS
            c2y, c2x = C2_POS
            c1 = dct[c1y, c1x]
            c2 = dct[c2y, c2x]

            # Avoid near-zero coefficients
            if abs(c1) < 1e-3:
                c1 = 1.0
            if abs(c2) < 1e-3:
                c2 = 1.0

            bit = int(bits[bit_idx])
            if bit == 1:
                if c1 - c2 < MARGIN:
                    diff = MARGIN - (c1 - c2)
                    c1 += diff / 2.0
                    c2 -= diff / 2.0
            else:
                if c2 - c1 < MARGIN:
                    diff = MARGIN - (c2 - c1)
                    c2 += diff / 2.0
                    c1 -= diff / 2.0

            dct[c1y, c1x] = c1
            dct[c2y, c2x] = c2

            idct = cv2.idct(dct)
            y[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE] = idct

            block_idx += 1
        if bit_idx >= len(bits):
            break

    ycrcb[:, :, 0] = np.rint(np.clip(y, 0, 255)).astype(np.uint8)
    out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def decode_pil(img: Image.Image) -> bytes:
    img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    h8 = (h // BLOCK_SIZE) * BLOCK_SIZE
    w8 = (w // BLOCK_SIZE) * BLOCK_SIZE
    raw_bits = []
    compact_bits = []
    payload_len = None
    total_bits_needed = 32  # start with header

    for by in range(0, h8, BLOCK_SIZE):
        for bx in range(0, w8, BLOCK_SIZE):
            block = y[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE]
            dct = cv2.dct(block)
            c1 = dct[C1_POS[0], C1_POS[1]]
            c2 = dct[C2_POS[0], C2_POS[1]]
            raw_bits.append(1 if (c1 - c2) >= 0 else 0)

            # Every REDUNDANCY blocks, commit a bit
            if len(raw_bits) % REDUNDANCY == 0:
                chunk = raw_bits[-REDUNDANCY:]
                compact_bits.append(str(_majority(chunk)))

                if payload_len is None and len(compact_bits) >= 32:
                    payload_len = int("".join(compact_bits[:32]), 2)
                    total_bits_needed = 32 + payload_len * 8

                if payload_len is not None and len(compact_bits) >= total_bits_needed:
                    break
        if payload_len is not None and len(compact_bits) >= total_bits_needed:
            break

    if len(compact_bits) < 32:
        raise ValueError("Encoded payload too short (no length header)")
    if payload_len is None:
        payload_len = int("".join(compact_bits[:32]), 2)
        total_bits_needed = 32 + payload_len * 8
    if len(compact_bits) < total_bits_needed:
        raise ValueError("Encoded payload incomplete")

    bitstream = "".join(compact_bits[:total_bits_needed])
    payload_bits = bitstream[32:total_bits_needed]
    return _from_bits(payload_bits)
