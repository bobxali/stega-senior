from PIL import Image

# Shared capacity helper for different stego algorithms.

RANGES = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]  # PVD buckets (inclusive)


def _bits_needed(d: int) -> int:
    import math
    for L, U in RANGES:
        if L <= d <= U:
            return int(math.floor(math.log2(U - L + 1)))
    return 1


class StegoCapacity:
    @staticmethod
    def lsb_bytes(img: Image.Image, delimiter_len: int) -> int:
        # 1 bit per channel; reserve header/delimiter bytes (use max for compatibility)
        w, h = img.size
        total_bits = w * h * 3  # 1 bit per channel
        reserve = max(4, delimiter_len)  # prefer length header; keep delimiter for legacy safety
        return max(0, total_bits // 8 - reserve)

    @staticmethod
    def pvd_bytes(img: Image.Image) -> int:
        # Estimate bits for PVD using dynamic bit width per pixel pair
        channel = list(img.convert("RGB").getchannel("G").getdata())
        bits = 0
        for i in range(0, len(channel) - 1, 2):
            d = abs(channel[i] - channel[i + 1])
            bits += _bits_needed(d)
        return bits // 8

    @staticmethod
    def dct_bytes(img: Image.Image, overhead_bytes: int = 4, redundancy: int = 1) -> int:
        # 1 bit per 8x8 block after redundancy and length header
        w, h = img.size
        blocks = (w // 8) * (h // 8)
        total_bits = blocks // max(1, redundancy)  # effective bits after redundancy
        total_bytes = total_bits // 8
        return max(0, total_bytes - overhead_bytes)

    @staticmethod
    def for_all(img: Image.Image, lsb_delimiter_len: int, dct_redundancy: int = 1) -> dict:
        # Convenience helper to compute capacities for UI display
        return {
            "LSB": StegoCapacity.lsb_bytes(img, lsb_delimiter_len),
            "PVD": StegoCapacity.pvd_bytes(img),
            "DCT": StegoCapacity.dct_bytes(img, redundancy=dct_redundancy),
        }
