
from PIL import Image
from algorithms.capacity import RANGES, StegoCapacity

# Pairs of differences and corresponding widths


def _bits_needed(d: int) -> int:
    import math
    for L, U in RANGES:
        if L <= d <= U:
            return int(math.floor(math.log2(U - L + 1)))
    return 1


def _to_bits(data: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in data)


def _from_bits(bitstr: str) -> bytes:
    return bytes(int(bitstr[i : i + 8], 2) for i in range(0, len(bitstr), 8))


def _embed_pair(p1: int, p2: int, d_new: int) -> tuple[int, int]:
    """
    Adjust both pixels to match the target difference as closely as possible,
    preserving which pixel was originally larger.
    """
    def adjust(a, b, target_diff):
        # ensure a >= b
        a = int(a); b = int(b)
        # initial attempt
        a_new = min(255, max(0, a))
        b_new = max(0, min(255, a_new - target_diff))
        # if diff too small, try to increase it
        if a_new - b_new < target_diff:
            need = target_diff - (a_new - b_new)
            add = min(need, 255 - a_new)
            a_new += add
            need -= add
            if need > 0:
                b_new = max(0, b_new - need)
        # if diff too large, reduce it
        if a_new - b_new > target_diff:
            reduce = (a_new - b_new) - target_diff
            add = min(reduce, 255 - b_new)
            b_new += add
            reduce -= add
            if reduce > 0:
                a_new = max(0, a_new - reduce)
        return a_new, b_new

    if p1 >= p2:
        a, b = adjust(p1, p2, d_new)
        return a, b
    a, b = adjust(p2, p1, d_new)
    return b, a


def encode_pil(img: Image.Image, payload: bytes) -> Image.Image:
    # Preserve color by embedding only in the green channel
    img_rgb = img.convert("RGB")
    r, g, b = img_rgb.split()
    channel = list(g.getdata())

    data_with_len = len(payload).to_bytes(4, "big") + payload
    bits = _to_bits(data_with_len)

    cap = StegoCapacity.pvd_bytes(img_rgb) * 8
    if len(bits) > cap:
        raise ValueError(f"Payload too large for image capacity ({cap // 8} bytes available)")

    new_channel = channel.copy()
    idx = 0
    for i in range(0, len(channel) - 1, 2):
        if idx >= len(bits):
            break
        p1, p2 = channel[i], channel[i + 1]
        d = abs(p1 - p2)
        nbits = _bits_needed(d)
        chunk = bits[idx : idx + nbits]
        if len(chunk) < nbits:
            chunk = chunk.ljust(nbits, "0")
        v = int(chunk, 2)
        for L, U in RANGES:
            if L <= d <= U:
                d_new = L + v
                break
        p1n, p2n = _embed_pair(p1, p2, d_new)
        new_channel[i], new_channel[i + 1] = p1n, p2n
        idx += nbits

    new_g = Image.new("L", img_rgb.size)
    new_g.putdata(new_channel)
    encoded = Image.merge("RGB", (r, new_g, b))
    return encoded


def decode_pil(img: Image.Image) -> bytes:
    img_rgb = img.convert("RGB")
    channel = list(img_rgb.getchannel("G").getdata())

    bits = ""
    need_bits = 32  # header first
    have_len = False
    for i in range(0, len(channel) - 1, 2):
        p1, p2 = channel[i], channel[i + 1]
        d = abs(p1 - p2)
        nbits = _bits_needed(d)
        for L, U in RANGES:
            if L <= d <= U:
                v = d - L
                bits += f"{v:0{nbits}b}"
                break
        if not have_len and len(bits) >= 32:
            payload_len = int(bits[:32], 2)
            need_bits = 32 + payload_len * 8
            have_len = True
        if have_len and len(bits) >= need_bits:
            break

    if len(bits) < 32:
        raise ValueError("Encoded payload too short (no length header)")

    payload_len = int(bits[:32], 2)
    total_bits_needed = 32 + payload_len * 8
    if len(bits) < total_bits_needed:
        raise ValueError("Encoded payload incomplete")

    payload_bits = bits[32:total_bits_needed]
    return _from_bits(payload_bits)
