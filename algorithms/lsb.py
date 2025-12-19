
# Import Pillow image class for pixel access
from typing import Optional
from PIL import Image
# Import capacity helpers for size checks
from algorithms.capacity import StegoCapacity
from algorithms.key_utils import shuffled_indices

# Delimiter kept for backward-compatible decoding; new writes use length header
DELIMITER = b'<<END>>'


# Convert bytes to a string of bits
def _to_bits(data: bytes) -> str:
    # Join each byte formatted as 8-bit binary
    return ''.join(f'{byte:08b}' for byte in data)


# Convert a bitstring back to bytes
def _from_bits(bitstr: str) -> bytes:
    # Slice bits into bytes and convert each to int
    return bytes(int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8))


# Encode payload into an image using 1 LSB per channel
def encode_pil(img: Image.Image, payload: bytes, key: Optional[str] = None) -> Image.Image:
    # Ensure RGB mode for three channels
    img = img.convert('RGB')
    # Compute available capacity for payload (includes 4-byte header reserve)
    available = StegoCapacity.lsb_bytes(img, len(DELIMITER))
    # Fail if payload exceeds capacity
    if len(payload) > available:
        raise ValueError(f"Payload too large for image capacity ({available} bytes available)")
    # Prepend 4-byte length header (length-prefixed to support binary data)
    data = len(payload).to_bytes(4, "big") + payload
    bits = _to_bits(data)  # Flatten data to a bitstring
    w,h = img.size  # width, height in pixels
    pixels = list(img.getdata())  # Read pixels as a flat list
    new_pixels = pixels.copy()  # Mutable copy for in-place updates
    total_slots = len(pixels) * 3  # three channels per pixel
    order = range(total_slots)
    if key:
        order = shuffled_indices(total_slots, key, salt="lsb")
    idx = 0  # Bit index tracker
    for pos in order:
        if idx >= len(bits):
            break
        pixel_index = pos // 3
        channel = pos % 3
        r, g, b = new_pixels[pixel_index]
        bit = int(bits[idx]); idx += 1
        if channel == 0:
            r = (r & ~1) | bit
        elif channel == 1:
            g = (g & ~1) | bit
        else:
            b = (b & ~1) | bit
        new_pixels[pixel_index] = (r, g, b)
    # Create new RGB image
    out = Image.new('RGB', (w,h))
    # Write modified pixels to image
    out.putdata(new_pixels)
    # Return encoded image
    return out


# Decode payload from an image using 1 LSB per channel
def decode_pil(img: Image.Image, key: Optional[str] = None) -> bytes:
    # Ensure RGB mode for consistent channel access
    img = img.convert('RGB')
    pixels = list(img.getdata())
    total_slots = len(pixels) * 3
    order = range(total_slots)
    if key:
        order = shuffled_indices(total_slots, key, salt="lsb")

    data_bytes = bytearray()
    bit_buffer = []
    target_len = None  # bytes expected after reading 4-byte header

    for pos in order:
        pixel_index = pos // 3
        channel = pos % 3
        r, g, b = pixels[pixel_index]
        if channel == 0:
            bit_buffer.append(str(r & 1))
        elif channel == 1:
            bit_buffer.append(str(g & 1))
        else:
            bit_buffer.append(str(b & 1))

        if len(bit_buffer) == 8:
            byte_val = int(''.join(bit_buffer), 2)
            data_bytes.append(byte_val)
            bit_buffer.clear()

            if target_len is None and len(data_bytes) >= 4:
                target_len = int.from_bytes(data_bytes[:4], "big")
                # sanity check: avoid absurd lengths
                max_possible = (total_slots // 8) - 4
                if target_len < 0 or target_len > max_possible:
                    target_len = None  # fall back to delimiter method
            if target_len is not None and len(data_bytes) >= 4 + target_len:
                return bytes(data_bytes[4:4 + target_len])

            # Legacy delimiter fallback if header path not used
            if target_len is None and data_bytes.endswith(DELIMITER):
                return bytes(data_bytes[:-len(DELIMITER)])

    # If header path incomplete but delimiter found, return that portion
    if target_len is None and data_bytes.endswith(DELIMITER):
        return bytes(data_bytes[:-len(DELIMITER)])
    return bytes(data_bytes)
