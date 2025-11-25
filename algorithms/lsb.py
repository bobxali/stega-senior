
from PIL import Image
from algorithms.capacity import StegoCapacity

DELIMITER = b'<<END>>'


def _to_bits(data: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in data)


def _from_bits(bitstr: str) -> bytes:
    return bytes(int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8))


def encode_pil(img: Image.Image, payload: bytes) -> Image.Image:
    img = img.convert('RGB')
    available = StegoCapacity.lsb_bytes(img, len(DELIMITER))
    if len(payload) > available:
        raise ValueError(f"Payload too large for image capacity ({available} bytes available)")
    data = payload + DELIMITER
    bits = _to_bits(data)
    w,h = img.size
    pixels = list(img.getdata())
    new_pixels = []
    idx = 0
    for p in pixels:
        r,g,b = p
        if idx < len(bits): r = (r & ~1) | int(bits[idx]); idx+=1
        if idx < len(bits): g = (g & ~1) | int(bits[idx]); idx+=1
        if idx < len(bits): b = (b & ~1) | int(bits[idx]); idx+=1
        new_pixels.append((r,g,b))
    out = Image.new('RGB', (w,h))
    out.putdata(new_pixels)
    return out


def decode_pil(img: Image.Image) -> bytes:
    img = img.convert('RGB')
    bits = ''
    for r,g,b in img.getdata():
        bits += str(r & 1)
        bits += str(g & 1)
        bits += str(b & 1)
    data_bytes = bytearray()
    for i in range(0,len(bits),8):
        byte = bits[i:i+8]
        if len(byte)<8: break
        data_bytes.append(int(byte,2))
        if data_bytes.endswith(DELIMITER):
            return bytes(data_bytes[:-len(DELIMITER)])
    return bytes(data_bytes)
