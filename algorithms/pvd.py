
from PIL import Image

RANGES = [(0,7),(8,15),(16,31),(32,63),(64,127),(128,255)]

def _bits_needed(d):
    import math
    for L,U in RANGES:
        if L<=d<=U:
            return int(math.floor(math.log2(U-L+1)))
    return 1

def _to_bits(data: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in data)

def _from_bits(bitstr: str) -> bytes:
    return bytes(int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8))

def encode_pil(img: Image.Image, payload: bytes) -> Image.Image:
    img = img.convert('L')
    bits = _to_bits(payload)
    pixels = list(img.getdata())
    new_pixels = pixels.copy()
    idx = 0
    for i in range(0,len(pixels)-1,2):
        if idx>=len(bits): break
        p1,p2 = pixels[i],pixels[i+1]
        d = abs(p1-p2)
        nbits = _bits_needed(d)
        to_write = bits[idx:idx+nbits]
        if not to_write: break
        v = int(to_write,2)
        for L,U in RANGES:
            if L<=d<=U: d_new=L+v; break
        if p1>=p2: p1n=p1; p2n=max(0,p1n-d_new)
        else: p2n=p2; p1n=max(0,p2n-d_new)
        new_pixels[i], new_pixels[i+1] = p1n,p2n
        idx += nbits
    out = Image.new('L', img.size)
    out.putdata(new_pixels)
    return out

def decode_pil(img: Image.Image) -> bytes:
    img = img.convert('L')
    pixels = list(img.getdata())
    bits=''
    for i in range(0,len(pixels)-1,2):
        p1,p2 = pixels[i],pixels[i+1]
        d=abs(p1-p2)
        nbits=_bits_needed(d)
        for L,U in RANGES:
            if L<=d<=U: v=d-L; bits+=f'{v:0{nbits}b}'; break
    return bytes(int(bits[i:i+8],2) for i in range(0,len(bits)//8*8,8))
