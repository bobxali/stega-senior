
import numpy as np
from PIL import Image
import cv2

def _to_bits(data: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in data)

def encode_pil(img: Image.Image, payload: bytes) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h,w,_ = img_cv.shape
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:,:,0].astype(np.float32)
    bits = _to_bits(payload)+'00000000'
    idx=0
    block_size=8
    for i in range(0,h-block_size+1,block_size):
        for j in range(0,w-block_size+1,block_size):
            block=y[i:i+block_size,j:j+block_size]
            dct = cv2.dct(block)
            if idx<len(bits):
                coef=int(round(dct[3,3]))
                coef=(coef & ~1)|int(bits[idx]); idx+=1
                dct[3,3]=coef
            idct = cv2.idct(dct)
            y[i:i+block_size,j:j+block_size]=idct
            if idx>=len(bits): break
        if idx>=len(bits): break
    ycrcb[:,:,0]=np.clip(y,0,255).astype(np.uint8)
    out_bgr=cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    out_rgb=cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)

def decode_pil(img: Image.Image) -> bytes:
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h,w,_=img_cv.shape
    ycrcb=cv2.cvtColor(img_cv,cv2.COLOR_BGR2YCrCb)
    y=ycrcb[:,:,0].astype(np.float32)
    bits=''
    block_size=8
    for i in range(0,h-block_size+1,block_size):
        for j in range(0,w-block_size+1,block_size):
            block=y[i:i+block_size,j:j+block_size]
            dct=cv2.dct(block)
            coef=int(round(dct[3,3]))
            bits+=str(coef & 1)
    return bytes(int(bits[i:i+8],2) for i in range(0,len(bits)//8*8,8))
