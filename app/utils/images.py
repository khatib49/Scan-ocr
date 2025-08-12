import base64
from io import BytesIO
from PIL import Image

# Resize/compress to cap tokens dramatically

def to_base64_optimized(raw_bytes: bytes, max_side: int = 1400, quality: int = 82) -> str:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")