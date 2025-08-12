from typing import Optional

try:
    from PIL import Image
    from pyzbar.pyzbar import decode as qr_decode
except ImportError:
    Image = None
    qr_decode = None

def decode_zatca_qr(image_bytes: bytes) -> Optional[dict]:
    if not (Image and qr_decode):
        return None

    try:
        image = Image.open(BytesIO(image_bytes))
        qrs = qr_decode(image)
        if not qrs:
            return None

        qr_data = qrs[0].data
        if not qr_data:
            return None

        raw = qr_data.decode("utf-8", errors="ignore")
        if raw.startswith('{'):
            return json.loads(raw)

        from base64 import b64decode
        data = b64decode(qr_data)
        out = {}

        i = 0
        while i < len(data):
            tag = data[i]
            i += 1
            length = data[i]
            i += 1
            value = data[i:i+length].decode("utf-8", errors="ignore")
            i += length

            if tag == 1:
                out["seller"] = value
            elif tag == 2:
                out["vat"] = value
            elif tag == 3:
                out["timestamp"] = value
            elif tag == 4:
                out["total"] = try_float(value)
            elif tag == 5:
                out["vat_amount"] = try_float(value)

        return out if out else None
    except Exception:
        return None

def try_float(x):
    try:
        return float(str(x).replace(",", "").replace("SAR", ""))
    except:
        return None
