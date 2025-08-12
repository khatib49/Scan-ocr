from typing import Dict

# Central place to log & tune usage

def merge_usage(u1: Dict, u2: Dict) -> Dict:
    if not u1: return u2 or {}
    if not u2: return u1 or {}
    out = dict(u1)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        out[k] = int(out.get(k, 0)) + int(u2.get(k, 0))
    return out