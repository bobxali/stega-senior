import hashlib
import random
from typing import List


def _seed_from_key(key: str, salt: str) -> int:
    """
    Deterministic 64-bit seed derived from user key and a per-algorithm salt.
    Salt keeps streams distinct across algorithms.
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def shuffled_indices(count: int, key: str, salt: str) -> List[int]:
    """
    Return a shuffled list of indices [0, count) using a deterministic PRNG.
    """
    order = list(range(count))
    random.Random(_seed_from_key(key, salt)).shuffle(order)
    return order
