"""
idempotent-kv: Zero-Copy In-Place KV-Cache Compactor for LLM Inference Engines.

Protected under U.S. Patent Application No. 64/148,668 ("Patent Pending").
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

__version__ = "0.2.0"
__author__ = "Dr. A. Emre ÇETİN"
__patent__ = "U.S. Patent Application No. 64/148,668 (Patent Pending)"

from .kernel import compact_kv_cache_inplace
from .compactor import InplaceKVCompactor

__all__ = [
    "compact_kv_cache_inplace",
    "InplaceKVCompactor",
    "__version__",
    "__patent__"
]
