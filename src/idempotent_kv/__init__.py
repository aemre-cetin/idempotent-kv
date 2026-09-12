"""
idempotent-kv: Universal Zero-Copy In-Place Memory Compactor for Transformer & SSM/Mamba.

Protected under U.S. Patent Application Nos. 64/148,668, 64/149,520 ("Patent Pending").
Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
"""

__version__ = "0.2.2"
__author__ = "Dr. A. Emre ÇETİN"
__patent__ = "U.S. Patent Application No. 64/148,668, 64/149,520 (Patent Pending)"

from .kernel import compact_kv_cache_inplace
from .compactor import InplaceKVCompactor, AutoContextEngine, ContextType
from .tarski_state import TarskiStateCompactor
from .integrations import VLLMInplaceCompactionHook, SGLangInplaceCompactionHook

__all__ = [
    "compact_kv_cache_inplace",
    "InplaceKVCompactor",
    "AutoContextEngine",
    "ContextType",
    "TarskiStateCompactor",
    "VLLMInplaceCompactionHook",
    "SGLangInplaceCompactionHook",
    "__version__",
    "__patent__",
]
