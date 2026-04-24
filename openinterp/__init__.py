"""openinterp — SDK + CLI for openinterp.org.

Quick start:
    pip install openinterp                     # lightweight — Atlas + CLI
    pip install "openinterp[full]"             # with torch/transformers for trace

Use:
    >>> from openinterp import search_features
    >>> features = search_features("overconfidence")

CLI:
    $ openinterp atlas "overconfidence"
    $ openinterp trace --model Qwen/Qwen3.6-27B \\
                       --sae-repo caiovicentino1/qwen36-27b-sae-multilayer \\
                       --prompt "A 52-year-old..." \\
                       --layer 31
"""
from openinterp.atlas import search_features, get_feature
from openinterp.trace import generate_trace, upload_trace, TraceUnavailable
from openinterp.models import AtlasFeature, Trace, TraceFeature

__version__ = "0.1.0"
__author__ = "Caio Vicentino"
__license__ = "Apache-2.0"

__all__ = [
    "search_features",
    "get_feature",
    "generate_trace",
    "upload_trace",
    "TraceUnavailable",
    "AtlasFeature",
    "Trace",
    "TraceFeature",
    "__version__",
]
