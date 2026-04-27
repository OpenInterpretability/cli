"""openinterp — SDK + CLI for openinterp.org.

Quick start:
    pip install openinterp                     # lightweight — Atlas + CLI
    pip install "openinterp[full]"             # torch/transformers for trace + guard

Use:
    >>> from openinterp import search_features, FabricationGuard
    >>> features = search_features("overconfidence")
    >>> guard = FabricationGuard.from_pretrained("Qwen/Qwen3.6-27B")

CLI:
    $ openinterp atlas "overconfidence"
    $ openinterp trace --model Qwen/Qwen3.6-27B \\
                       --sae-repo caiovicentino1/qwen36-27b-sae-multilayer \\
                       --prompt "A 52-year-old..." \\
                       --layer 31
    $ openinterp guard --model Qwen/Qwen3.6-27B \\
                       --prompt "Who is Bambale Osby?" \\
                       --mode abstain
"""
from openinterp.atlas import search_features, get_feature
from openinterp.trace import generate_trace, upload_trace, TraceUnavailable
from openinterp.guard import FabricationGuard, FabricationGuardError, GuardOutput
from openinterp.models import AtlasFeature, Trace, TraceFeature

__version__ = "0.2.0"
__author__ = "Caio Vicentino"
__license__ = "Apache-2.0"

__all__ = [
    "search_features",
    "get_feature",
    "generate_trace",
    "upload_trace",
    "TraceUnavailable",
    "FabricationGuard",
    "FabricationGuardError",
    "GuardOutput",
    "AtlasFeature",
    "Trace",
    "TraceFeature",
    "__version__",
]
