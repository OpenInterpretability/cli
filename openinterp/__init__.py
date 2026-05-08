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
from openinterp.agent_probe_guard import AgentProbeGuard, AgentProbeGuardError, Decision
from openinterp.models import AtlasFeature, Trace, TraceFeature
from openinterp.lora import safe_load_qwen36_lora, strip_language_model_infix, LoRAVerificationError
from openinterp import probebench

__version__ = "0.3.1"
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
    "AgentProbeGuard",
    "AgentProbeGuardError",
    "Decision",
    "AtlasFeature",
    "Trace",
    "TraceFeature",
    "safe_load_qwen36_lora",
    "strip_language_model_infix",
    "LoRAVerificationError",
    "probebench",
    "__version__",
]
