"""Select a real scan or explicitly request historical identity behavior."""
import os
import warnings

SCAN_BACKEND = os.environ.get("CEM_SCAN_BACKEND", "official")
if SCAN_BACKEND == "legacy_identity":
    warnings.warn("LEGACY IDENTITY SCAN: historical test behavior, not real Mamba.", RuntimeWarning)
    from compat.legacy_scan import selective_scan_fn
elif SCAN_BACKEND == "official":
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError as exc:
        raise ImportError("Install the official mamba-ssm CUDA dependency; see README.md. "
                          "No identity fallback is enabled.") from exc
else:
    raise ValueError("CEM_SCAN_BACKEND must be official or legacy_identity")
