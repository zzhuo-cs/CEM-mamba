# Third-party notices

`mambavision/models/mamba_vision.py` derives from [NVIDIA MambaVision](https://github.com/NVlabs/MambaVision). Original copyright notices are retained. The full NVIDIA Source Code License-NC is in `licenses/NVIDIA-MambaVision.txt`, retrieved from the upstream main branch on September 23, 2026. Release packaging changes to scan imports and checkpoint loading are described in `docs/REPRODUCIBILITY.md`.

`mambavision/models/registry.py` identifies its source as the timm model registry/factory by Ross Wightman. That attribution remains intact. The upstream Apache 2.0 license is included in `licenses/Apache-2.0-timm.txt`.

Installed dependencies, including PyTorch, torchvision, timm and mamba-ssm, retain their respective upstream terms. No mamba-ssm source package or binary extension is bundled here. `compat/legacy_scan.py` is the supplied experiment's identity test behavior, not the official Mamba implementation.

Project-specific files have not been assigned a new blanket license. A license for those contributions should be supplied by their rights holders. Including third-party license files does not relicense the full repository under those terms.
