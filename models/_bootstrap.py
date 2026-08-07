"""Make the vendored build of `flash-linear-attention` (which ships `gdn2`)
take precedence over any site-packages `fla` (which lacks gdn2).

Importing this module *before* any `import fla` guarantees the vendored copy
(`third_party/flash-linear-attention`) wins. Keeping the path injection in one
place avoids touching the global environment (no `pip install` into
site-packages required).
"""
from __future__ import annotations

import os
import sys
import warnings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDORED_FLA = os.path.join(_REPO_ROOT, "third_party", "flash-linear-attention")


def ensure_vendored_fla() -> None:
    if not os.path.isdir(_VENDORED_FLA):
        raise FileNotFoundError(
            f"vendored flash-linear-attention not found at {_VENDORED_FLA}"
        )
    if _VENDORED_FLA not in sys.path:
        sys.path.insert(0, _VENDORED_FLA)
    # If a different `fla` was already imported, warn loudly — gdn2 will be missing.
    mod = sys.modules.get("fla")
    if mod is not None and getattr(mod, "__file__", "").startswith(_VENDORED_FLA) is False:
        warnings.warn(
            "`fla` was imported from a non-vendored location before models bootstrap; "
            "gdn2 may be unavailable. Import `models` before `fla`.",
            stacklevel=2,
        )


ensure_vendored_fla()
