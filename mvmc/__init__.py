"""mvmc public API; retain existing model and checkpoint identities."""

from multid4rt import (
    EncodedScene,
    MultiD4RT,
    MultiD4RTConfig,
    pack_synchronized_images,
)

MVMC = MultiD4RT
MVMCConfig = MultiD4RTConfig

__all__ = [
    "MVMC", "MVMCConfig", "MultiD4RT", "MultiD4RTConfig",
    "EncodedScene", "pack_synchronized_images",
]
