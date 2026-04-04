from .wild import Wild


class FMB(Wild):
    """Thin adapter: FMB follows the same RGB-T split format as Wild/CART."""

    def __init__(self, cfg, mode="train", do_aug=True):
        super().__init__(cfg, mode=mode, do_aug=do_aug)
