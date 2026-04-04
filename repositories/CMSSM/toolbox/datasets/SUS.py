import torch

from .wild import Wild


class SUS(Wild):
    """SUS adapter with compatibility targets expected by legacy training code."""

    def __init__(self, cfg, mode="train", do_aug=True):
        super().__init__(cfg, mode=mode, do_aug=do_aug)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        label = sample["label"]

        # Legacy scripts expect these keys during SUS training.
        binary = (label > 0).long()
        sample["boundary"] = binary.clone()
        sample["binary"] = binary

        # Keep aliases for cross-script compatibility.
        sample["bound"] = sample["boundary"]
        sample["binary_label"] = sample["binary"]
        return sample
