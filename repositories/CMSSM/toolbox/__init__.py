from .metrics_CART import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger

from .utils import (
    ClassWeight,
    save_ckpt,
    load_ckpt,
    class_to_RGB,
    compute_speed,
    setup_seed,
    group_weight_decay,
    unwrap_logits,
    split_logits_features,
)


def get_dataset(cfg):
    dataset = str(cfg["dataset"]).strip().lower()

    if dataset == "cart":
        from .datasets.wild import Wild

        return Wild(cfg, mode="train"), Wild(cfg, mode="val"), Wild(cfg, mode="test")

    if dataset == "fmb":
        from .datasets.FMB import FMB

        return FMB(cfg, mode="train"), FMB(cfg, mode="test")

    if dataset == "sus":
        from .datasets.SUS import SUS

        return SUS(cfg, mode="train"), SUS(cfg, mode="val"), SUS(cfg, mode="test")

    if dataset == "pst900":
        from .datasets.pst900 import PST900

        return PST900(cfg, mode="train"), PST900(cfg, mode="test")

    raise ValueError(f"Unsupported dataset '{cfg['dataset']}'. Expected one of: CART, FMB, SUS, pst900")


def get_model(cfg):
    model_name = str(cfg.get("model_name", "")).strip().lower()
    n_classes = int(cfg["n_classes"])
    inputs = cfg["inputs"]
    emb_dim = cfg.get("embed_dim", None)  # Override decoder embed dim if specified

    from models.CM_SSM import Model

    def _make(mode, fusion_mode):
        m = Model(mode=mode, n_class=n_classes, inputs=inputs, fusion_mode=fusion_mode)
        if emb_dim is not None:
            # Rebuild decoder with custom embed dim
            from models.decoder.MLP import Decoder_MLP
            channels = {"b0": [16,32,64,128], "b1": [32,64,128,256], "b2": [48,96,192,384],
                        "b3": [64,128,256,512], "l1": [64,128,256,512], "l2": [64,128,256,512]}
            ch = channels.get(mode, [32,64,128,256])
            m.decoder = Decoder_MLP(in_channels=ch, embed_dim=emb_dim, num_classes=n_classes)
        return m

    if model_name == "b1_add":
        return _make("b1", "add")

    if model_name in {"b1_cm-ssm", "b1_cm_ssm", "b1cmssm", "model1_b1_mamba4"}:
        return _make("b1", "CM-SSM")

    if model_name in {"b1_m-ssm", "b1_m_ssm"}:
        return _make("b1", "M-SSM")

    if model_name == "b1_cat":
        return _make("b1", "cat")

    if model_name == "b1_max":
        return _make("b1", "max")

    raise ValueError(f"Unsupported model_name '{cfg.get('model_name')}'")
