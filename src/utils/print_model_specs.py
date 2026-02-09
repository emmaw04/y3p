from __future__ import annotations
import json
from pathlib import Path
from sklearn.base import clone

from src.models import (
    ModelConfig,
    get_binary_base_learners,
    get_multiclass_base_learners,
    make_meta_binary_lr,
    make_meta_multinomial_lr,
)

def serialisable_params(est):
    # sklearn params are usually JSONable; convert anything awkward to str
    out = {}
    for k, v in est.get_params(deep=True).items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out

def main():
    cfg = ModelConfig(random_state=42)

    specs = {"binary_base": {}, "multiclass_base": {}, "meta": {}}

    bin_base = get_binary_base_learners(cfg)
    for name, est in bin_base.items():
        e = clone(est)
        specs["binary_base"][name] = {
            "class": e.__class__.__name__,
            "params": serialisable_params(e),
        }

    mc_base = get_multiclass_base_learners(cfg, n_classes=5)
    for name, est in mc_base.items():
        e = clone(est)
        specs["multiclass_base"][name] = {
            "class": e.__class__.__name__,
            "params": serialisable_params(e),
        }

    meta1 = make_meta_binary_lr(cfg)
    meta2 = make_meta_multinomial_lr(cfg)
    specs["meta"]["binary_lr"] = {"class": meta1.__class__.__name__, "params": serialisable_params(meta1)}
    specs["meta"]["multinomial_lr"] = {"class": meta2.__class__.__name__, "params": serialisable_params(meta2)}

    out = Path("runs/model_specs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specs, indent=2))
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()