# optimizer
optimizer = {"type": "Adam", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001}

optimizer_config = {"grad_clip": {"max_norm": 35, "norm_type": 2}}

# learning policy
lr_config = {
    "policy": "step",
    "warmup": "linear",
    "warmup_iters": 500,
    "warmup_ratio": 0.001,
    "step": [8, 11],
}

runner = {"type": "EpochBasedRunner", "max_epochs": 12}
