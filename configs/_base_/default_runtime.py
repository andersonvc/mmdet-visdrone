log_level = "ERROR"

parameter_names = {"learning_rate", "momentum"}

log_config = {
    "interval": 50,
    "hooks": [{"type": "CustomMlflowLoggerHook", "parameters": parameter_names}],
}
evaluation = {"metric": "mAP", "interval": 1}
checkpoint_config = {"interval": 1}
dist_params = {"backend": "nccl"}
load_from = None
resume_from = None

work_dir = None
seed = 0
gpu_ids = range(0, 1)
