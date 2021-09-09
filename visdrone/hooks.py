import os
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class CustomMlflowLoggerHook(LoggerHook):
    def __init__(
        self,
        exp_name=None,
        tags=None,
        log_model=True,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        by_epoch=True,
        parameters=set(),
    ):
        """Class to log metrics and (optionally) a trained model to MLflow.

        It requires `MLflow`_ to be installed.

        Args:
            exp_name (str, optional): Name of the experiment to be used.
                Default None.
                If not None, set the active experiment.
                If experiment does not exist, an experiment with provided name
                will be created.
            tags (dict of str: str, optional): Tags for the current run.
                Default None.
                If not None, set tags for the current run.
            log_model (bool, optional): Wheter to log an MLflow artifact.
                Default True.
                If True, log runner.model as an MLflow artifact
                for the current run.
            interval (int): Logging interval (every k iterations).
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`.
            reset_flag (bool): Whether to clear the output buffer after logging
            by_epoch (bool): Whether EpochBasedRunner is used.

        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """
        super(CustomMlflowLoggerHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch
        )
        self.import_mlflow()
        self.exp_name = exp_name
        self.tags = tags
        self.log_model = log_model
        self.parameters = parameters

    def import_mlflow(self):
        try:
            import mlflow
            import mlflow.pytorch as mlflow_pytorch
        except ImportError:
            raise ImportError('Please run "pip install mlflow" to install mlflow')
        self.mlflow = mlflow
        self.mlflow_pytorch = mlflow_pytorch

    @master_only
    def before_run(self, runner):
        super(CustomMlflowLoggerHook, self).before_run(runner)
        if self.exp_name is not None:
            self.mlflow.set_experiment(self.exp_name)
        if self.tags is not None:
            self.mlflow.set_tags(self.tags)
            print(self.get_loggable_tags(runner))

    @master_only
    def log(self, runner):
        """
        key names in tags updated to replace '/' chars with '_'.
        fixes a bug in the mlflow ui.
        """
        tags = self.get_loggable_tags(runner)
        tags = {
            k.replace("/", "_"): v for k, v in tags.items() if k not in self.parameters
        }
        if tags:
            self.mlflow.log_metrics(tags, step=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        if self.log_model:
            self.mlflow_pytorch.log_model(runner.model, "models")



@HOOKS.register_module()
class CustomTensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(CustomTensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(CustomTensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'CustomTensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = os.path.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        raise Exception(next(iter(runner.data_loader)))
        image_tensor = next(iter(runner.data_loader))._data
        self.writer.add_graph(runner.model,image_tensor)


    @master_only
    def after_run(self, runner):
        self.writer.close()