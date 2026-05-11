from __future__ import annotations

from typing import Iterable, Optional

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    RenderableColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class Callback:
    def setup(self, trainer) -> None:
        pass

    def on_train_start(self, trainer) -> None:
        pass

    def on_train_iteration_end(self, trainer, iteration: int, loss_scalars, render_pkg) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_eval_start(self, trainer, iteration: int, total: int) -> None:
        pass

    def on_eval_batch_end(self, trainer, iteration: int, batch_idx: int, metrics) -> None:
        pass

    def on_eval_end(self, trainer, iteration: int, metrics) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: Optional[Iterable[Callback]] = None):
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def setup(self, trainer) -> None:
        for callback in self.callbacks:
            callback.setup(trainer)

    def on_train_start(self, trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_train_iteration_end(self, trainer, iteration: int, loss_scalars, render_pkg) -> None:
        for callback in self.callbacks:
            callback.on_train_iteration_end(trainer, iteration, loss_scalars, render_pkg)

    def on_train_end(self, trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_eval_start(self, trainer, iteration: int, total: int) -> None:
        for callback in self.callbacks:
            callback.on_eval_start(trainer, iteration, total)

    def on_eval_batch_end(self, trainer, iteration: int, batch_idx: int, metrics) -> None:
        for callback in self.callbacks:
            callback.on_eval_batch_end(trainer, iteration, batch_idx, metrics)

    def on_eval_end(self, trainer, iteration: int, metrics) -> None:
        for callback in self.callbacks:
            callback.on_eval_end(trainer, iteration, metrics)


class ProgressCallback(Callback):
    def setup(self, trainer) -> None:
        self._stopped = False
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            SpinnerColumn(),
            RenderableColumn(),
        )
        self.progress.start()
        self.train_task = self.progress.add_task("[red]Training...", total=trainer.iterations)
        self.eval_task = self.progress.add_task("[green]Evaluating...", total=len(trainer.indices_for_eval))
        trainer.pbar = self.progress
        trainer.train_task = self.train_task
        trainer.eval_task = self.eval_task

    def on_train_iteration_end(self, trainer, iteration: int, loss_scalars, render_pkg) -> None:
        self.progress.update(
            self.train_task,
            advance=1,
            description=f"[red]Training...  | Loss: {loss_scalars['total']:.4f}",
        )

    def on_train_end(self, trainer) -> None:
        if not self._stopped:
            self.progress.stop()
            self._stopped = True

    def on_eval_start(self, trainer, iteration: int, total: int) -> None:
        self.progress.reset(self.eval_task, total=total)

    def on_eval_batch_end(self, trainer, iteration: int, batch_idx: int, metrics) -> None:
        self.progress.update(
            self.eval_task,
            advance=1,
            description=(
                f"[green]Evaluating... {batch_idx} | PSNR: {metrics['psnr']:.4f} "
                f"| SSIM: {metrics['ssim']:.4f} | L1: {metrics['l1']:.4f}"
            ),
        )


class EvalCallback(Callback):
    def on_train_iteration_end(self, trainer, iteration: int, loss_scalars, render_pkg) -> None:
        if trainer.should_eval(iteration):
            trainer.last_metrics = trainer.eval(iteration)


class CheckpointCallback(Callback):
    def on_train_iteration_end(self, trainer, iteration: int, loss_scalars, render_pkg) -> None:
        if trainer.should_save(iteration):
            trainer.checkpoints.save(iteration, trainer.gaussians, trainer.cams, metrics=trainer.last_metrics)


class WandbCallback(Callback):
    def __init__(self, project: str, name: Optional[str] = None, save_dir: Optional[str] = None):
        self.project = project
        self.name = name
        self.save_dir = save_dir
        self.wandb = None

    def setup(self, trainer) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is required when wandb_project is set. Install it with `pip install wandb`.") from exc
        self.wandb = wandb
        self.wandb.init(config=trainer.run_config, project=self.project, name=self.name, dir=self.save_dir)

    def on_eval_end(self, trainer, iteration: int, metrics) -> None:
        self.wandb.log({"PSNR": metrics["psnr"], "SSIM": metrics["ssim"], "L1": metrics["l1"]}, step=iteration)


class ConsoleEvalSummaryCallback(Callback):
    def on_eval_end(self, trainer, iteration: int, metrics) -> None:
        print(
            f"Evaluation results of {iteration}: "
            f"PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, L1: {metrics['l1']:.4f}"
        )


def default_callbacks(
    *,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    enable_progress: bool = True,
):
    callbacks = []
    if enable_progress:
        callbacks.append(ProgressCallback())
    callbacks.extend([EvalCallback(), CheckpointCallback(), ConsoleEvalSummaryCallback()])
    if wandb_project is not None:
        callbacks.append(WandbCallback(project=wandb_project, name=wandb_name, save_dir=save_dir))
    return callbacks
