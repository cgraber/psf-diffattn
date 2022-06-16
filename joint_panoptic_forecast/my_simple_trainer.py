import time

import numpy as np
import torch
from typing import List, Mapping, Optional
from detectron2.engine import SimpleTrainer
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage



'''
This class exists to change the loss computation/logging procedure.

Specifically, it allows for plotting of tensorboard metrics before 
being multiplied by loss coefficients. This will make it easier to compare
different runs with different coefficients
'''
class MySimpleTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, grad_accumulate_steps=None):
        super().__init__(model, data_loader, optimizer)
        self.grad_accumulate_steps = grad_accumulate_steps
        self.current_steps = 0


    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            first_val = next(iter(loss_dict.values()))
            if isinstance(first_val, list) or isinstance(first_val, tuple):
                # New logic
                losses = sum([coef*loss for loss, coef in loss_dict.values()])
            else: 
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        #self.optimizer.zero_grad()
        if self.grad_accumulate_steps is not None:
            losses = losses / self.grad_accumulate_steps
            self.current_steps += 1
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.grad_accumulate_steps is None or self.current_steps % self.grad_accumulate_steps == 0:
            self.current_steps = 0
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        MySimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        first_val = next(iter(loss_dict.values()))
        if isinstance(first_val, list) or isinstance(first_val, tuple):
            metrics_dict = {k: v[0].detach().cpu().item() for k, v in loss_dict.items()}
        else:
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)