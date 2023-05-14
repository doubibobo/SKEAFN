import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter


class Summary:
    def __init__(self, summary_root):
        if os.path.exists(summary_root):
            shutil.rmtree(summary_root)
        os.makedirs(summary_root)
        
        self.summary_writer = SummaryWriter(summary_root)

    def summary_model(self, model, model_input):
        """
        保存模型整体结构
        :param input: 模型的理论输入
        """
        model.eval()
        with torch.no_grad():
            self.summary_writer.add_graph(
                model=model, input_to_model=model_input, verbose=False
            )

    def summary_writer_add_scalars(self, epoch, train_metrics, valid_metrics, tag="epoch"):
        """
        同时记录训练和验证的指标
        :param tag: 图组标签, 默认指标是一个epoch下的, 而非一次iteration
        """
        for key, value in train_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.summary_writer.add_scalars(
                        "{}/{}/{}".format(tag, key, sub_key),
                        {
                            "train": train_metrics[key][sub_key],
                            "valid": valid_metrics[key][sub_key],
                        },
                        epoch,
                    )
            else:
                self.summary_writer.add_scalars(
                    "{}/{}".format(tag, key),
                    {
                        "train": train_metrics[key], 
                        "valid": valid_metrics[key],
                    },
                    epoch,
                )

    def summary_writer_add_scalar(self, train_step, train_metrics, tag="iteration"):
        """
        仅仅记录训练的指标
        :param tag: 图组标签, 默认指标是一个iteration下的, 而非一次epoch
        """
        for key, value in train_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.summary_writer.add_scalar(
                        "{}/{}/{}".format(tag, key, sub_key),
                        train_metrics[key][sub_key],
                        train_step,
                    )
            else:
                self.summary_writer.add_scalar(
                    "{}/{}".format(tag, key), train_metrics[key], train_step
                )

    def learning_rate_summary(self, epoch, lr_params):
        lrs = {}
        for param_group in lr_params:
            group_id = len(lrs)
            lrs["{}".format(group_id)] = param_group["lr"]

        self.summary_writer.add_scalars(
            "lr", lrs, epoch,
        )
        pass

    def close(self):
        self.summary_writer.close()
