import os
import shutil
import time

import numpy as np
import optuna
import torch

from torch.cuda.amp import autocast, GradScaler

from metrics import (calculate_metrics, print_confusion_matrix)
from logger import Summary
from trick import (EMA, FGM)


class TrainProcess:

    def __init__(
        self,
        args,
        train_loader,
        valid_loader,
        test_loader,
        model,
        criteria,
        optimizer,
        scheduler,
        logger,
        checkpoint_log_dir,
        device="cpu",
    ):
        self.args = args

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.checkpoint_log_dir = checkpoint_log_dir

        self.model_root = os.path.join(checkpoint_log_dir, "models")
        if os.path.exists(self.model_root):
            shutil.rmtree(self.model_root)
        os.makedirs(self.model_root)

        self.criterion = criteria
        self.optimizer = optimizer
        self.schedule = scheduler

        self.summary = Summary(os.path.join(checkpoint_log_dir, "logs"))
        self.logger = logger

        self.device = device

        self.model = torch.nn.parallel.DataParallel(model.to(device))
        
        # 添加fgm trick
        self.fgm = FGM(self.model)

        self.epoch = 0
        self.best_epoch = 0
        self.best_accuracy = 0
        self.best_loss = 1e8

    def train_process(self, max_epoch, wait_epoch, iteration, trail):
        start_time = time.time()
        train_step = iteration
        
        while True:
            self.model.train()
            train_losses, train_predictions, train_labels = [], [], []

            for _, (train_data, train_label, _) in enumerate(self.train_loader):
                for key, value in train_data.items():
                    train_data[key] = value.to(self.device)
                train_label = train_label.to(self.device)
                
                self.optimizer.zero_grad()
                
                model_output = self.model(train_data)
                loss = self.criterion(model_output["logistics"], train_label)
                    
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                if self.schedule is not None and self.args.scheduler["name"] in ["LinearWarmup"]:
                    self.schedule.step()
                    self.summary.learning_rate_summary(train_step, self.optimizer.param_groups)

                train_losses.append(loss.detach().cpu().numpy())
                train_predictions.extend(model_output["predictions"].cpu().numpy().tolist())
                train_labels.extend(train_label.cpu().numpy().tolist())
                
                # 可视化训练过程中的损失, 每个batch更新一次
                self.summary.summary_writer_add_scalar(
                    train_step, 
                    {
                        "total_loss": loss.detach().cpu().numpy(),                     
                    },
                    tag="loss"
                )

                train_step += 1
            
            if self.schedule is not None and self.args.scheduler["name"] not in ["LinearWarmup"]:
                self.schedule.step()
                self.summary.learning_rate_summary(self.epoch, self.optimizer.param_groups)

            train_loss = np.mean(train_losses)
            train_metrics = calculate_metrics(train_predictions, train_labels, self.args.class_number, self.args.dataset_name)
      
            valid_loss, valid_metrics = self.valid_process()

            self.logger.info("Epoch {} - Train-Metrics: {}; loss - {}.".format(
                self.epoch, 
                str(train_metrics), 
                str({"loss": train_loss})
            ))

            self.logger.info("Epoch {} - Valid-Metrics: {}; loss - {}. \n".format(
                self.epoch, 
                str(valid_metrics), 
                str({"loss": valid_loss})
            ))
            
            self.summary.summary_writer_add_scalars(
                self.epoch, 
                {"loss": train_loss, **train_metrics},
                {"loss": valid_loss, **valid_metrics}
            )            

            self.save_model(valid_metrics)

            # 早停机制
            if self.epoch >= max_epoch or (self.epoch - self.best_epoch) >= wait_epoch:
                self.summary.close()
                self.logger.info("Summary writer close!")
                self.logger.info("The epoch number is {}, and the best_epoch is {}, and the best {} is {}".format(
                    self.epoch, self.best_epoch, 
                    self.args.core_metrics, 
                    self.best_accuracy if self.args.optimize_direction_high else self.best_loss))
                time_consuming = time.time() - start_time
                self.logger.info("Time consume for traing: {} mins; {} hours\n". format(time_consuming / 60, time_consuming / 3600))
                break

            trail.report(valid_metrics[self.args.core_metrics], self.epoch)

            if trail.should_prune():
                self.summary.close()
                self.logger.info("Summary writer close!")
                # 自动删除剪枝时时的模型和日志
                shutil.rmtree(self.checkpoint_log_dir)
                raise optuna.exceptions.TrialPruned()

            self.epoch += 1
            # break

    def valid_process(self):
        self.model.eval()
        with torch.no_grad():
            valid_losses, valid_predictions, valid_labels = [], [], []
            for _, (valid_data, valid_label, _) in enumerate(self.valid_loader):
                for key, value in valid_data.items():
                    valid_data[key] = value.to(self.device)
                valid_label = valid_label.to(self.device)
                
                model_output = self.model(valid_data)
                loss = self.criterion(model_output["logistics"], valid_label)

                valid_losses.append(loss.cpu().numpy().tolist())
                valid_predictions.extend(model_output["predictions"].cpu().numpy().tolist())
                valid_labels.extend(valid_label.cpu().numpy().tolist())

            valid_metrics = calculate_metrics(valid_predictions, valid_labels, self.args.class_number, self.args.dataset_name)
            valid_loss = np.mean(valid_losses)
            
            # print("Confusion matrix for valid set:\n")
            # print_confusion_matrix(valid_labels, valid_predictions, self.args.class_number)

        return valid_loss, valid_metrics

    def test_process(self, test_pretrained_dir=None):
        models = []
        if test_pretrained_dir is None:
            checkpoint_files = None
            self.model = torch.load(os.path.join(self.model_root, "best_model.pth"))["state_dict"]
            self.model = torch.nn.parallel.DataParallel(self.model.to(self.device))
            models.append(self.model)
        else:
            checkpoint_files = os.listdir(test_pretrained_dir)
            for checkpoint_file in checkpoint_files:
                model_path = os.path.join(test_pretrained_dir, checkpoint_file)
                self.logger.info("Model Checkpoint: {}".format(model_path))
                self.model = torch.load(model_path)["state_dict"]
                self.model = self.model.to(self.device)
                models.append(self.model)
        for i in range(len(models)):
            model = models[i]
            model.eval()
            
            data_loaders = {"train": self.train_loader, "valid": self.valid_loader, "test": self.test_loader}
            for dataloader_key, data_loader in data_loaders.items():
                with torch.no_grad():
                    ids, raw_texts = [], []
                    test_losses, test_predictions, test_labels = [], [], []
                    l_features, l1_features, l2_features, l3_features = [], [], [], []
                    for _, (test_data, test_label, test_message) in enumerate(data_loader):
                        for key, value in test_data.items():
                            test_data[key] = value.to(self.device)
                        test_label = test_label.to(self.device)

                        model_output = self.model(test_data)
                        loss = self.criterion(model_output["logistics"], test_label)

                        test_losses.append(loss.cpu().numpy().tolist())
                        test_predictions.extend(model_output["predictions"].cpu().numpy().tolist())
                        test_labels.extend(test_label.cpu().numpy().tolist())
                        
                        # # for case study
                        # l_features.extend(model_output['fusion_features'])
                        # l1_features.extend(l1_output['fusion_features'])
                        # l2_features.extend(l2_output['fusion_features'])
                        # l3_features.extend(l3_output['fusion_features'])
                        
                        # ids.extend(test_message["id"])
                        # raw_texts.extend(test_message["raw_text"])

                    test_metrics = calculate_metrics(test_predictions, test_labels, self.args.class_number, self.args.dataset_name)

                    # print("* Confusion matrix for test set:\n")
                    confusion = print_confusion_matrix(test_labels, test_predictions, self.args.class_number)
                    
                    self.logger.info("Loss for {}: {}".format(dataloader_key, np.mean(test_losses)))
                    self.logger.info("Metrics for {}: {}".format(dataloader_key, str(test_metrics)))

                    # l_features = np.array(l_features)
                    # l1_features = np.array(l1_features)
                    # l2_features = np.array(l2_features)
                    # l3_features = np.array(l3_features)
                    # ids = np.expand_dims(np.array(ids), axis=1)
                    # raw_texts = np.expand_dims(np.array(raw_texts), axis=1)
                    
                    # test_predictions = np.expand_dims(np.array(test_predictions), axis=1)
                    # test_labels = np.expand_dims(np.array(test_labels), axis=1)
                    
                    # visual_data = np.concatenate((ids, raw_texts, test_predictions, test_labels, l_features, l1_features, l2_features, l3_features), axis=1)
                    # if checkpoint_files is None:
                    #     np.save("visualization/{}/visual_data/{}.npy".format(self.args.dataset_name.lower(), dataloader_key), visual_data, allow_pickle=True)
                    # else:
                    #     np.save("visualization/{}/visual_data/{}_{}.npy".format(self.args.dataset_name.lower(), dataloader_key, checkpoint_files[i].split(".")[0]), visual_data, allow_pickle=True)

        return {
            "loss": np.mean(test_losses),
            "best_epoch": self.best_epoch,
            **test_metrics,
            "confusion": str(confusion).replace("\n", ""), 
        }

    def save_model(self, metrics):
        if self.args.optimize_direction_high:
            # 朝值高的方向优化
            if metrics[self.args.core_metrics] >= (self.best_accuracy + 1e-6):
                self.best_epoch = self.epoch
                self.best_accuracy = metrics[self.args.core_metrics]
                model_save_path = os.path.join(self.model_root, "best_model.pth")
                torch.save(
                    {
                        "epoch": self.epoch,
                        "metrics": metrics,
                        "state_dict": self.model.module,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.schedule.state_dict() if self.schedule is not None else None,
                    },
                    model_save_path,
                )
        else:
            if metrics[self.args.core_metrics] <= (self.best_loss - 1e-6):
                # 朝值小的方向优化
                self.best_epoch = self.epoch
                self.best_loss = metrics[self.args.core_metrics]
                model_save_path = os.path.join(self.model_root, "best_model.pth")
                torch.save(
                    {
                        "epoch": self.epoch,
                        "metrics": metrics,
                        "state_dict": self.model.module,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.schedule.state_dict() if self.schedule is not None else None,
                    },
                    model_save_path,
                )
