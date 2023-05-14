import os
from tools import Storage


class Config:

    def __init__(self, args):
        self.work_dir = args.work_dir
        self.dataset_prefix = args.dataset_prefix
        self.pretrained_models_dir = args.pretrained_path
        self.dataset_name = args.dataset_name.upper()
        self.category_number = args.category_number
        self.pretrained_arch = "BertTokenizer" if args.pretrained_arch == "bert" else "RobertaTokenizer"
        self.pretrained_path = os.path.join(args.pretrained_path, "bert-base-uncased" if args.pretrained_arch == "bert" else "roberta-base")
        self.with_text = args.with_text
        self.with_acoustic = args.with_acoustic
        self.with_visual = args.with_visual
        self.with_kb = args.with_kb

        try:
            self.global_params = vars(args)
        except TypeError:
            self.global_params = args

    def __dataset_config_params(self):
        return {
            "twitter": {
                "text": {
                    "pretrained_arch": self.pretrained_arch,
                    "pretrained_path": self.pretrained_path
                },
                "acoustic": {
                    "feature_path": os.path.join(self.work_dir, "data/processed/features/ocr"),
                },
                "visual": {
                    "image_size": 224,
                    "data_mean": [0.485, 0.456, 0.406],
                    "data_std": [0.229, 0.224, 0.225],
                    "feature_arch": "vit-base-patch16-224",
                    "feature_path": os.path.join(self.work_dir, "data/processed/features/images"),
                },
                "kb": {
                    # 期望的词汇表为字典格式，key为word, value为ID
                    "vocab_path": os.path.join(self.work_dir, "data/processed/KB/TWITTER/word_to_id.npy"),
                    # 数据集每个样本中的数据在情感字典中的分布
                    "word_in_sentiment": os.path.join(self.work_dir, "data/processed/KB/TWITTER/word_in_sentiment_dict.npy"),
                },
                "label": {
                    "dir": os.path.join(self.work_dir, "data/origin"),  
                },
                "class_number": self.category_number,
                "criterion": {
                    "type": "loss.ASLSingleLabel",
                    "params": {
                        "reduction": "mean",
                    },
                },
                "dataset_class": "dataloader.TwitterDataset",
                "core_metrics": "f1_weighted",
                "optimize_direction_high": True,
            },
            "mosi": {
                "all_feature_path": os.path.join(self.dataset_prefix, "MSA-Datasets/MOSI/Processed"),
                "text": {
                    "pretrained_arch": self.pretrained_arch,
                    "pretrained_path": self.pretrained_path
                },
                "kb": {
                    # 期望的词汇表为字典格式，key为word, value为ID
                    "vocab_path": os.path.join(self.work_dir, "data/processed/KB/MOSI/word_to_id.npy"),
                    # 数据集每个样本中的数据在情感字典中的分布
                    "word_in_sentiment": os.path.join(self.work_dir, "data/processed/KB/MOSI/word_in_sentiment_dict.npy"),
                },
                "is_align": False,
                "need_align": False,
                "class_number": self.category_number,
                "criterion": {
                    "type": "torch.nn.L1Loss",
                    "params": {
                        "reduction": "mean"
                    },
                },
                "dataset_class": "dataloader.MOSIDataset",
                "core_metrics": "mae",
                "optimize_direction_high": False,
            },
            "mosei": {
                "all_feature_path": os.path.join(self.dataset_prefix, "MSA-Datasets/MOSEI/Processed"),
                "text": {
                    "pretrained_arch": self.pretrained_arch,
                    "pretrained_path": self.pretrained_path
                },
                "kb": {
                    # 期望的词汇表为字典格式，key为word, value为ID
                    "vocab_path": os.path.join(self.work_dir, "data/processed/KB/MOSEI/word_to_id.npy"),
                    # 数据集每个样本中的数据在情感字典中的分布
                    "word_in_sentiment": os.path.join(self.work_dir, "data/processed/KB/MOSEI/word_in_sentiment_dict.npy"),
                },
                "is_align": False,
                "need_align": False,
                "class_number": self.category_number,
                "criterion": {
                    "type": "torch.nn.L1Loss",
                    "params": {
                        "reduction": "mean"
                    },
                },
                "dataset_class": "dataloader.MOSEIDataset",
                "core_metrics": "mae",
                "optimize_direction_high": False,
            },
        }

    def __train_config_params(self):
        text_lr = 5e-5
        text_wd = 1e-3
        visual_lr = 0.0009
        visual_wd = 0.003
        acoustic_lr = 0.002
        acoustic_wd = 0.007
        kb_lr = 0.008
        kb_wd = 0.01
        other_lr = 0.001
        other_wd = 0.003

        batch_size = 64
        
        return {
            "train_config": {
                "train_type": "train.TrainProcess",
                "lr": {
                    "text": text_lr,
                    "acoustic": acoustic_lr,
                    "visual": visual_lr,
                    "kb": kb_lr,
                    "other": other_lr,
                },
                "wd": {
                    "text": text_wd,
                    "acoustic": acoustic_wd,
                    "visual": visual_wd,
                    "kb": kb_wd,
                    "other": other_wd,
                },
                "bs": batch_size,
                "early_stop": 10,  # default is 5
                "max_epoch": 32,
                "optimizer": "torch.optim.Adam",
                "scheduler": {
                    "name":
                    "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                    "params": {
                        "T_0": 2,
                        "T_mult": 2
                    },
                },
            },
        }

    def __model_config_params(self):
        transformer_layers = 2  # roberta: 2
        feature_droprate = 0.1
        return {
            "model_config": {
                "net_type": "model.SKEAFN",
                "input_channels": {
                    "text": self.with_text,
                    "acoustic": self.with_acoustic,
                    "visual": self.with_visual,
                    "kb": self.with_kb,
                },
                "me_text": {
                    "pre": {
                        "type": "Roberta" if "Roberta" in self.pretrained_arch else "Bert",
                        "params": {
                            "model_name": os.path.join(self.pretrained_models_dir, "roberta-base" if "Roberta" in self.pretrained_arch else "bert-base-uncased"),
                            "use_finetune": True,
                        }
                    },
                    "post": {
                        "type": "LSTM",
                        "params": {
                            "input_size": 768,  
                            "hidden_size": 512,
                            "output_size": 768,
                            "number_layers": 1,
                            "drop_rate": feature_droprate * 1,
                            "bidirectional": False,
                        }
                    },
                },
                "me_acoustic": {
                    "type": "LSTM",
                    "params": {
                        "input_size": 768,  # 768 for Twitter2019, 74 for MOSEI, 5 for MOSI
                        "hidden_size": 768,  # default is 768
                        "output_size": 768,  # default is 768
                        "number_layers": 1,  # 4 for image test, 2 for others
                        "drop_rate": feature_droprate * 1,
                        "bidirectional": False
                    }
                },
                "me_visual": {
                    "type": "LSTM",
                    "params": {
                        "input_size": 768,  # 2048 for Twitter2019, 20 for MOSI, and 35 for MOSEI
                        "hidden_size": 768,  # default is 768
                        "output_size": 768,  # default is 768
                        "number_layers": 1,  # 4 for image test, 2 for others
                        "drop_rate": feature_droprate * 1,
                        "bidirectional": False
                    }
                },
                "eke": {
                    "type": "TextGCN",
                    "params": {
                        "node_hidden_size": 768,
                        "vocab_path": os.path.join(self.work_dir, "data/processed/KB", self.dataset_name, "word_to_id.npy"), "edge_weights_path": os.path.join(self.work_dir, "data/processed/KB", self.dataset_name, "edge_weights.npy"),
                        "edge_matrix_path": os.path.join(self.work_dir, "data/processed/KB", self.dataset_name, "edge_mappings.npy"),
                        "vocab_embedding_path": os.path.join(self.work_dir, "data/processed/KB", self.dataset_name, "bert_embedding.npy"),
                        "edge_trainable": True,
                        "graph_embedding_drop_rate": feature_droprate,
                    }
                },
                "fusion": {
                    "pre": {
                        "type": "CrossTransformer",
                        "params": {
                            "hidden_size": 768,  # default is 768
                            "num_attention_heads": 12,  # default is 12
                            "attention_probs_dropout_prob":
                            feature_droprate * 1,
                            "hidden_dropout_prob": feature_droprate * 1,
                            "intermediate_size": 3072,  # default is 3072
                            "num_hidden_layers": transformer_layers,
                            "position_embedding_type": "absolute",
                            "max_position_embeddings": 512,
                            "add_cross_attention": True,
                        },
                    },
                    "post": {
                        "type": "ConcatDenseSE",
                        "params": {
                            "input_dim": 768,  # default is 768
                        }
                    }
                },
                "classifier": {
                    "type": "LogisticModel",
                    "params": {
                        "classes_number": self.category_number,
                        "input_dim": 768,  # default is 768
                        "dropout_rate": feature_droprate * 1,
                    }
                }
            }
        }
    
    def get_config(self):
        return Storage(dict(
                self.global_params,
                **self.__model_config_params(),
                **self.__train_config_params()["train_config"],
                **self.__dataset_config_params()[self.global_params["dataset_name"]]))
