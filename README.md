# SKEAFN [![](https://badgen.net/badge/license/GNU/green)](#LICENSE)
## 1. Introduction
The source code for the paper titled "Sentiment Knowledge Enhanced Attention Fusion Network (SKEAFN)".

### 1.1 Datasets
| Dataset           | Train   | Valid   | Test   | Total     |  Modality   | Type | Sentiment |
|:-------------:    |:-------:|:-------:|:------:|:-------:|:-------:|:-------:|:-------:|
| CMU-MOSI          | 1284    | 229     | 686    | 2199    | Text, Acoustic, Visual | Regression | HN, NG, WN, NU, WP, PS, HP |
| CMU-MOSEI         | 16326   | 1871    | 4659   | 22856   | Text, Acoustic, Visual | Regression | HN, NG, WN, NU, WP, PS, HP |
| Twitter2019       | 19816   | 2410    | 2409   | 24635   | Text, Visual | Classification | A, no SA |

### 1.2 Models
| Models    | Metrics   |
|:---------:|:---------:|
| SKEAFN    | accuracy  |

## 2. Code Structure
```
├── checkpoints                     # dir for saving best model and tensorboard logs.
├── logs                            # dir for recording outputs of the training process.
├── pretrained_models               # dir for saving pretrained models from HuggingFace.
├── results                         # dir for recording performance of the SKEAFN model.
├── ca.py                           # implement for text-guided interaction.
├── classifier.py                   # implement for classifier.
├── dataloader.py                   # implement for constructing PyTorch dataloader.
├── fafw.py                         # implement for feature-wised attention fusion.
├── get_data.py                     # implement for getting train/valid/test dataloader.
├── logger.py                       # implement for obtaining tensorboard log.
├── loss.py                         # implement for ASLSingleLabel loss.
├── main.py                         # implement for main.
├── metrics.py                      # implement for evaluation metrics.
├── model.py                        # implement for our SKEAFN model.
├── tools.py                        # implement for train tools.
├── train.py                        # implement for the train process.
├── trick.py                        # implement for tricks in training process.
├── LICENSE                     
└── README.md
```

## 3. Run
### 3.1 Environment
![](https://badgen.net/badge/python/3.9.13/blue)
![](https://badgen.net/badge/pypi/v21.2.4/orange)
![](https://badgen.net/badge/Pytorch/1.9.0/red)
```
pip install -r requirements.txt
```
### 3.2 Args
+ seed: random seed, default is `1`.
+ dataset_name: dataset name, support twitter mosi mosei, default is `mosi`.
+ net_type: model name, default is `SKEAFN`.
+ category_number: dim of the model outputs, default is `2`.
+ optimize_times: optimize times for optuna, default is `100`.
+ num_workers: num workers of loading data, default is `2`.
+ checkpoint_log_path: path to save model and tensorboard log, default is `checkpoints`.
+ res_save_path: path to save results, default is `results`.
+ work_dir: path of working directory, default is `.`.
+ dataset_prefix: path to dataset prefix, default is `dataset`.
+ pretrained_path: path to save pretrained models, default is `pretrained_models`.
+ test_pretrained_path: path of the best model and determine the mode (train, test), default is `None`.

### 3.3 Training
```
nohup python main.py --dataset_name=twitter --category_number=2 --pretrained_arch=roberta  --optimize_times=200 --num_workers=2 > logs/twitter_roberta_train.log 2>&1 &
```
### 3.4 Testing
```
nohup python main.py --dataset_name=twitter --test_pretrained_path=best_model.pth > logs/twitter_roberta_test.log 2>&1 &
```

## 5 Cite
```todo```
