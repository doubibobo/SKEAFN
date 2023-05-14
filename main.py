import json
import os
import sys
import time
import traceback

import argparse
import optuna
import torch
from optuna.trial import TrialState

import logging
logging.basicConfig(
    level=logging.DEBUG, 
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/zhuchuanbo/Documents/competition/SKEAFN")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
logger.info("Nvidia number: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # This is crucial for reproducibility

from config import Config
from get_data import get_dataloader
from tools import (check_and_create_dir, set_seed, write_json_result_to_csv, write_single_json_to_whole, get_class_from_name)

results_number = 0      # 实验结果数量
result_list_path = ''   # 所有实验结果的路径


def objective(trail):
    # 设置系列参数
    global_args = Config(parse_args()).get_config(trail)
    global_args.checkpoint_log_path = os.path.join(global_args.work_dir, global_args.checkpoint_log_path)
    global_args.res_save_path = os.path.join(global_args.work_dir, global_args.res_save_path)

    time_dir = time.strftime("%Y-%m-%d@%H-%M-%S")
    
    global_args.checkpoint_log_path = check_and_create_dir(global_args, global_args.checkpoint_log_path, time_dir)
    global_args.res_save_path = check_and_create_dir(global_args, global_args.res_save_path)
    global result_list_path
    result_list_path = global_args.res_save_path

    set_seed(global_args.seed)
    
    # print(json.dumps(global_args, indent=4))
    # 保存参数文件
    with open(os.path.join(global_args.checkpoint_log_path, "args_file.json"), "w+") as file:
        json.dump(global_args, fp=file, indent=4)
    
    test_metrics = do_train(global_args, trail)
    
    global results_number     
    results_number += 1
    
    # 保存一次的实验结果
    results_to_save = {
        "id": results_number,
        "metrics": test_metrics,
        "params": global_args,
    }
    trail_result_path = os.path.join(global_args.res_save_path, "performance_{}.json".format(results_number))
    with open(trail_result_path, "w+") as file:
        json.dump(results_to_save, fp=file, indent=4)
    logger.info("Experiment results are save to {}.\n".format(trail_result_path))    

    return test_metrics[global_args.core_metrics]


def do_train(args, trail):
    using_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if using_cuda else "cpu")
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args)

    model = get_class_from_name(args.model_config["net_type"])(args.model_config)
    criterion = get_class_from_name(args.criterion["type"])(**args.criterion["params"]).to(device)

    text_bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    text_bert_params = list(model.me_text.named_parameters())
    
    acoustic_params = list(model.me_acoustic.named_parameters())
    visual_params = list(model.me_visual.named_parameters())
    kb_params = list(model.eke.named_parameters())
    
    text_bert_params_decay = [p for n, p in text_bert_params if not any(nd in n for nd in text_bert_no_decay)]
    text_bert_params_no_decay = [p for n, p in text_bert_params if any(nd in n for nd in text_bert_no_decay)]
    acoustic_params = [p for n, p in acoustic_params]
    visual_params = [p for n, p in visual_params]
    kb_params = [p for n, p in kb_params]
    other_params = [p for n, p in list(model.named_parameters()) if 'me_text' not in n and 'me_acoustic' not in n and 'me_visual' not in n and 'eke' not in n]

    optimizer = get_class_from_name(args.optimizer)(
        [
            {
                "params": text_bert_params_decay, 
                "weight_decay": args.wd["text"],
                "lr": args.lr["text"],
            },
            {
                "params": text_bert_params_no_decay,
                "weight_decay": 0.0,
                "lr": args.lr["text"],
            },
            {
                "params": visual_params,
                "weight_decay": args.wd["visual"],
                "lr": args.lr["visual"],
            },
            {
                "params": acoustic_params,
                'weight_decay': args.wd["acoustic"],
                "lr": args.lr["acoustic"],
            },
            {
                "params": kb_params,
                'weight_decay': args.wd["kb"],
                "lr": args.lr["kb"],
            },
            {
                "params": other_params,
                'weight_decay': args.wd["other"],
                "lr": args.lr["other"],
            },
        ],
    )

    scheduler = get_class_from_name(args.scheduler["name"])(optimizer=optimizer, **args.scheduler["params"])

    train_process = get_class_from_name(args.train_type)(
        args,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        logger,
        args.checkpoint_log_path,
        device=device,
    )

    if args.test_pretrained_path is None:
        train_process.train_process(
            max_epoch=args.max_epoch,
            wait_epoch=args.early_stop,
            iteration=0,
            trail=trail,
        )
    test_metrics = train_process.test_process(
        args.test_pretrained_path)
    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="mosi",help="support twitter mosi mosei.")
    parser.add_argument("--net_type", type=str, default="SKEAFN",help="support SKEAFN")
    parser.add_argument("--category_number", type=int, default=1, help="the categories of the input data.")
    parser.add_argument("--optimize_times", type=int, default=100, help="optimize times for optuna.")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers of loading data.")
    parser.add_argument("--checkpoint_log_path", type=str, default="checkpoints", help="path to save model and tensorboard log.")
    parser.add_argument("--res_save_path", type=str, default="results", help="path to save results.")
    parser.add_argument("--work_dir", type=str, default="/home/zhuchuanbo/Documents/competition/SKEAFN", help="path of working directory.")
    parser.add_argument("--dataset_prefix", type=str, default="/home/zhuchuanbo/Documents/datasets/multimodal_sentiment_analysis", help="path to dataset prefix.")
    parser.add_argument("--pretrained_path", type=str, default="pretrained_models")
    parser.add_argument("--test_pretrained_path", type=str, default=None, help="default mode for train or test.")
    parser.add_argument("--is_seed_valid", action='store_true', help="determine whether is seed validation.")
    parser.add_argument("--optuna_direction_max", action='store_true', help="optuna direction is max.")
    parser.add_argument("--pretrained_arch", type=str, default="bert")
    parser.add_argument("--with_text", action='store_false', help="")
    parser.add_argument("--with_acoustic", action='store_false', help="")
    parser.add_argument("--with_visual", action='store_false', help="")
    parser.add_argument("--with_kb", action='store_false', help="")
    return parser.parse_args()


if __name__ == "__main__":
    parser_args = parse_args()
    try:
        optuna_direction_max = parser_args.optuna_direction_max
        logger.info("Optimization direction: {}".format("maximize" if optuna_direction_max else "minimize"))

        if parser_args.is_seed_valid:
            parser_args.optimize_times = 1
        
        study = optuna.create_study(
            direction="maximize" if optuna_direction_max else "minimize", 
            study_name="{}-{}-{}".format(parser_args.dataset_name, parser_args.net_type, time.strftime("%Y-%m-%d=%H-%M-%S")),
            storage='sqlite:///db.sqlite3',
        )
        study.optimize(objective, n_trials=parser_args.optimize_times)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        
        logger.info("Study statistics: ")
        logger.info("Number of finished trials: {}".format(len(study.trials)))
        logger.info("Number of pruned trials: {}".format(len(pruned_trials)))
        logger.info("Number of complete trials: {}".format(len(complete_trials)))

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info("Value: {}".format(trial.value))
        logger.info("Params: ")
        for key, value in trial.params.items():
            logger.info("{}: {}".format(key, value))
    except Exception as e:
        logger.info("Study trails are not finished, some errors occurred!")
        traceback.print_exc()
    finally:
        write_json_result_to_csv(
            json_dir=result_list_path,
            csv_output=os.path.join(result_list_path, "result_list.csv")
        )
        write_single_json_to_whole(
            json_dir=result_list_path,
            json_output=os.path.join(result_list_path, "result_list.json")
        )
