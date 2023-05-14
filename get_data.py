import torch
from tools import get_class_from_name

def get_dataloader(args):
    special_params = {}
    # TODO 只在包含KB时用，这里可以根据需要自动化修改
    if args.dataset_name in ["mosi", "mosei", "twitter"]:
        special_params = {
            "word_sentiment_dict_path": args.kb["word_in_sentiment"],
            "vocab_path": args.kb["vocab_path"],
        }

    data_loader_params = {}

    if args.dataset_name in ["twitter"]:
        data_loader_params = {
            "text_params": args.text,
            "visual_params": args.visual,
            "acoustic_params": args.acoustic,
            "kb_params": special_params,
            "label_params": args.label,
            "with_text": args.with_text, 
            "with_visual": args.with_visual, 
            "with_acoustic": args.with_acoustic, 
            "with_kb": args.with_kb,
            "classes": args.class_number,
            "arch": "train",
        }
    elif args.dataset_name in ["mosi", "mosei"]:
        data_loader_params = {
            "all_feature_path": args.all_feature_path,
            **special_params,
            "with_text": args.with_text, 
            "with_visual": args.with_visual, 
            "with_acoustic": args.with_acoustic, 
            "with_kb": args.with_kb,
            "pretrained_arch": args.text["pretrained_arch"],
            "pretrained_path": args.text["pretrained_path"],
            "is_align": args.is_align,
            "need_align": args.need_align,
            "classes": args.class_number,
            "arch": "train",
        }
    else:
        pass

    train_dataloader = torch.utils.data.DataLoader(
        get_class_from_name(args.dataset_class)(**data_loader_params),
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    data_loader_params["arch"] = "valid"
    valid_dataloader = torch.utils.data.DataLoader(
        get_class_from_name(args.dataset_class)(**data_loader_params),
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    data_loader_params["arch"] = "test"
    test_dataloader = torch.utils.data.DataLoader(
        get_class_from_name(args.dataset_class)(**data_loader_params),
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return train_dataloader, valid_dataloader, test_dataloader
