import os
import pickle
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from tools import get_class_from_name

origin_data_dir = {
    "train": "train.txt",
    "valid": "valid2.txt",
    "test": "test2.txt"
}


class TwitterDataset(Dataset, ABC):

    def __init__(
        self, text_params, visual_params, acoustic_params, kb_params, label_params,
        with_text=True, with_visual=True, with_acoustic=True, with_kb=True,
        classes=2,
        arch="train",
    ):
        super(TwitterDataset, self).__init__()
        self.pretrained_arch = text_params["pretrained_arch"]
        self.pretrained_path = text_params["pretrained_path"]
        self.visual_feature_arch = visual_params["feature_arch"]
        self.transform = transforms.Compose([
            transforms.Resize(visual_params["image_size"]),
            transforms.CenterCrop(visual_params["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=visual_params["data_mean"], std=visual_params["data_std"]),
        ]),
        self.vocab_path=kb_params["vocab_path"]
        self.word_sentiment_dict_path = kb_params["word_sentiment_dict_path"]
        self.number_classes = classes
        self.arch = arch
        
        self.visual_dir = visual_params["feature_path"]
        self.acoustic_dir = acoustic_params["feature_path"]
        self.label_dir = label_params["dir"]
        
        self.tokenizer = get_class_from_name("transformers." + self.pretrained_arch).from_pretrained(self.pretrained_path, do_lower_case=True)
        self.word_sentiment_dict = np.load(self.word_sentiment_dict_path, allow_pickle=True).item()
        self.vocab = np.load(self.vocab_path, allow_pickle=True).item()

        self.data_dict = {}
        self.length = 0

        self.__create_sample_label_dict()
        print("Dataloader {} length: {}.".format(self.arch, self.length))

        self.with_text = with_text
        self.with_visual = with_visual
        self.with_acoustic = with_acoustic
        self.with_kb = with_kb

    def __create_sample_label_dict(self):
        visual_features = np.load(os.path.join(self.visual_dir, "sequence_{}_features.npy".format(self.visual_feature_arch)), allow_pickle=True).item()
        acoustic_features = np.load(os.path.join(self.acoustic_dir, "{}_sequence_features.npy".format(self.arch)), allow_pickle=True).item()
        text_label_file = os.path.join(self.label_dir, origin_data_dir[self.arch])

        with open(text_label_file) as file:
            lines = file.readlines()
            for line in tqdm(lines):
                # line = re.sub(r"(\s)emoji\w+", "", line)
                values = line.strip("\n").replace("[", "").replace("]", "").split(", ")
                id_value = values[0].replace('"', "").replace("'", "")
                label_value = int(values[-1])
                text_value = ""
                # TODO 验证集和测试集有两个标签，之前没有考虑过
                for value in values[1:-1 if self.arch == "train" else -2]:
                    text_value += value + ", "

                # TODO 直接将原始数据输入bert进行处理, 不使用jieba分词进行后处理
                text_value = text_value.strip("'").strip('"')
                if text_value.endswith(", "):
                    text_value = text_value.strip(", ")

                text_value_ids_dict = self.tokenizer(
                    text_value,
                    add_special_tokens=True,
                    max_length=75,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text_value_ids = text_value_ids_dict["input_ids"].squeeze().numpy()
                text_mask = text_value_ids_dict["attention_mask"].squeeze().numpy()
                
                if "token_type_ids" not in text_value_ids_dict.keys():
                    token_type_ids = text_mask
                else:
                    token_type_ids = text_value_ids_dict["token_type_ids"].squeeze().numpy()

                if id_value not in visual_features.keys():
                    continue
                else:
                    # 只用训练集的情感数据
                    sentiment_words_for_one_sample = self.word_sentiment_dict[self.arch][id_value]["words"]

                    # 补零，每个样本的在情感字典中出现的长度设置为10，后续图卷积模块会进行处理
                    if len(sentiment_words_for_one_sample) < 10:
                        sentiment_words_for_one_sample.extend(["[PAD]"] * (10 - len(sentiment_words_for_one_sample)))
                    else:
                        sentiment_words_for_one_sample = sentiment_words_for_one_sample[0:10]

                    # 根据vocab字典，将其转化为ID
                    sentiment_word_ids_for_one_sample = [self.vocab[word] for word in sentiment_words_for_one_sample]
                    
                    t_len = np.array([75])
                    
                    v_mask = np.array([1 for _ in range(197)])
                    v_len = np.array([197])
                    
                    a_mask = np.array([1 for _ in range(75)])
                    a_len = np.array([75])

                    self.data_dict[self.length] = {
                        "text": torch.from_numpy(text_value_ids).squeeze(),
                        "text_mask": torch.tensor(text_mask).squeeze(),
                        "text_segment_ids": torch.tensor(token_type_ids).squeeze(),
                        "text_length": torch.tensor(t_len).squeeze(),
                        "visual": torch.from_numpy(visual_features[id_value]).squeeze().float(),
                        "visual_mask": torch.tensor(v_mask).squeeze(),
                        "visual_length": torch.tensor(v_len).squeeze(),
                        "acoustic": torch.from_numpy(acoustic_features[id_value]).squeeze().float(),
                        "acoustic_mask": torch.tensor(a_mask).squeeze(),
                        "acoustic_length": torch.tensor(a_len).squeeze(),
                        "kb":  torch.tensor(sentiment_word_ids_for_one_sample).squeeze(),
                        "raw_kb":' '.join(sentiment_words_for_one_sample),
                        "label": F.one_hot(torch.tensor(int(label_value)), num_classes=self.number_classes).float(),
                        "id": id_value,
                        "raw_text": text_value,
                    }
                    self.length += 1

    def __getitem__(self, item):
        data = self.data_dict[item]
        return (
            {
                "text": data["text"],
                "text_mask": data["text_mask"],
                "text_segment_ids": data["text_segment_ids"],
                "text_length": data["text_length"],
                "visual": data["visual"],
                "visual_mask": data["visual_mask"],
                "visual_length": data["visual_length"],
                "acoustic": data["acoustic"],
                "acoustic_mask": data["acoustic_mask"],
                "acoustic_length": data["acoustic_length"],
                "kb": data["kb"],
            },
            data["label"],
            {
                "id": data["id"],
                "raw_text": data["raw_text"],
                "raw_kb": data["raw_kb"]
            }
        )

    def __len__(self):
        return self.length


class MOSIDataset(Dataset, ABC):

    def __init__(
        self, all_feature_path, word_sentiment_dict_path=None, vocab_path=None,
        with_text=True, with_visual=True, with_acoustic=True, with_kb=True,
        pretrained_path=None, pretrained_arch="BertTokenizer", 
        is_align=True, need_align=True,
        classes=1,
        arch="train",
    ):
        """
        参数与初始Dataset稍有不同, 该类加载的均为提取之后的特征, 而非原始数据
        text_origin字段方便与情感知识库进行关联
        """
        super(MOSIDataset, self).__init__()
        self.arch = arch
        
        self.is_align=is_align
        self.need_align = need_align
        self.feature_path = os.path.join(
            all_feature_path, 
            "aligned_50.pkl" if self.is_align else "unaligned_50.pkl"
        )
        
        self.with_text = with_text
        self.with_visual = with_visual
        self.with_acoustic = with_acoustic
        self.with_kb = with_kb
        self.word_sentiment_dict = np.load(word_sentiment_dict_path, allow_pickle=True).item()
        self.vocab = np.load(vocab_path, allow_pickle=True).item()

        self.tokenizer = get_class_from_name("transformers." + pretrained_arch).from_pretrained(pretrained_path, do_lower_case=True)

        self.data_dict = {}
        self.length = 0
        self.number_classes = classes

        self.__create_sample_label_dict()

    def __create_sample_label_dict(self):
        def normalization_data(data):
            mu = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            data = (data - mu) / std
            data[np.isnan(data)] = 0
            return data
        
        all_features = np.load(self.feature_path, allow_pickle=True)[self.arch]
        all_features["vision"] = normalization_data(all_features["vision"])
        all_features["audio"] = normalization_data(all_features["audio"])
        if self.is_align:
            all_features["audio_lengths"] = (np.ones((all_features["audio"].shape[0])).astype(int) * 50).tolist()
            all_features["vision_lengths"] = (np.ones((all_features["vision"].shape[0])).astype(int) * 50).tolist()

        for raw_text, acoustic_feature, visual_feature, acoustic_length, visual_length, sample_id, regression_label, classification_label in zip(
            all_features["raw_text"], 
            all_features["audio"], 
            all_features["vision"],
            all_features["audio_lengths"],
            all_features["vision_lengths"],
            all_features["id"], 
            all_features["regression_labels"],
            all_features["classification_labels"]
        ):
            text_value_ids_dict = self.tokenizer(
                raw_text.lower(),
                add_special_tokens=True,
                max_length=50,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_value_ids = text_value_ids_dict["input_ids"].squeeze().numpy()
            text_mask = text_value_ids_dict["attention_mask"].squeeze().numpy()
            if "token_type_ids" not in text_value_ids_dict.keys():
                token_type_ids = text_mask
            else:
                token_type_ids = text_value_ids_dict["token_type_ids"].squeeze().numpy()
            t_len = np.array([50])
            
            v_mask = np.array([1 for _ in range(visual_length)] + [0 for _ in range(visual_length, 500)])
            v_len = np.array([visual_length])
            
            # acoustic length is 375 for CMU-MOSI
            a_mask = np.array([1 for _ in range(acoustic_length)] + [0 for _ in range(acoustic_length, 375)])
            a_len = np.array([acoustic_length])
            
            # 只用训练集的情感数据
            kb_key = sample_id.replace("$_$", "[") + "]"
            if kb_key not in self.word_sentiment_dict[self.arch].keys():
                sentiment_words_for_one_sample = []
            else:
                sentiment_words_for_one_sample = self.word_sentiment_dict[self.arch][kb_key]["words"]

            # 补零，每个样本的在情感字典中出现的长度设置为10，后续图卷积模块会进行处理
            if len(sentiment_words_for_one_sample) < 10:
                sentiment_words_for_one_sample.extend(["[PAD]"] * (10 - len(sentiment_words_for_one_sample)))
            else:
                sentiment_words_for_one_sample = sentiment_words_for_one_sample[0:10]

            # 根据vocab字典，将其转化为ID
            sentiment_word_ids_for_one_sample = [self.vocab[word] for word in sentiment_words_for_one_sample]
            
            self.data_dict[self.length] = {
                "text": torch.from_numpy(text_value_ids).squeeze(),
                "text_mask": torch.tensor(text_mask).squeeze(),
                "text_segment_ids": torch.tensor(token_type_ids).squeeze(),
                "text_length": torch.tensor(t_len).squeeze(),
                "visual": torch.from_numpy(visual_feature).squeeze().float(),
                "visual_mask": torch.tensor(v_mask).squeeze(),
                "visual_length": torch.tensor(v_len).squeeze(),
                "acoustic": torch.from_numpy(acoustic_feature).squeeze().float(),
                "acoustic_mask": torch.tensor(a_mask).squeeze(),
                "acoustic_length": torch.tensor(a_len).squeeze(),
                "kb":  torch.tensor(sentiment_word_ids_for_one_sample).squeeze(),
                "raw_kb":' '.join(sentiment_words_for_one_sample),
                "regression_label": torch.tensor(regression_label).squeeze().float(),
                "classification_label": F.one_hot(torch.tensor(int(classification_label)), num_classes=3).float(),
                "raw_text": raw_text,
                "id": sample_id
            }
            self.length += 1

    def __getitem__(self, item):
        data = self.data_dict[item]
        return (
            {
                "text": data["text"],
                "text_mask": data["text_mask"],
                "text_segment_ids": data["text_segment_ids"],
                "text_length": data["text_length"],
                "visual": data["visual"],
                "visual_mask": data["visual_mask"],
                "visual_length": data["visual_length"],
                "acoustic": data["acoustic"],
                "acoustic_mask": data["acoustic_mask"],
                "acoustic_length": data["acoustic_length"],
                "kb": data["kb"],
            },
            data["regression_label"],
            {
                "id": data["id"],
                "raw_text": data["raw_text"],
                "raw_kb": data["raw_kb"]
            }
        )

    def __len__(self):
        return self.length

    def get_all_data(self):
        return self.data_dict


class MOSEIDataset(MOSIDataset):

    def __init__(
        self, all_feature_path, word_sentiment_dict_path=None, vocab_path=None,
        with_text=True, with_visual=True, with_acoustic=True, with_kb=True,
        pretrained_path=None, pretrained_arch="BertTokenizer", 
        is_align=True, need_align=True,
        classes=1,
        arch="train",
    ):
        super(MOSEIDataset, self).__init__(
            all_feature_path, word_sentiment_dict_path, vocab_path,
            with_text, with_visual, with_acoustic, with_kb,
            pretrained_path, pretrained_arch,
            is_align, need_align,
            classes,
            arch,
        )



# dataset = MOSIDataset(
#     all_feature_path="/home/zhuchuanbo/Documents/datasets/multimodal_sentiment_analysis/MSA-Datasets/MOSI/Processed", 
#     word_sentiment_dict_path="/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/KB/MOSI/word_in_sentiment_dict.npy", 
#     vocab_path="/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/KB/MOSI/word_to_id.npy",
#     with_text=True, with_visual=True, with_acoustic=True, with_kb=True,
#     pretrained_arch="RobertaTokenizer",
#     pretrained_path="/home/zhuchuanbo/Documents/competition/SKEAFN/pretrained_models/roberta-base", 
#     classes=1,
#     arch="train",
#     is_align=True,
# )
# dataset.get_all_data()
# print("123")

# dataset = TwitterDataset(
#     text_params={
#         "pretrained_arch": "RobertaTokenizer",
#         "pretrained_path": "/home/zhuchuanbo/Documents/competition/SKEAFN/pretrained_models/roberta-base"}, 
#     visual_params={
#         "image_size": 224,
#         "data_mean": [0.485, 0.456, 0.406],
#         "data_std": [0.229, 0.224, 0.225],
#         "feature_arch": "vit-base-patch16-224",
#         "feature_path": "/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/features/images"}, 
#     acoustic_params={
#         "feature_path": "/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/features/ocr"}, 
#     kb_params={
#         "vocab_path": "/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/KB/TWITTER/word_to_id.npy",
#         "word_sentiment_dict_path": "/home/zhuchuanbo/Documents/competition/SKEAFN/data/processed/KB/TWITTER/word_in_sentiment_dict.npy"}, 
#     label_params={
#          "dir": "/home/zhuchuanbo/Documents/competition/SKEAFN/data/origin"},
#     with_text=True, with_visual=True, with_acoustic=True, with_kb=True,
#     classes=2,
#     arch="train",
# )
