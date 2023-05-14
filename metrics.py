import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


def calculate_metrics_for_classification(outputs, targets, num_classes):
    """
    计算模型的准确率、召回率、F1等指标
    :param outputs: 模型输出结果, 为 N * num_classes
    :param targets: 样本标签, 为 N * number_classes
    :param num_classes
    :return: 预测准确度
    """
    targets = np.argmax(targets, axis=-1)

    accuracy = accuracy_score(targets, outputs)

    precision_every_class = precision_score(targets, outputs, average=None)
    precision_macro = precision_score(targets, outputs, average="macro")
    precision_micro = precision_score(targets, outputs, average="micro")
    precision_weighted = precision_score(targets, outputs, average="weighted")
    precision_default = precision_score(targets, outputs)
    precision_every_class_dict = dict(
        zip(["class_{}".format(i) for i in range(num_classes)], precision_every_class)
    )

    recall_every_class = recall_score(targets, outputs, average=None)
    recall_macro = recall_score(targets, outputs, average="macro")
    recall_micro = recall_score(targets, outputs, average="micro")
    recall_weighted = recall_score(targets, outputs, average="weighted")
    recall_default = recall_score(targets, outputs)
    recall_every_class_dict = dict(
        zip(["class_{}".format(i) for i in range(num_classes)], recall_every_class)
    )

    f1_every_class = f1_score(targets, outputs, average=None)
    f1_macro = f1_score(targets, outputs, average="macro")
    f1_micro = f1_score(targets, outputs, average="micro")
    f1_weighted = f1_score(targets, outputs, average="weighted")
    f1_default = f1_score(targets, outputs)
    f1_every_class_dict = dict(
        zip(["class_{}".format(i) for i in range(num_classes)], f1_every_class)
    )

    # precision, recall, f1, _ = precision_recall_fscore_support(targets,
    #                                                            outputs,
    #                                                            labels=[i for i in range(num_classes)],
    #                                                            average='micro',
    #                                                            zero_division=0)
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "precision": {
            "default": precision_default,
            "weighted": precision_weighted,
            "macro": precision_macro,
            "micro": precision_micro,
            **precision_every_class_dict,
        },
        "recall": {
            "default": recall_default,
            "weighted": recall_weighted,
            "macro": recall_macro,
            "micro": recall_micro,
            **recall_every_class_dict,
        },
        "f1": {
            "default": f1_default,
            "weighted": f1_weighted,
            "macro": f1_macro, 
            "micro": f1_micro, 
            **f1_every_class_dict
        },
    }


def print_confusion_matrix_for_classification(image_labels, image_predictions):
    """
    打印混淆矩阵
    :param image_labels:
    :param image_predictions:
    :return:
    """
    image_labels = np.argmax(image_labels, axis=-1)
    print(
        classification_report(
            y_true=np.array(image_labels), y_pred=np.array(image_predictions)
        )
    )
    print(
        confusion_matrix(
            y_true=np.array(image_labels), y_pred=np.array(image_predictions)
        )
    )


def calculate_metrics_for_regression(y_pred, y_true):
    """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        y_true: shape of <B>
        y_pred: shape of <B>
    """

    def multiclass_acc(preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    test_preds = np.array(y_pred)
    test_truth = np.array(y_true)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # pos - neg
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    mult_a2_pos_neg = accuracy_score(binary_truth, binary_preds)
    f_score = f1_score(binary_preds, binary_truth, average="weighted")

    # if to_print:
    #     print("mae: ", mae)
    #     print("corr: ", corr)
    #     print("mult_acc: ", mult_a7)
    #     print("Classification Report (pos/neg) :")
    #     print(classification_report(binary_truth, binary_preds, digits=5))
    #     print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))

    # non-neg - neg
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    mult_a2_non_neg_neg = accuracy_score(binary_truth, binary_preds)
    f1_a2_non_neg_neg = f1_score(binary_preds, binary_truth, average="weighted")

    # if to_print:
    #     print("Classification Report (non-neg/neg) :")
    #     print(classification_report(binary_truth, binary_preds, digits=5))
    #     print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))

    # return accuracy_score(binary_truth, binary_preds)

    return {
        "mult_a2_non_neg_neg": mult_a2_non_neg_neg,
        "accuracy": mult_a2_pos_neg,
        "mult_a5": mult_a5,
        "mult_a7": mult_a7,
        "f1": f_score,
        "f1_a2_non_neg_neg": f1_a2_non_neg_neg,
        "mae": mae,
        "corr": corr,
    }


def print_confusion_matrix_for_regression(y_true, y_pred):
    """
    打印混淆矩阵
    :param image_labels:
    :param image_predictions:
    :return:
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    binary_truth = y_true >= 0
    binary_preds = y_pred >= 0
    print(classification_report(y_true=binary_truth, y_pred=binary_preds, digits=5))
    print(confusion_matrix(y_true=binary_truth, y_pred=binary_preds))

# 统一接口
def calculate_metrics(outputs, targets, num_classes, dataset_name=None):
    if num_classes != 1:
        # 分类问题
        return calculate_metrics_for_classification(outputs, targets, num_classes)
    else:
        # 回归问题
        return calculate_metrics_for_regression(outputs, targets)


def print_confusion_matrix(image_labels, image_predictions, num_classes):
    if num_classes != 1:
        # 分类问题
        return print_confusion_matrix_for_classification(image_labels, image_predictions)
    else:
        # 回归问题
        return print_confusion_matrix_for_regression(image_labels, image_predictions)
