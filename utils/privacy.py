import torch
import torch.nn as nn

import numpy as np

from utils.eval import get_output_for_batch


def thresholding_based_inference_attack(train_conf, test_conf, consider_all=True):
    list_all = np.concatenate((train_conf, test_conf))
    max_gap = 0
    thre_chosen = 0
    for thre in list_all if consider_all else test_conf:
        ratio1 = np.sum(train_conf >= thre) / len(train_conf)
        ratio2 = np.sum(test_conf >= thre) / len(test_conf)
        if np.absolute(ratio1 - ratio2) > max_gap:
            max_gap = np.absolute(ratio1 - ratio2)
            thre_chosen = thre
    return 0.5 * (1 + max_gap), thre_chosen


def get_inference_accuracy(
    model, train_loader, test_loader, device, consider_all=False, temp=1, verbose=False
):
    """
    all: whether to iterate over both list, over just test_conf. 
    """
    print("Assuming that model(x) returns logits instead of probs ")

    print(
        "Switching to evaluation model using .eval() for inferency accuracy calculation."
    )
    model.eval()

    train_conf = []
    test_conf = []
    for img, _ in train_loader:
        train_conf += list(get_output_for_batch(model, img.to(device), temp)[0])
    for img, _ in test_loader:
        test_conf += list(get_output_for_batch(model, img.to(device), temp)[0])

    a, t = thresholding_based_inference_attack(train_conf, test_conf, consider_all)
    if verbose:
        print("inference accuracy is: ", a)
        print("chosen threshold is: ", t)
    return a, t
