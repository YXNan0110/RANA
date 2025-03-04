import numpy as np
from matcher import top_k, greedy_match
from scipy.sparse import csr_matrix


def get_nn_alignment_matrix(alignment_matrix):
    # Sparse
    row = np.arange(len(alignment_matrix))
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result


def get_statistics(alignment_matrix, groundtruth, groundtruth_matrix=None, use_greedy_match=False, get_all_metric = False):
    if use_greedy_match:
        # print("This is greedy match accuracy")
        S_pairs = greedy_match(alignment_matrix)   # return a metric with alignment pairs
    pred = get_nn_alignment_matrix(alignment_matrix)  # create a max value csr matrix
    acc = compute_accuracy(pred, groundtruth)

    if get_all_metric:
        MAP, Hit, AUC  = compute_MAP_Hit_AUC(alignment_matrix, groundtruth)
        pred_top_5 = top_k(alignment_matrix, 5)
        top5 = compute_precision_k(pred_top_5, groundtruth)
        pred_top_10 = top_k(alignment_matrix, 10)
        top10 = compute_precision_k(pred_top_10, groundtruth)
        if use_greedy_match:
            return acc, MAP, top5, top10, S_pairs
        return acc, MAP, top5, top10
    return acc, S_pairs

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            if top_k_matrix[key, value] == 1:
                n_matched += 1
        return n_matched/len(gt)

    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1

    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_accuracy(pred, gt):
    n_matched = 0
    if type(gt) == dict:
        for key, value in gt.items():
            if pred[key, value] == 1:
                n_matched += 1
        return n_matched/len(gt)

    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_MAP_Hit_AUC(alignment_matrix, gt):
    MAP = 0
    AUC = 0
    Hit = 0
    for key, value in gt.items():
        ele_key = alignment_matrix[key].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == value:
                ra = i + 1 # r1
                MAP += 1/ra
                Hit += (alignment_matrix.shape[1] + 1) / alignment_matrix.shape[1]
                AUC += (alignment_matrix.shape[1] - ra) / (alignment_matrix.shape[1] - 1)
                break
    n_nodes = len(gt)
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, Hit, AUC
