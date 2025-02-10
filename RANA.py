import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from metrics import get_statistics
from graph_utils import get_H
import graph_utils as graph_utils
import argparse
import os
import copy
from dataset import Dataset
from FINAL import FINAL
import scipy.sparse as sp
import torch
import random
import numpy.linalg as la

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="FINAL")
    parser.add_argument('--prefix1',             default="graph_data/douban/offline/graphsage/")
    parser.add_argument('--prefix2',             default="graph_data/douban/online/graphsage/")    
    parser.add_argument('--dataset',             default='douban')
    parser.add_argument('--groundtruth',        default="graph_data/douban/dictionaries/")
    parser.add_argument('--max_iter',         default=30, type=int)
    parser.add_argument('--alpha',          default=0.82, type=float)
    parser.add_argument('--tol',          default=1e-4, type=float)
    parser.add_argument('--rate', default=0.1, type=float)  # training rate
    parser.add_argument('--oracle_acc', default=0.7, type=float)
    parser.add_argument('--AL_batch_size', default=10, type=int)
    parser.add_argument('--th', default=0.98, type=float)  # 相似度大于这个阈值就不作为available_pair 大一些能够尽可能保留更多的可选节点对
    parser.add_argument('--gamma', default=0.05, type=float)
    parser.add_argument('--theta', default=0.9, type=float)  # 可信度指标
    parser.add_argument('--max_al_num', default=300, type=int)
    parser.add_argument('--noise_rate', default=0, type=int)  # 添加噪声边比例

    return parser.parse_args()

def compute_cos_sim(vec_a,vec_b):
    return (vec_a.dot(vec_b.T))/(la.norm(vec_a)*la.norm(vec_b))

def left_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def calculate_similarity(similarity_feature, aax_feature):
    similarity_feature = cosine_similarity(aax_feature)
    dis_range = np.max(similarity_feature) - np.min(similarity_feature)
    similarity_feature = (similarity_feature - np.min(similarity_feature)) / dis_range

    return similarity_feature

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item

def clean_score(pair, adj1_csr, adj2_csr, similarity1, similarity2):
    node1, node2 = pair
    # 计算节点1在图1中的邻居节点的特征相似性
    neighbors1 = adj1_csr[node1].indices  # 图1中node1的邻居节点
    sum_similarity1 = np.sum(similarity1[node1, neighbors1])/neighbors1.shape[0]  # 求和
    # 计算节点2在图2中的邻居节点的特征相似性
    neighbors2 = adj2_csr[node2].indices  # 图2中node2的邻居节点
    sum_similarity2 = np.sum(similarity2[node2, neighbors2])/neighbors2.shape[0]  # 求和
    # 计算归一化清洁性分数
    clean_score_value = sum_similarity1 + sum_similarity2
    return clean_score_value/2

def get_activated_node_init(pair, score1, activated_nodes): 
    n1, n2 = pair
    activated_vector1=(adj1_matrix2[n1] > args.gamma)+0
    activated_vector2=(adj2_matrix2[n2] > args.gamma)+0
    activated_vector=np.concatenate((activated_vector1, activated_vector2))*activated_nodes
     # 将 clean_score 与 activated_vector 逐元素相乘
    adjusted_activated_vector = activated_vector * score1
    # 判断是否激活节点：乘积大于阈值则激活
    activated_vector = (adjusted_activated_vector >= args.theta) + 0
    activated_cnt=num_ones.dot(activated_vector)
    return activated_cnt,activated_vector

def get_max_info_nodes_init(available_pairs, activated_nodes): 
    max_influence_pair = 0
    max_activated_nodes = 0
    max_activated_num = 0 
    for pair in available_pairs:
        cl_score = clean_score(pair, adj_g1, adj_g2, similarity_feature_g1, similarity_feature_g2)
        activated_num, activated_node_tmp = get_activated_node_init(pair, cl_score, activated_nodes)
        if activated_num > max_activated_num:
            max_activated_num = activated_num
            max_influence_pair = pair
            max_activated_nodes = activated_node_tmp        
    return max_influence_pair, max_activated_nodes, max_activated_num

def get_activated_node(pair, activated_nodes, Confidence_matrix): 
    a, b = pair
    activated_vector1=(adj1_matrix2[a]*Confidence_matrix[a][b] >= args.gamma)+0
    activated_vector2=(adj2_matrix2[b]*Confidence_matrix[a][b] >= args.gamma)+0
    activated_vector=np.concatenate((activated_vector1, activated_vector2))*activated_nodes
    # 判断是否激活节点：乘积大于阈值则激活
    # 让模型更倾向于选择更相似的两个节点作为节点对
    # adjusted_activated_vector = activated_vector * compute_cos_sim(feature_g1_cpu[a], feature_g2_cpu[b])
    activated_vector = (activated_vector >= args.theta) + 0
    activated_cnt=num_ones.dot(activated_vector)
    return activated_cnt,activated_vector

def get_max_info_nodes(available_pairs, activated_nodes, Confidence_matrix): 
    max_influence_pair = 0
    max_activated_nodes = 0
    max_activated_num = 0 
    for pair in available_pairs:
        if flag[pair[0]] == 0 and flag[pair[1]] == 0:
            activated_num, activated_node_tmp =get_activated_node(pair, activated_nodes, Confidence_matrix)
            if activated_num > max_activated_num:
                max_activated_num = activated_num
                max_influence_pair = pair
                max_activated_nodes = activated_node_tmp
        else:
            continue
    a = max_influence_pair[0]
    b = max_influence_pair[1]
    if labels[a][b] == 1:
        flag[a] = 1
        flag[b] = 1        
    return max_influence_pair, max_activated_nodes, max_activated_num

def begin_self_labeling():
    model = FINAL(source_dataset, target_dataset, None, args.alpha, args.max_iter, args.tol, e_train_dict)
    S, normalized_S = model.align()  # 对齐概率矩阵
    # print(S)
    acc, MAP, top5, top10, S_pairs = get_statistics(S, groundtruth, use_greedy_match=True, get_all_metric=True)
    print("Acc:", acc, "MAP:", MAP, "Top5:", top5, "Top10:", top10)
    model_r = acc
    model_belief_list = np.zeros(n1 * n2)
    self_label_pairs = list()
    for pair in available_pairs:
        a, b = pair
        bt = normalized_S[a][b] * model_r  # 如果模型对单个节点的预测*模型整体的预测准确率大于oracle_acc，就认为预测是可用的
        model_belief_list[a * b] = bt
        if bt >= args.oracle_acc:
            self_label_pairs.append(pair)  # 模型预测的节点集合
            labels[a][b] = S_pairs[a][b]  # 经过贪婪选择的对齐矩阵
    return self_label_pairs, model_belief_list, S_pairs  # model_belief_list中保存所有avaliable_pairs的模型预测可信度

def find_mirror_pair(i, j):
    # Step 1: Find the most similar node to i in graph 1 and to j in graph 2
    # Find the index of the second most similar node to i in graph 1
    i_temp = similarity_feature_g1[i, :].copy()  # Make a copy to avoid modifying the original array
    i_temp[np.argmax(i_temp)] = -np.inf  # Exclude the maximum value
    i_prime = np.argmax(i_temp)  # Find the second largest value

    # Find the index of the second most similar node to j in graph 2
    j_temp = similarity_feature_g2[j, :].copy()  # Make a copy to avoid modifying the original array
    j_temp[np.argmax(j_temp)] = -np.inf  # Exclude the maximum value
    j_prime = np.argmax(j_temp)  # Find the second largest value
    
    # Step 2: Calculate the cross-graph similarity between (i, j') and (i', j)
    sim_ij_prime = similarity_feature_g1[i, j_prime] * similarity_feature_g2[j, j_prime]
    sim_i_primej = similarity_feature_g1[i, i_prime] * similarity_feature_g2[j, j]
    
    # Step 3: Find the best cross-graph pair
    if sim_ij_prime >= sim_i_primej:
        best_i_prime, best_j_prime = i, j_prime
    else:
        best_i_prime, best_j_prime = i_prime, j
    
    return best_i_prime, best_j_prime


def update_Confidence(count, cnt_batch, self_labeled_pairs, belief_list, S_pairs):
    activated_nodes = np.zeros(n1 + n2)
    cnt = count
    if len(self_labeled_pairs) != 0:
        for pair in self_labeled_pairs:
            a, b = pair
            activated_vector1=(adj1_matrix2[a]*Confidence_matrix[a][b] >= args.gamma)+0
            activated_vector2=(adj2_matrix2[b]*Confidence_matrix[a][b] >= args.gamma)+0
            for i in range(len(activated_vector1)):
                activated_nodes[i] = max(activated_nodes[i], activated_vector1[i])
            for i in range(len(activated_vector2)):
                activated_nodes[len(activated_vector1) + i] = max(activated_nodes[len(activated_vector1) + i], activated_vector2[i])
    for pair in train_list[0:len(train_list)+count-cnt_batch]:  # 这个batch前的所有train_pair激活
        a, b = pair
        activated_vector1=(adj1_matrix2[a]*Confidence_matrix[a][b] >= args.gamma)+0
        activated_vector2=(adj2_matrix2[b]*Confidence_matrix[a][b] >= args.gamma)+0
        for i in range(len(activated_vector1)):
            activated_nodes[i] = max(activated_nodes[i], activated_vector1[i])
        for i in range(len(activated_vector2)):
            activated_nodes[len(activated_vector1) + i] = max(activated_nodes[len(activated_vector1) + i], activated_vector2[i])
        # activated_nodes=np.concatenate((activated_vector1, activated_vector2))*activated_nodes
    for pair in train_list[count-cnt_batch:count]:  # this batch
        a, b = pair
        confidence_score = 0
        oracle_label = labels[a][b]
        model_label = S_pairs[a][b]
        if oracle_label == model_label:
            confidence_score = (args.oracle_acc * belief_list[a * b])/((1 - args.oracle_acc) * (1 - belief_list[a * b]) + (args.oracle_acc * belief_list[a * b]))
        if oracle_label != model_label:
            if belief_list[a * b] <= 0.0001:
                confidence_score = args.oracle_acc
            if belief_list[a * b] > 0.0001:
                mirror_a, mirror_b = find_mirror_pair(a, b)
                mirror_label = labels[mirror_a][mirror_b]
                if mirror_label == oracle_label:
                    confidence_score = (1 - belief_list[a * b]) * args.oracle_acc / (1 - args.oracle_acc * belief_list[a * b])
                else:
                    confidence_score = (1 - args.oracle_acc) * belief_list[a * b] / (1 - args.oracle_acc * belief_list[a * b])
        Confidence_matrix[a][b] = confidence_score
        activated_vector1=(adj1_matrix2[a]*Confidence_matrix[a][b] >= args.gamma)+0
        activated_vector2=(adj2_matrix2[b]*Confidence_matrix[a][b] >= args.gamma)+0
        for i in range(len(activated_vector1)):
            activated_nodes[i] = max(activated_nodes[i], activated_vector1[i])
        for i in range(len(activated_vector2)):
            activated_nodes[len(activated_vector1) + i] = max(activated_nodes[len(activated_vector1) + i], activated_vector2[i])
    count = cnt
    return activated_nodes



args = parse_args()
print(args)

source_dataset = Dataset(args.prefix1, args.noise_rate)
target_dataset = Dataset(args.prefix2, args.noise_rate)

groundtruth = graph_utils.load_gt(os.path.join(args.groundtruth, f"groundtruth"), source_dataset.id2idx, target_dataset.id2idx, 'dict', args.dataset)
print("Anchor links: ", len(groundtruth))
adj_g1 = sp.csr_matrix(source_dataset.get_adjacency_matrix())
adj_g2 = sp.csr_matrix(target_dataset.get_adjacency_matrix())
# if source_dataset.features:
features_g1 = torch.tensor(source_dataset.features, dtype=torch.float32).cuda()
features_g2 = torch.tensor(target_dataset.features, dtype=torch.float32).cuda()
feature_dim = source_dataset.features.shape[1]
print("Node feature dim: ", source_dataset.features.shape[1])
n1, n2 = len(source_dataset.G.nodes()), len(target_dataset.G.nodes())  # num_nodes

train_dict = graph_utils.load_gt(os.path.join(args.groundtruth, f"node,split={args.rate}.train.dict"),
                                    source_dataset.id2idx, target_dataset.id2idx, 'dict', args.dataset)


# 归一化邻接矩阵
adj1 = left_normalized_adjacency(adj_g1)  # 单边归一化
adj2 = left_normalized_adjacency(adj_g2)
adj1_matrix = torch.FloatTensor(adj1.todense()).cuda()
adj2_matrix = torch.FloatTensor(adj2.todense()).cuda()
adj1_matrix2 = torch.mm(adj1_matrix,adj1_matrix).cuda()
adj2_matrix2 = torch.mm(adj2_matrix,adj2_matrix).cuda()
# if source_dataset.features != None:
aax1_feature = torch.mm(adj1_matrix2,features_g1)  # 用于计算相似性 综合了邻接矩阵和特征的相似性
aax2_feature = torch.mm(adj2_matrix2,features_g2)
aax1_feature = np.array(aax1_feature.cpu())
adj1_matrix2 = np.array(adj1_matrix2.cpu())  # g1的二阶邻接矩阵
aax2_feature = np.array(aax2_feature.cpu())
adj2_matrix2 = np.array(adj2_matrix2.cpu())  # g2的二阶邻接矩阵
feature_g1_cpu = np.array(features_g1.cpu())
feature_g2_cpu = np.array(features_g2.cpu())
'''
else:
    aax1_feature = np.array(adj1_matrix2.cpu())
    aax2_feature = np.array(adj2_matrix2.cpu())
    adj1_matrix2 = aax1_feature
    adj2_matrix2 = aax2_feature
'''
    
# calculate normalized similarity
similarity_feature_g1 = np.ones((n1, n1))
similarity_feature_g1 = calculate_similarity(similarity_feature_g1, aax1_feature)
similarity_feature_g2 = np.ones((n2, n2))
similarity_feature_g2 = calculate_similarity(similarity_feature_g2, aax2_feature)

available_g1 = list(range(n1))
available_g2 = list(range(n2))

for i in train_dict.keys():
    for j in range(n1):
        if similarity_feature_g1[i][j]>args.th and j in available_g1:
            available_g1.remove(j)
for i in train_dict.values():
    for j in range(n2):
        if similarity_feature_g2[i][j]>args.th and j in available_g2:
            available_g2.remove(j)
'''
for i in train_dict.keys():
    available_g1.remove(i)
for i in train_dict.values():
    available_g2.remove(i)
'''
available_pairs = list()
for i in available_g1:
    for j in available_g2:
        available_pairs.append((int(i), int(j)))  # 经过相似度筛选得到的用于选择的节点对
print("Available_pairs: ", len(available_g1), "*", len(available_g2))
'''
init_Confidence_matrix = np.ones((n1, n2)) * args.oracle_acc
for i in range(n1):
    for j in range(n2):
        pair = (i, j)
        cl_score = clean_score(pair, adj_g1, adj_g2, similarity_feature_g1, similarity_feature_g2)
        if cl_score < args.oracle_acc:
            init_Confidence_matrix[i][j] = cl_score
np.save("graph_data/douban/Confidence_matrix_douban.npy", init_Confidence_matrix)
'''
# Initialize Confidence score
# Confidence_matrix = np.ones((n1, n2)) * args.oracle_acc
Confidence_matrix = np.load('graph_data/douban/Confidence_matrix_douban.npy')
for k, v in train_dict.items():
    Confidence_matrix[k][v] = 1

# alignment label
labels = np.zeros((n1, n2))
for i in groundtruth.keys():
    labels[i][groundtruth[i]] = 1

label_list=[]
prob_list = np.full((2, 2),(1-args.oracle_acc)).tolist()
for i in range(2):
    label_list.append(i)
    prob_list[i][i]=args.oracle_acc
for pair in available_pairs:
    labels[pair[0]][pair[1]]=torch.tensor(random_pick(label_list,prob_list[int(labels[pair[0]][pair[1]])]))  # labels already have noise
# print(labels)


count = len(train_dict)
batch_node_num = 0
pairs_label = [1] * count
activated_nodes = np.ones(n1 + n2)  # 两个图上激活的节点集合
flag = np.zeros(n1 + n2)  # 被选择过的节点，并且pair_label=1就不能再选择了，0表示没有被选择过
train_list = [(k, v) for k, v in train_dict.items()]
effective_train_list = copy.deepcopy(train_list)
num_ones = np.ones(n1 + n2)
while True:
    t1 = time.time()
    if count == 0:
        max_influence_pair, max_activated_nodes, max_activated_num = get_max_info_nodes_init(available_pairs, activated_nodes)
    else:
        max_influence_pair, max_activated_nodes, max_activated_num = get_max_info_nodes(available_pairs, activated_nodes, Confidence_matrix)
    if max_activated_num <= 0:
        break 
    train_list.append(max_influence_pair)  # train_list记录所有被选中的节点对，effective_train_list记录有链接的节点对
    available_pairs.remove(max_influence_pair)
    pair_label = labels[max_influence_pair[0]][max_influence_pair[1]].item()
    pairs_label.append(pair_label)
    if pair_label == 1:
        effective_train_list.append(max_influence_pair)
    # pairs in train_list have labels 0 or 1
    e_train_dict = dict(effective_train_list)
    count += 1
    batch_node_num  += 1
    if count % args.AL_batch_size == 0:
        # labels = torch.tensor(labels).cuda()
        # train_list = torch.tensor(train_list).cuda()
        self_labeled_pairs, model_belief_list, S_pairs = begin_self_labeling()
        if len(self_labeled_pairs) != 0:
            print("Self labeled pairs num:", len(self_labeled_pairs))
            for pair in self_labeled_pairs:
                a, b = pair
                Confidence_matrix[a][b] = model_belief_list[a*b]
        activated_nodes = update_Confidence(count, batch_node_num, self_labeled_pairs, model_belief_list, S_pairs)
        if len(self_labeled_pairs) != 0:
            for pair in self_labeled_pairs:
                train_list.append(pair)
                available_pairs.remove(pair)
                pair_label = S_pairs[pair[0]][pair[1]]
                pairs_label.append(pair_label)
                if pair_label == 1:
                    effective_train_list.append(pair)
                    flag[pair[0]] = 1
                    flag[pair[1]] = 1
        batch_node_num = 0
    print('Pair '+str(max_influence_pair)+' is selected, with '+str(max_activated_num)+' nodes activated')
    print("select time elapsed: {:.4f}s".format(time.time() - t1))
    print("Effective Pairs:", len(effective_train_list))
    for i in range(len(max_activated_nodes)):
        if max_activated_nodes[i] > 0 and activated_nodes[i] > 0:  # 将上一轮选中的激活节点从激活节点中移除，如果不在激活节点中就不用移除
            activated_nodes[i] = activated_nodes[i] - max_activated_nodes[i]
    activated_nodes = np.ones(n1 + n2)-((activated_nodes>0)+0)
    # activated_nodes = activated_nodes - max_activated_nodes
    if len(effective_train_list) >= (args.max_al_num + len(train_dict)):
        break

e_train_dict = dict(effective_train_list)

print("-------Evaluation--------")
t_total = time.time()
model = FINAL(source_dataset, target_dataset, None, args.alpha, args.max_iter, args.tol, e_train_dict)
S, _ = model.align()  # 对齐概率矩阵
acc, MAP, top5, top10 = get_statistics(S, groundtruth, use_greedy_match=False, get_all_metric=True)
print("Accuracy: {:.4f}".format(acc))
print("MAP: {:.4f}".format(MAP))
print("Precision_5: {:.4f}".format(top5))
print("Precision_10: {:.4f}".format(top10))
print(len(train_list))






