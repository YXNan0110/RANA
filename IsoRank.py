import numpy as np
from numpy import inf
from copy import deepcopy
import argparse
import os
from network_alignment_model import NetworkAlignmentModel
from dataset import Dataset
import graph_utils
from graph_utils import get_H, normalize_matrix
class IsoRank(NetworkAlignmentModel):

    """
    Description:
      The algorithm computes the alignment/similarity matrix by a random walk
      based method. This algorithm is for non-attributed networks.
    Input:
      - A1, A2: adjacency matrices of two networks
      - H: the prior node similarity matrix, e.g., degree similarity matrix
      - alpha: decay factor, i.e., how important the global topology
               consistency is
      - maxiter: maximum number of iterations
    Output:
      - S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
       x in A2 is aligned to node-y in A1
    Reference:
      Singh, Rohit, Jinbo Xu, and Bonnie Berger.
      Global alignment of multiple protein interaction networks with application to functional orthology detection.
      Proceedings of the National Academy of Sciences 105.35 (2008): 12763-12768.
    """

    def __init__(self, source_dataset, target_dataset, H=None, alpha=0.82, maxiter=30, tol=1e-4, train_dict=None):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alignment_matrix = None
        self.A1 = source_dataset.get_adjacency_matrix()
        self.A2 = target_dataset.get_adjacency_matrix()
        self.alpha = alpha
        self.maxiter = maxiter
        # if train_dict is not None:
            # print("This is supervised IsoRank")
        self.H = get_H(H, source_dataset, target_dataset, train_dict)
        self.tol = tol

    def align(self):

        n1 = self.A1.shape[0]
        n2 = self.A2.shape[0]

        # normalize the adjacency matrices
        d1 = 1 / self.A1.sum(axis=1)
        d2 = 1 / self.A2.sum(axis=1)

        d1[d1 == inf] = 0
        d2[d2 == inf] = 0
        d1 = d1.reshape(-1,1)
        d2 = d2.reshape(-1,1)

        W1 = d1*self.A1
        W2 = d2*self.A2
        S = np.ones((n2,n1)) / (n1 * n2) # Map target to source
        # IsoRank Algorithm in matrix form
        for iter in range(1, self.maxiter + 1):
            prev = S.flatten()
            if self.H is not None:
                S = (self.alpha*W2.T).dot(S).dot(W1) + (1-self.alpha) * self.H
            else:
                S = W2.T.dot(S).dot(W1)
            delta = np.linalg.norm(S.flatten()-prev, 2)
            # print("Iteration: ", iter, " with delta = ", delta)
            if delta < self.tol:
                continue
                break

        self.alignment_matrix = S.T
        S_normalized = normalize_matrix(self.alignment_matrix)
        return self.alignment_matrix, S_normalized

    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="IsoRank")
    parser.add_argument('--prefix1',             default="../graph_data/douban/offline/graphsage")
    parser.add_argument('--prefix2',             default="../graph_data/douban/online/graphsage")
    parser.add_argument('--groundtruth',         default="../graph_data/douban/dictionaries")
    parser.add_argument('--dataset',             default='douban')
    parser.add_argument('--H',                   default=None)
    parser.add_argument('--max_iter',            default=30, type=int)
    parser.add_argument('--alpha',               default=0.82, type=float)
    parser.add_argument('--tol',                 default=1e-4, type=float)
    parser.add_argument('--rate',                default=0.01, type=float)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    source_dataset = Dataset(args.prefix1)
    target_dataset = Dataset(args.prefix2)
    train_dict = graph_utils.load_gt(os.path.join(args.groundtruth, f"node,split={args.rate}.train.dict"),
                                    source_dataset.id2idx, target_dataset.id2idx, 'dict', args.dataset)

    model = IsoRank(source_dataset, target_dataset, None, args.alpha, args.max_iter, args.tol, train_dict)
    S = model.align()



