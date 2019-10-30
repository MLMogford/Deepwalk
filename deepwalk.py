
from collections import defaultdict
import data_utils_cora
import numpy as np
from random import Random
from six import iteritems
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


class Graph():

# graph builder
    def __init__(self, A):
        self.graph = defaultdict(list)
        for i in range(A.shape[0]):
            for vertex in A[i].indices:
                self.graph[i].append(vertex)
        self.vector = self.graph.keys()

    def sparse(x):
        graph = defaultdict(lambda: set())
        adj = x.tocoo()
        for i, j, vertex in zip(adj.row, adj.col, adj.data):
            graph[i].add(j)
        return {str(k): [str(x) for x in vertex] for k, vertex in iteritems(graph)}

# random walk generator - random walks on graph nodes to generate node sequences
    def randomWalk(self, walk_length, start, alpha=0):
        rand = Random()
        path = [start]
        while len(path) < walk_length:
            cur = path[len(path) - 1]
            v = self.graph[cur]
            if len(v) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(v))
                else:
                    path.append(path[0])
            else:
                return [str(vect) for vect in path]
        return [str(vect) for vect in path]


def deepWalk(_windowSize=5, _embeddingSize=128, _walkLength=35):
    X, A, y = data_utils_cora.load_data(dataset='cora')
    graph = Graph(A)

    walk = []
    vector = list(graph.vector)

#build corpus from random walks
    for vect in range (0, len(vector)):
        walk.append(graph.randomWalk(_walkLength, vect))

#set hyperparameters - word2vec utilises the skipgram algorithm to create word embeddings
    model = Word2Vec(walk, size=_embeddingSize, window=_windowSize, min_count=0, sg=1, hs=1, workers=4)

    G = Graph.sparse(A)

    y = np.ravel(np.array([np.where(y[i] == 1)[0] for i in range(y.shape[0])]))
    X = np.array([model.wv[str(i)] for i in range(len(G))])
    features = np.asarray([model[str(X)] for X in range(len(Graph.sparse(A)))])

    y_train, y_val, y_test, idx_train, idx_val, idx_test = data_utils_cora.get_splits(y)
    x_train, x_val, x_test, idx_train, idx_val, idx_test = data_utils_cora.get_splits(X)
    test = LogisticRegression(max_iter=500, multi_class='ovr').fit(features[idx_train], y[idx_train].ravel())
    test.fit(x_train, y_train)
    print(test.score(x_train, y_train))
    print(test.score(x_test, y_test))


deepWalk()
