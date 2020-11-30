import numpy as np

def batch_rank_eval(answer, scores, hitAT=(1, 3, 5, 10, 20, 30)):
    """
    :param answer:  [batch]
    :param scores:  [batch, labels]
    :return:
    """
    answer = np.array(answer)
    answer = np.reshape(answer,[-1])
    # 取反后才能得到值
    scores = np.array(scores) * -1
    argindex = np.argsort(np.argsort(scores)) + 1
    ranks = [argindex[x,answer[x]] for x in range(len(answer))]
    reversed_ranks = [1/x for x in ranks]
    hits = []
    for k in hitAT:
        tmp = [1 if x <= k else 0 for x in ranks]
        hits.append(tmp)
    return ranks, reversed_ranks, hits

def batch_top_k(scores, labels, k=10):
    """
    :param scores:  [batch, N]
    :param labels:  [batch, N]
    :return:
    """
    scores = np.array(scores)
    scores *= -1
    argindexs = np.argsort(scores)
    res_index = []
    res_label = []
    for argindex, label in zip(argindexs, labels):
        index = argindex[0:k]
        index = list(index) + [0] * (k - len(index))
        score = [label[x] for x in index]
        res_index.append(index)
        res_label.append(score)

    return res_index, res_label

a = [2,2]
scores = [[5,2,4,3],[9,33,6,4]]
labels = scores # [[1,2,3,4],[1,2,3,4]]
print(batch_top_k(scores, labels))