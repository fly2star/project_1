def build_label_graph(l_t):
    """根据标签共现关系构建邻接矩阵"""
    co_occurrence = (l_t.T @ l_t)  # shape [c, c]
    adj = (co_occurrence > 0).float()
    return adj

def compute_label_weight(adj, y_all_labels):
    """用简单的线性层或统计计算每个标签的权重"""
    deg = adj.sum(dim=1)
    weight = deg / deg.max()
    return weight

def compute_affinity_matrix(l_t, weight):
    """根据标签权重计算 S 矩阵"""
    numerator = (l_t * weight) @ (l_t * weight).T
    denominator = (l_t * weight).sum(dim=1, keepdim=True) + (l_t * weight).sum(dim=1, keepdim=True).T + 1e-8
    S = numerator / denominator
    return S
