import os
import random
import numpy as np


def split_areas(areas, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    assert train_ratio + valid_ratio + test_ratio == 1
    random.shuffle(areas)
    n = len(areas)
    train_areas = areas[:int(n * train_ratio)]
    valid_areas = areas[int(n * train_ratio):int(n * (train_ratio + valid_ratio))]
    test_areas = areas[int(n * (train_ratio + valid_ratio)):]
    return train_areas, valid_areas, test_areas


# def construct_train(city_path, train_areas):
#     attr = np.load(f"{city_path}/attr.npy")    # [area, feat_dim]
#     dis = np.load(f"{city_path}/dis.npy")      # [area, area]
#     od = np.load(f"{city_path}/od.npy")        # [area, area]

#     idx_o = [int(a) for a in train_areas]
#     idx_d = [int(a) for a in train_areas]

#     dis_sub = dis[np.ix_(idx_o, idx_d)]  # [len_o, len_d]
#     y = od[np.ix_(idx_o, idx_d)].reshape(-1)

#     origin_idx = np.repeat(idx_o, len(idx_d))  # shape: [len_o * len_d]
#     dest_idx   = np.tile(idx_d, len(idx_o))    # shape: [len_o * len_d]
#     dis_feat   = dis_sub.reshape(-1, 1)        # shape: [len_o * len_d, 1]

#     x = np.stack([origin_idx, dest_idx], axis=1)
#     x = np.concatenate([x, dis_feat], axis=1)  # shape: [N, 3] => [origin_id, dest_id, dis]

#     return x, y


# def construct_validtest(city_path, target_areas, all_areas):
#     attr = np.load(f"{city_path}/attr.npy")
#     dis = np.load(f"{city_path}/dis.npy")
#     od = np.load(f"{city_path}/od.npy")

#     idx_o = [int(a) for a in target_areas]
#     idx_d = [int(a) for a in all_areas]

#     dis_sub = dis[np.ix_(idx_o, idx_d)]  # [len_o, len_d]
#     y1 = od[np.ix_(idx_o, idx_d)].reshape(-1)

#     origin_idx = np.repeat(idx_o, len(idx_d))         # shape: [len_o * len_d]
#     dest_idx = np.tile(idx_d, len(idx_o))             # shape: [len_o * len_d]
#     dis_feature = dis_sub.reshape(-1, 1)              # shape: [len_o * len_d, 1]

#     # 正向方向
#     x1 = np.stack([origin_idx, dest_idx], axis=1)     # shape: [N, 2]
#     x1 = np.concatenate([x1, dis_feature], axis=1)    # shape: [N, 3]

#     # 反方向
#     dis_sub_rev = dis[np.ix_(idx_d, idx_o)]
#     y2 = od[np.ix_(idx_d, idx_o)].reshape(-1)
#     origin_idx_rev = np.repeat(idx_d, len(idx_o))
#     dest_idx_rev = np.tile(idx_o, len(idx_d))
#     dis_feature_rev = dis_sub_rev.reshape(-1, 1)
#     x2 = np.stack([origin_idx_rev, dest_idx_rev], axis=1)
#     x2 = np.concatenate([x2, dis_feature_rev], axis=1)

#     x = np.concatenate([x1, x2], axis=0)     # shape: [2N, 3]
#     y = np.concatenate([y1, y2], axis=0)

#     return x, y

def get_node_neighbors_and_features(city_path, k=30):
    """
    获取每个节点的最近 k 个邻居和节点属性特征。

    返回：
        neighbors_index: [num_nodes, k]，每个节点的最近邻索引（不足k补-1）
        neighbors_mask:  [num_nodes, k]，每个位置是否在阈值内（1表示是邻居，0表示非邻居）
        node_attr:       [num_nodes, feat_dim]，节点属性特征
    """
    attr = np.load(f"{city_path}/attr.npy")   # [num_nodes, feat_dim]
    dis = np.load(f"{city_path}/dis.npy")     # [num_nodes, num_nodes]
    num_nodes = dis.shape[0]

    # 排除自身（对角线赋极大值）
    dis_no_self = dis + np.eye(num_nodes) * np.max(dis) * 10

    # 获取每个节点的最近 k 个邻居索引
    sorted_idx = np.argsort(dis_no_self, axis=1)  # 全排序后的索引 [num_nodes, num_nodes]
    sorted_dis = np.take_along_axis(dis, sorted_idx, axis=1)  # 排序后的实际距离

    neighbors_index = sorted_idx[:, :k]         # 最近的 k 个邻居索引
    neighbors_distance = sorted_dis[:, :k]      # 对应的距离
    # neighbors_mask = (neighbors_distance < threshold).astype(np.int32)  # 小于阈值记为 1，否则 0

    return neighbors_index, attr



def construct_train_new(city_path, train_areas):
    """
    训练样本：仅包含 train_areas 之间的 OD（origin in train, dest in train, origin != dest）
    返回 samples 列表，每项为 dict:
      {
        'origin': int,
        'destinations': [int,...],
        'distances': np.array([...]),
        'total_flow': float,
        'prob': np.array([...]),  # 原始流量向量
        'nonzero_mask': [idx,...] # 在 destinations 中流量>0 的位置索引
      }
    """
    od = np.load(f"{city_path}/od.npy")
    dis = np.load(f"{city_path}/dis.npy")

    train_list = sorted(set(map(int, train_areas)))
    samples = []
    for origin in train_list:
        # 目的地仅为训练区内其它区域 (排除自身)
        destinations = [d for d in train_list if d != origin]
        if len(destinations) == 0:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append({
            'origin': origin,
            'destinations': destinations,
            'distances': distances,
            'total_flow': float(np.sum(flows)),
            'prob': flows,
            'nonzero_mask': nonzero_mask
        })

    return samples


def construct_eval_samples(city_path, split_areas, train_areas):
    """
    构建验证/测试样本的通用函数（按你要求的结构）：
      - 正向 (forward)： origins in split_areas, destinations in (train_areas + split_areas)，去除 origin==dest
      - 反向 (reverse)： origins in train_areas, destinations in split_areas，去除 origin==dest（一般 origin 不会等于 dest）
    注意：
      - 参数 train_areas 可以是单独的 train 列表（用于 val），也可以传入 train+valid（用于 test），以实现你要的覆盖范围。
    返回 samples 列表（forward + reverse 合并），每项同 construct_train_new 的格式。
    """
    od = np.load(f"{city_path}/od.npy")
    dis = np.load(f"{city_path}/dis.npy")

    split_list = sorted(set(map(int, split_areas)))
    train_list = sorted(set(map(int, train_areas)))

    # forward destinations = train_list + split_list （保持顺序并去重）
    forward_dests = list(dict.fromkeys(train_list + split_list))

    samples = []
    # 正向： split -> (train + split)
    for origin in split_list:
        destinations = [d for d in forward_dests if d != origin]
        if len(destinations) == 0:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append({
            'origin': origin,
            'destinations': destinations,
            'distances': distances,
            'total_flow': float(np.sum(flows)),
            'prob': flows,
            'nonzero_mask': nonzero_mask
        })

    # 反向： train -> split
    for origin in train_list:
        destinations = [d for d in split_list if d != origin]
        if len(destinations) == 0:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append({
            'origin': origin,
            'destinations': destinations,
            'distances': distances,
            'total_flow': float(np.sum(flows)),
            'prob': flows,
            'nonzero_mask': nonzero_mask
        })

    return samples

def count_total_pairs(samples):
    """
    输入: samples (list of dict)，每个 dict 里有 'destinations'
    输出: 样本总数（OD对数）
    """
    total = 0
    for s in samples:
        total += len(s['destinations'])
    return total

import numpy as np

def compute_train_features(city_path, train_areas, threshold=10):
    """
    计算训练区域的属性向量和 OD 特征向量（仅在训练区域内）
    参数:
        city_path: 数据路径
        train_areas: 训练区域列表（字符串或整数）
        threshold: OD值阈值，小于threshold置0，其余取log
    返回:
        train_attr: [num_train_nodes, feat_dim] 属性矩阵
        od_features: [num_train_nodes, num_train_nodes] 仅训练区域内的 OD 特征矩阵
    """
    # 加载原始数据
    attr = np.load(f"{city_path}/attr.npy")   # [num_nodes, feat_dim]
    od   = np.load(f"{city_path}/od.npy")     # [num_nodes, num_nodes]

    # 训练区域索引（转成整型 np.array）
    train_idx = np.asarray([int(a) for a in train_areas], dtype=int)

    # 训练区域属性
    train_attr = attr[train_idx]              # [num_train_nodes, feat_dim]

    # 仅选取训练区域内的 OD 子矩阵（行: 起点/origin；列: 终点/destination）
    od_sub = od[np.ix_( train_idx, train_idx)].astype(np.float32)  # [num_train_nodes, num_train_nodes]

    # 阈值 + log 处理：小于阈值置0，其余取log
    with np.errstate(divide='ignore'):
        od_features = np.where(od_sub >= threshold, np.log(od_sub), 0.0).astype(np.float32)
        # od_features = np.where(od_sub >= threshold, od_sub, 0.0).astype(np.float32)


    return train_attr, od_features

def load_data(city_path, neighbors_k, if_shuffle=True):
    """
    新的 load_data：
      - 把区域划分为 train/valid/test（70/10/20）
      - train_samples: train -> train
      - val_samples:  val -> (train+val)  +  train -> val
      - test_samples: test -> (train+valid+test)  +  (train+valid) -> test

    返回：
      train_samples, val_samples, test_samples,
      neighbors_index, attr,
      train_od_features, train_attr,
      train_idx, valid_idx, test_idx
    """
    # 所有区域（字符串形式）
    od = np.load(f"{city_path}/od.npy")
    areas = [str(i) for i in range(od.shape[0])]
    if if_shuffle:
        random.shuffle(areas)

    # 7:1:2 划分
    train_areas, valid_areas, test_areas = split_areas(areas)

    # 训练样本：仅训练区内部对
    train_samples = construct_train_new(city_path, train_areas)

    # 验证样本：val -> (train + val) 以及 train -> val
    val_samples = construct_eval_samples(city_path, valid_areas, train_areas)

    # 测试样本：test -> (train + valid + test) 以及 (train + valid) -> test
    # 通过把 train_areas 参数传成 train + valid 来实现由 (train+valid) 发向 test 的反向样本
    train_plus_valid = list(dict.fromkeys(list(map(int, train_areas)) + list(map(int, valid_areas))))
    train_plus_valid = [str(x) for x in train_plus_valid]
    test_samples = construct_eval_samples(city_path, test_areas, train_plus_valid)

    # 邻居与原始属性
    neighbors_index, attr = get_node_neighbors_and_features(city_path, neighbors_k)

    # 训练区域的 OD 交互特征
    train_attr, train_od_features = compute_train_features(city_path, train_areas)

    print(f"Split sizes -> Train areas: {len(train_areas)}, Valid areas: {len(valid_areas)}, Test areas: {len(test_areas)}")
    print(f"Samples -> Train: {len(train_samples)}, Valid: {len(val_samples)}, Test: {len(test_samples)}")
    print(f"Neighbors index: {neighbors_index.shape}, attr: {attr.shape}")

    print(f"Train origins: {len(train_samples)}, Train OD pairs: {count_total_pairs(train_samples)}")
    print(f"Valid origins: {len(val_samples)}, Valid OD pairs: {count_total_pairs(val_samples)}")
    print(f"Test origins: {len(test_samples)}, Test OD pairs: {count_total_pairs(test_samples)}")

    # 把三个 split 的区域编号都转成 int 列表返回，方便后面用来索引
    train_idx = [int(a) for a in train_areas]
    valid_idx = [int(a) for a in valid_areas]
    test_idx  = [int(a) for a in test_areas]

    return (
        train_samples, val_samples, test_samples,
        neighbors_index, attr,
        train_od_features, train_attr,
        train_idx, valid_idx, test_idx
    )
