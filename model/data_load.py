import random
import numpy as np


def split_areas(areas, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    assert train_ratio + valid_ratio + test_ratio == 1
    random.shuffle(areas)
    n = len(areas)
    train_areas = areas[: int(n * train_ratio)]
    valid_areas = areas[int(n * train_ratio) : int(n * (train_ratio + valid_ratio))]
    test_areas = areas[int(n * (train_ratio + valid_ratio)) :]
    return train_areas, valid_areas, test_areas


def get_node_neighbors_and_features(city_path, k=30):
    attr = np.load(f"{city_path}/attr.npy")
    dis = np.load(f"{city_path}/dis.npy")
    num_nodes = dis.shape[0]

    dis_no_self = dis + np.eye(num_nodes) * np.max(dis) * 10
    sorted_idx = np.argsort(dis_no_self, axis=1)
    neighbors_index = sorted_idx[:, :k]

    return neighbors_index, attr


def construct_train_new(city_path, train_areas):
    od = np.load(f"{city_path}/od.npy")
    dis = np.load(f"{city_path}/dis.npy")

    train_list = sorted(set(map(int, train_areas)))
    samples = []
    for origin in train_list:
        destinations = [d for d in train_list if d != origin]
        if not destinations:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append(
            {
                "origin": origin,
                "destinations": destinations,
                "distances": distances,
                "total_flow": float(np.sum(flows)),
                "prob": flows,
                "nonzero_mask": nonzero_mask,
            }
        )
    return samples


def construct_eval_samples(city_path, split_areas, train_areas):
    od = np.load(f"{city_path}/od.npy")
    dis = np.load(f"{city_path}/dis.npy")

    split_list = sorted(set(map(int, split_areas)))
    train_list = sorted(set(map(int, train_areas)))

    forward_dests = list(dict.fromkeys(train_list + split_list))

    samples = []
    for origin in split_list:
        destinations = [d for d in forward_dests if d != origin]
        if not destinations:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append(
            {
                "origin": origin,
                "destinations": destinations,
                "distances": distances,
                "total_flow": float(np.sum(flows)),
                "prob": flows,
                "nonzero_mask": nonzero_mask,
            }
        )

    for origin in train_list:
        destinations = [d for d in split_list if d != origin]
        if not destinations:
            continue
        flows = od[origin, destinations]
        distances = dis[origin, destinations]
        nonzero_mask = [i for i, f in enumerate(flows) if f > 0]

        samples.append(
            {
                "origin": origin,
                "destinations": destinations,
                "distances": distances,
                "total_flow": float(np.sum(flows)),
                "prob": flows,
                "nonzero_mask": nonzero_mask,
            }
        )

    return samples


def count_total_pairs(samples):
    total = 0
    for s in samples:
        total += len(s["destinations"])
    return total


def compute_train_features(city_path, train_areas, threshold=10):
    attr = np.load(f"{city_path}/attr.npy")
    od = np.load(f"{city_path}/od.npy")

    train_idx = np.asarray([int(a) for a in train_areas], dtype=int)
    train_attr = attr[train_idx]
    od_sub = od[np.ix_(train_idx, train_idx)].astype(np.float32)

    with np.errstate(divide="ignore"):
        od_features = np.where(od_sub >= threshold, np.log(od_sub), 0.0).astype(np.float32)

    return train_attr, od_features


def load_data(city_path, neighbors_k, if_shuffle=True):
    od = np.load(f"{city_path}/od.npy")
    areas = [str(i) for i in range(od.shape[0])]
    if if_shuffle:
        random.shuffle(areas)

    train_areas, valid_areas, test_areas = split_areas(areas)

    train_samples = construct_train_new(city_path, train_areas)
    val_samples = construct_eval_samples(city_path, valid_areas, train_areas)

    train_plus_valid = list(dict.fromkeys(list(map(int, train_areas)) + list(map(int, valid_areas))))
    train_plus_valid = [str(x) for x in train_plus_valid]
    test_samples = construct_eval_samples(city_path, test_areas, train_plus_valid)

    neighbors_index, attr = get_node_neighbors_and_features(city_path, neighbors_k)
    train_attr, train_od_features = compute_train_features(city_path, train_areas)


    train_idx = [int(a) for a in train_areas]
    valid_idx = [int(a) for a in valid_areas]
    test_idx = [int(a) for a in test_areas]

    return (
        train_samples,
        val_samples,
        test_samples,
        neighbors_index,
        attr,
        train_od_features,
        train_attr,
        train_idx,
        valid_idx,
        test_idx,
    )