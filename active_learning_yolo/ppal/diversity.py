"""在预计算距离矩阵上进行多样性采样。"""

from __future__ import annotations

import numpy as np


def _validate(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix 必须是方阵")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance_matrix 不能包含 NaN 或无穷值")
    if np.any(matrix < -1e-8):
        raise ValueError("距离不能为负数")
    return matrix


def farthest_first_init(
    distance_matrix: np.ndarray, n_clusters: int, seed: int = 0
) -> np.ndarray:
    """最远点初始化，seed 用于让结果可复现。"""

    matrix = _validate(distance_matrix)
    size = matrix.shape[0]
    if not 1 <= n_clusters <= size:
        raise ValueError("n_clusters 必须位于 [1, 样本数]")
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(0, size))]
    while len(medoids) < n_clusters:
        nearest = matrix[:, medoids].min(axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    return np.asarray(medoids, dtype=np.int64)


def kmedoids(
    distance_matrix: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    seed: int = 0,
) -> list[int]:
    """k-medoids：每个簇选择真实样本作为代表。"""

    matrix = _validate(distance_matrix)
    medoids = farthest_first_init(matrix, n_clusters, seed)
    for _ in range(max_iter):
        assignment = np.argmin(matrix[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for cluster_id in range(n_clusters):
            members = np.flatnonzero(assignment == cluster_id)
            if len(members) == 0:
                nearest = matrix[:, new_medoids].min(axis=1)
                nearest[new_medoids] = -1.0
                new_medoids[cluster_id] = int(np.argmax(nearest))
                continue
            cluster_matrix = matrix[np.ix_(members, members)]
            new_medoids[cluster_id] = int(
                members[np.argmin(cluster_matrix.sum(axis=1))]
            )
        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            medoids = new_medoids
            break
        medoids = new_medoids
    return medoids.tolist()
