"""
Prototype Selection Algorithms

Implements:
1. Variance-Weighted Centroid Prototype Selection (our method)
2. Random Selection (baseline)
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def select_prototypes_variance_centroid(X_train, y_train, M, random_state=None):
    """
    Variance-Weighted Centroid Prototype Selection.

    Algorithm:
    1. Compute variance for each class
    2. Allocate M prototypes proportionally based on variance
    3. For each class:
       a. K-Means clustering to find cluster centers
       b. Select point closest to each centroid

    Args:
        X_train: (N, D) training features
        y_train: (N,) training labels
        M: Number of prototypes to select
        random_state: Random seed for reproducibility

    Returns:
        indices: (M,) indices of selected prototypes
    """
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y_train)

    # Step 1: Compute variance for each class
    variances = {}
    for c in classes:
        mask = y_train == c
        X_c = X_train[mask]
        variances[c] = np.var(X_c, axis=0).sum()

    total_variance = sum(variances.values())

    # Step 2: Allocate prototypes proportionally
    allocations = {}
    allocated = 0
    for c in classes:
        ratio = variances[c] / total_variance
        allocations[c] = max(1, int(round(M * ratio)))
        allocated += allocations[c]

    # Adjust to exactly M
    diff = M - allocated
    if diff != 0:
        sorted_classes = sorted(classes, key=lambda c: variances[c], reverse=True)
        for c in sorted_classes:
            if diff > 0:
                allocations[c] += 1
                diff -= 1
            elif diff < 0 and allocations[c] > 1:
                allocations[c] -= 1
                diff += 1
            if diff == 0:
                break

    # Step 3: Select prototypes for each class
    selected_indices = []

    for c in classes:
        mask = y_train == c
        indices_c = np.where(mask)[0]
        X_c = X_train[mask]
        n_c = allocations[c]

        if n_c >= len(indices_c):
            selected_indices.extend(indices_c.tolist())
            continue

        # Fast K-Means clustering
        kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=random_state,
                                  n_init=3, batch_size=256, max_iter=100)
        kmeans.fit(X_c)
        centroids = kmeans.cluster_centers_

        # For each centroid, find closest real point
        for centroid in centroids:
            distances = np.sum((X_c - centroid) ** 2, axis=1)
            closest_local_idx = np.argmin(distances)
            closest_global_idx = indices_c[closest_local_idx]

            if closest_global_idx not in selected_indices:
                selected_indices.append(closest_global_idx)
            else:
                # If already selected, pick next closest
                sorted_idx = np.argsort(distances)
                for idx in sorted_idx:
                    if indices_c[idx] not in selected_indices:
                        selected_indices.append(indices_c[idx])
                        break

    return np.array(selected_indices[:M], dtype=np.int32)


def select_prototypes_random(X_train, y_train, M, random_state=None):
    """
    Random prototype selection (baseline).

    Randomly selects M points from training set, ensuring at least one per class.
    """
    rng = np.random.RandomState(random_state)

    classes = np.unique(y_train)
    n_classes = len(classes)

    # Ensure at least one per class
    selected = []
    for c in classes:
        indices_c = np.where(y_train == c)[0]
        idx = rng.choice(indices_c)
        selected.append(idx)

    # Fill remaining slots randomly
    remaining = M - n_classes
    if remaining > 0:
        all_indices = np.arange(len(y_train))
        available = np.setdiff1d(all_indices, selected)
        additional = rng.choice(available, size=remaining, replace=False)
        selected.extend(additional.tolist())

    return np.array(selected, dtype=np.int32)


if __name__ == '__main__':
    import time
    np.random.seed(42)

    # Create dummy data
    N = 6000
    D = 784
    X = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, 10, N)

    # Test our method
    print("Testing variance-centroid selection...")
    t0 = time.time()
    indices = select_prototypes_variance_centroid(X, y, M=500, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices], minlength=10)}")

    # Test random baseline
    print("\nTesting random selection...")
    t0 = time.time()
    indices_random = select_prototypes_random(X, y, M=500, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_random)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_random], minlength=10)}")
