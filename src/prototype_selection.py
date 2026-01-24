"""
Prototype Selection Algorithms

Implements:
1. Variance-Weighted Centroid Prototype Selection
2. KNN-Overlap Boundary Selection (new)
3. Random Selection (baseline)
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


def select_prototypes_cluster_boundary(X_train, y_train, M, random_state=None):
    """
    Cluster-based Boundary Prototype Selection.

    Algorithm:
    1. For each class, do K-Means to get cluster centroids
    2. Collect all centroids from all classes
    3. For each centroid, compute "boundary score" = inverse distance to nearest
       centroid from a DIFFERENT class (closer = higher score = more boundary-like)
    4. Rank clusters by boundary score
    5. Select points from high-scoring (boundary) clusters first

    This is efficient because we only compare ~M centroids, not N points.

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
    n_classes = len(classes)

    # Step 1: Compute variance for allocation (same as before)
    variances = {}
    for c in classes:
        mask = y_train == c
        X_c = X_train[mask]
        variances[c] = np.var(X_c, axis=0).sum()

    total_variance = sum(variances.values())

    # Allocate prototypes proportionally
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

    # Step 2: For each class, do K-Means and store centroids with metadata
    all_clusters = []  # List of (class, centroid, cluster_points_indices)

    for c in classes:
        mask = y_train == c
        indices_c = np.where(mask)[0]
        X_c = X_train[mask]
        n_c = allocations[c]

        if n_c >= len(indices_c):
            # If we need all points, treat each point as its own cluster
            for i, idx in enumerate(indices_c):
                all_clusters.append({
                    'class': c,
                    'centroid': X_c[i],
                    'points': [idx],
                    'n_select': 1
                })
            continue

        # K-Means clustering
        kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=random_state,
                                  n_init=3, batch_size=256, max_iter=100)
        cluster_labels = kmeans.fit_predict(X_c)
        centroids = kmeans.cluster_centers_

        for k in range(n_c):
            cluster_mask = cluster_labels == k
            cluster_points = indices_c[cluster_mask].tolist()
            if len(cluster_points) > 0:
                all_clusters.append({
                    'class': c,
                    'centroid': centroids[k],
                    'points': cluster_points,
                    'n_select': 1  # Select 1 point per cluster
                })

    # Step 3: Compute boundary score for each cluster
    # Boundary score = 1 / (distance to nearest centroid of different class)
    n_clusters = len(all_clusters)
    centroids_matrix = np.array([cl['centroid'] for cl in all_clusters])
    cluster_classes = np.array([cl['class'] for cl in all_clusters])

    for i, cluster in enumerate(all_clusters):
        c = cluster['class']
        centroid = cluster['centroid']

        # Find centroids from different classes
        diff_mask = cluster_classes != c
        if not diff_mask.any():
            cluster['boundary_score'] = 0
            continue

        diff_centroids = centroids_matrix[diff_mask]

        # Distance to nearest different-class centroid
        distances = np.sqrt(np.sum((diff_centroids - centroid) ** 2, axis=1))
        min_dist = np.min(distances)

        # Boundary score: closer to other class = higher score
        cluster['boundary_score'] = 1.0 / (min_dist + 1e-10)

    # Step 4: Sort clusters by boundary score (highest first)
    all_clusters.sort(key=lambda x: x['boundary_score'], reverse=True)

    # Step 5: Select points from clusters, prioritizing boundary clusters
    selected_indices = []
    selected_set = set()

    for cluster in all_clusters:
        points = cluster['points']
        centroid = cluster['centroid']

        # Find the point closest to centroid that hasn't been selected
        X_points = X_train[points]
        distances = np.sum((X_points - centroid) ** 2, axis=1)
        sorted_local = np.argsort(distances)

        for local_idx in sorted_local:
            global_idx = points[local_idx]
            if global_idx not in selected_set:
                selected_indices.append(global_idx)
                selected_set.add(global_idx)
                break

        if len(selected_indices) >= M:
            break

    # If we still need more points (due to empty clusters), fill from boundary clusters
    if len(selected_indices) < M:
        for cluster in all_clusters:
            points = cluster['points']
            for global_idx in points:
                if global_idx not in selected_set:
                    selected_indices.append(global_idx)
                    selected_set.add(global_idx)
                    if len(selected_indices) >= M:
                        break
            if len(selected_indices) >= M:
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

    # Test variance-centroid method
    print("Testing variance-centroid selection...")
    t0 = time.time()
    indices = select_prototypes_variance_centroid(X, y, M=500, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices], minlength=10)}")

    # Test cluster-boundary method
    print("\nTesting cluster-boundary selection...")
    t0 = time.time()
    indices_boundary = select_prototypes_cluster_boundary(X, y, M=500, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_boundary)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_boundary], minlength=10)}")

    # Test random baseline
    print("\nTesting random selection...")
    t0 = time.time()
    indices_random = select_prototypes_random(X, y, M=500, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_random)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_random], minlength=10)}")
