"""
Prototype Selection Algorithms

Implements:
1. Variance-Weighted Centroid Prototype Selection
2. Cluster-based Boundary Selection
3. Boundary-First Selection (new) - K-Means on boundary points only
4. Condensed Nearest Neighbor (CNN) (new) - iteratively add misclassified points
5. Random Selection (baseline)
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


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


def select_prototypes_boundary_first(X_train, y_train, M, random_state=None, k=5):
    """
    Boundary-First Prototype Selection.

    Key insight: Centroids are cluster centers, not boundary representatives.
    This algorithm first identifies boundary points (points with mixed-class neighbors),
    then does K-Means ONLY on boundary points to select prototypes.

    Algorithm:
    1. For each point, check if its k-NN contains points from other classes
    2. Mark those as "boundary points"
    3. Allocate prototypes per class based on variance
    4. For each class, do K-Means on its boundary points
    5. Select points closest to each centroid

    Args:
        X_train: (N, D) training features
        y_train: (N,) training labels
        M: Number of prototypes to select
        random_state: Random seed
        k: Number of neighbors to check for boundary detection

    Returns:
        indices: (M,) indices of selected prototypes
    """
    if random_state is not None:
        np.random.seed(random_state)

    N = len(X_train)
    classes = np.unique(y_train)

    # Step 1: Identify boundary points using k-NN
    # A point is a boundary point if any of its k neighbors has a different label
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1)
    nn.fit(X_train)
    _, indices_knn = nn.kneighbors(X_train)

    # indices_knn[:, 0] is the point itself, so we use [:, 1:]
    neighbor_labels = y_train[indices_knn[:, 1:]]  # (N, k)
    own_labels = y_train.reshape(-1, 1)  # (N, 1)

    # Check if any neighbor has a different label
    has_diff_neighbor = np.any(neighbor_labels != own_labels, axis=1)  # (N,)
    boundary_mask = has_diff_neighbor

    n_boundary = np.sum(boundary_mask)
    print(f"  [boundary_first] Found {n_boundary} boundary points ({100*n_boundary/N:.1f}% of training set)")

    # Step 2: Compute variance for allocation
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

    # Step 3: For each class, do K-Means on boundary points
    selected_indices = []
    selected_set = set()

    for c in classes:
        class_mask = y_train == c
        n_c = allocations[c]

        # Get boundary points of this class
        boundary_class_mask = class_mask & boundary_mask
        boundary_indices = np.where(boundary_class_mask)[0]
        X_boundary = X_train[boundary_class_mask]

        # Fallback: if not enough boundary points, use all points of this class
        if len(boundary_indices) < n_c:
            print(f"  [boundary_first] Class {c}: only {len(boundary_indices)} boundary points, using all class points")
            boundary_indices = np.where(class_mask)[0]
            X_boundary = X_train[class_mask]

        if n_c >= len(boundary_indices):
            # Select all boundary points
            for idx in boundary_indices:
                if idx not in selected_set:
                    selected_indices.append(idx)
                    selected_set.add(idx)
            continue

        # K-Means on boundary points only
        kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=random_state,
                                  n_init=3, batch_size=256, max_iter=100)
        kmeans.fit(X_boundary)
        centroids = kmeans.cluster_centers_

        # Select point closest to each centroid
        for centroid in centroids:
            distances = np.sum((X_boundary - centroid) ** 2, axis=1)
            sorted_idx = np.argsort(distances)

            for local_idx in sorted_idx:
                global_idx = boundary_indices[local_idx]
                if global_idx not in selected_set:
                    selected_indices.append(global_idx)
                    selected_set.add(global_idx)
                    break

    return np.array(selected_indices[:M], dtype=np.int32)


def select_prototypes_cnn(X_train, y_train, M, random_state=None, max_iter=10):
    """
    Condensed Nearest Neighbor (CNN) inspired Prototype Selection.

    Classic CNN algorithm builds a prototype set by iteratively adding
    misclassified points. This version adapts CNN to select exactly M prototypes.

    Algorithm:
    1. Initialize: randomly select 1 point per class
    2. Iterate:
       a. Use current prototype set S for 1-NN classification
       b. Find all misclassified training points
       c. Add the most "important" misclassified points to S
       d. Stop when |S| = M

    "Important" = misclassified points that are close to the decision boundary
    (i.e., distance to nearest prototype is small)

    Args:
        X_train: (N, D) training features
        y_train: (N,) training labels
        M: Number of prototypes to select
        random_state: Random seed
        max_iter: Maximum iterations

    Returns:
        indices: (M,) indices of selected prototypes
    """
    rng = np.random.RandomState(random_state)

    N = len(X_train)
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Step 1: Initialize with 1 random point per class
    selected_indices = []
    selected_set = set()

    for c in classes:
        indices_c = np.where(y_train == c)[0]
        idx = rng.choice(indices_c)
        selected_indices.append(idx)
        selected_set.add(idx)

    print(f"  [CNN] Starting with {len(selected_indices)} prototypes (1 per class)")

    # Step 2: Iteratively add misclassified points
    remaining = M - len(selected_indices)

    for iteration in range(max_iter):
        if remaining <= 0:
            break

        # Build 1-NN classifier with current prototypes
        X_proto = X_train[selected_indices]
        y_proto = y_train[selected_indices]

        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
        nn.fit(X_proto)

        # Classify all training points
        distances, neighbor_idx = nn.kneighbors(X_train)
        distances = distances.ravel()
        neighbor_idx = neighbor_idx.ravel()
        predictions = y_proto[neighbor_idx]

        # Find misclassified points (not already in prototype set)
        misclassified_mask = (predictions != y_train)
        for idx in selected_set:
            misclassified_mask[idx] = False

        misclassified_indices = np.where(misclassified_mask)[0]
        n_misclassified = len(misclassified_indices)

        if n_misclassified == 0:
            print(f"  [CNN] Iteration {iteration+1}: No misclassified points, stopping early")
            break

        # Sort misclassified points by distance to nearest prototype (ascending)
        # Points closer to the boundary are more important
        misclassified_distances = distances[misclassified_indices]
        sorted_order = np.argsort(misclassified_distances)

        # Add points, but ensure class balance
        # Count how many we have per class
        current_counts = {c: 0 for c in classes}
        for idx in selected_indices:
            current_counts[y_train[idx]] += 1

        # Target: roughly equal distribution, but prioritize underrepresented classes
        target_per_class = M // n_classes

        n_to_add = min(remaining, max(1, remaining // (max_iter - iteration)))
        added_this_iter = 0

        for local_idx in sorted_order:
            if added_this_iter >= n_to_add:
                break

            global_idx = misclassified_indices[local_idx]
            c = y_train[global_idx]

            # Prefer adding points from underrepresented classes
            if current_counts[c] < target_per_class * 1.5 or remaining <= n_classes:
                selected_indices.append(global_idx)
                selected_set.add(global_idx)
                current_counts[c] += 1
                added_this_iter += 1
                remaining -= 1

        print(f"  [CNN] Iteration {iteration+1}: {n_misclassified} misclassified, added {added_this_iter}, total {len(selected_indices)}")

    # Step 3: If still under M, fill with random points from each class
    if len(selected_indices) < M:
        print(f"  [CNN] Filling remaining {M - len(selected_indices)} slots randomly")
        all_indices = np.arange(N)
        available = np.setdiff1d(all_indices, list(selected_set))

        # Prefer to fill from underrepresented classes
        current_counts = {c: 0 for c in classes}
        for idx in selected_indices:
            current_counts[y_train[idx]] += 1

        for c in sorted(classes, key=lambda c: current_counts[c]):
            if len(selected_indices) >= M:
                break
            available_c = [i for i in available if y_train[i] == c and i not in selected_set]
            n_need = max(0, M // n_classes - current_counts[c])
            n_add = min(n_need, len(available_c), M - len(selected_indices))
            if n_add > 0:
                added = rng.choice(available_c, size=n_add, replace=False)
                selected_indices.extend(added.tolist())
                selected_set.update(added.tolist())

        # Fill any remaining slots
        while len(selected_indices) < M:
            available = [i for i in range(N) if i not in selected_set]
            if not available:
                break
            idx = rng.choice(available)
            selected_indices.append(idx)
            selected_set.add(idx)

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

    M = 500

    # Test variance-centroid method
    print("=" * 50)
    print("Testing variance-centroid selection...")
    t0 = time.time()
    indices = select_prototypes_variance_centroid(X, y, M=M, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices], minlength=10)}")

    # Test cluster-boundary method
    print("\n" + "=" * 50)
    print("Testing cluster-boundary selection...")
    t0 = time.time()
    indices_boundary = select_prototypes_cluster_boundary(X, y, M=M, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_boundary)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_boundary], minlength=10)}")

    # Test boundary-first method (NEW)
    print("\n" + "=" * 50)
    print("Testing boundary-first selection...")
    t0 = time.time()
    indices_bf = select_prototypes_boundary_first(X, y, M=M, random_state=42, k=5)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_bf)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_bf], minlength=10)}")

    # Test CNN method (NEW)
    print("\n" + "=" * 50)
    print("Testing CNN selection...")
    t0 = time.time()
    indices_cnn = select_prototypes_cnn(X, y, M=M, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_cnn)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_cnn], minlength=10)}")

    # Test random baseline
    print("\n" + "=" * 50)
    print("Testing random selection...")
    t0 = time.time()
    indices_random = select_prototypes_random(X, y, M=M, random_state=42)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Selected {len(indices_random)} prototypes")
    print(f"Label distribution: {np.bincount(y[indices_random], minlength=10)}")
