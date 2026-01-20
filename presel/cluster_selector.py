"""
Cluster-Based Selection Module

Implements the task-wise cluster-based selection described in Section 3.2:
1. Extract visual features using DINOv2
2. Cluster images within each task using k-means
3. Select representative images using Neighbor Centrality (NC) score
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class ClusterBasedSelector:
    """
    Select representative images using clustering and Neighbor Centrality.

    This implements the selection strategy from Section 3.2 of the paper.
    """

    def __init__(
        self,
        feature_extractor=None,
        num_neighbors: int = 5,
        num_clusters_per_100: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            feature_extractor: Vision encoder (e.g., DINOv2) for feature extraction
            num_neighbors: k in k-nearest neighbors for NC score (default: 5)
            num_clusters_per_100: Number of clusters per 100 images (default: 1)
            device: Device to run computations on
        """
        self.feature_extractor = feature_extractor
        self.num_neighbors = num_neighbors
        self.num_clusters_per_100 = num_clusters_per_100
        self.device = device

        if self.feature_extractor is not None and hasattr(self.feature_extractor, 'to'):
            self.feature_extractor = self.feature_extractor.to(device)
            self.feature_extractor.eval()

    def extract_features(
        self,
        images: List[torch.Tensor],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract visual features for a list of images.

        Args:
            images: List of image tensors
            batch_size: Batch size for feature extraction
            show_progress: Whether to show progress bar

        Returns:
            Feature matrix of shape (num_images, feature_dim)
        """
        if self.feature_extractor is None:
            # Use dummy features for testing
            return self._extract_dummy_features(images)

        features_list = []

        num_batches = (len(images) + batch_size - 1) // batch_size
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")

        with torch.no_grad():
            for i in iterator:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images))
                batch_images = images[start_idx:end_idx]

                # Stack images into batch
                batch_tensor = torch.stack([
                    img.to(self.device) if isinstance(img, torch.Tensor)
                    else torch.tensor(img).to(self.device)
                    for img in batch_images
                ])

                # Extract features
                batch_features = self._forward_feature_extractor(batch_tensor)
                features_list.append(batch_features.cpu().numpy())

        # Concatenate all features
        features = np.vstack(features_list)
        return features

    def _forward_feature_extractor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Feature vectors (B, D)
        """
        try:
            # For DINOv2 and similar models
            if hasattr(self.feature_extractor, 'forward_features'):
                features = self.feature_extractor.forward_features(images)
                # Get [CLS] token
                if isinstance(features, dict):
                    features = features.get('x_norm_clstoken', features.get('cls_token'))
                elif features.dim() == 3:  # (B, N, D)
                    features = features[:, 0]  # Take [CLS] token
            else:
                # Generic forward
                features = self.feature_extractor(images)
                if isinstance(features, dict):
                    features = features.get('pooler_output', features.get('last_hidden_state'))
                if features.dim() == 3:
                    features = features[:, 0]

            return features
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            # Return dummy features
            return torch.randn(images.size(0), 768).to(self.device)

    def _extract_dummy_features(self, images: List[torch.Tensor]) -> np.ndarray:
        """Extract dummy features for testing without a real model."""
        num_images = len(images)
        feature_dim = 768  # Standard dimension

        # Create simple features based on image statistics
        features = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # Use image statistics as features
                feat = torch.tensor([
                    img.mean().item(),
                    img.std().item(),
                    img.min().item(),
                    img.max().item(),
                ])
                # Pad to feature_dim
                feat = F.pad(feat, (0, feature_dim - len(feat)))
            else:
                feat = torch.randn(feature_dim)

            features.append(feat.numpy())

        return np.array(features)

    def cluster_images(
        self,
        features: np.ndarray,
        num_clusters: int = None
    ) -> Tuple[np.ndarray, KMeans]:
        """
        Cluster images using k-means.

        Args:
            features: Feature matrix (num_images, feature_dim)
            num_clusters: Number of clusters (default: |T_i| / 100)

        Returns:
            Tuple of (cluster_labels, kmeans_model)
        """
        num_images = features.shape[0]

        # Set number of clusters as |T_i| / 100 if not provided
        if num_clusters is None:
            num_clusters = max(1, num_images // 100)

        # Ensure num_clusters doesn't exceed num_images
        num_clusters = min(num_clusters, num_images)

        print(f"Clustering {num_images} images into {num_clusters} clusters...")

        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(features)

        return cluster_labels, kmeans

    def compute_neighbor_centrality(
        self,
        features: np.ndarray,
        k: int = None
    ) -> np.ndarray:
        """
        Compute Neighbor Centrality (NC) score for each image (Equation 7).

        NC(I) = (1/k) * Σ sim(v_I, v_Ia) for Ia in kNN(I)

        Args:
            features: Feature matrix (num_images, feature_dim)
            k: Number of nearest neighbors (default: self.num_neighbors)

        Returns:
            NC scores for each image (num_images,)
        """
        if k is None:
            k = self.num_neighbors

        num_images = features.shape[0]
        k = min(k, num_images - 1)  # Ensure k < num_images

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(features)

        # For each image, find k nearest neighbors and compute average similarity
        nc_scores = np.zeros(num_images)

        for i in range(num_images):
            # Get similarities for image i (excluding itself)
            sim_i = similarities[i].copy()
            sim_i[i] = -np.inf  # Exclude self

            # Get indices of k nearest neighbors
            knn_indices = np.argpartition(sim_i, -k)[-k:]

            # Compute average similarity to k nearest neighbors
            nc_scores[i] = np.mean(sim_i[knn_indices])

        return nc_scores

    def select_from_cluster(
        self,
        cluster_features: np.ndarray,
        cluster_indices: np.ndarray,
        budget: int
    ) -> np.ndarray:
        """
        Select representative images from a cluster based on NC scores.

        Args:
            cluster_features: Features of images in this cluster
            cluster_indices: Original indices of images in this cluster
            budget: Number of images to select from this cluster

        Returns:
            Indices of selected images (from original image list)
        """
        num_images = len(cluster_indices)
        budget = min(budget, num_images)

        if budget <= 0:
            return np.array([], dtype=int)

        # Compute NC scores for images in this cluster
        nc_scores = self.compute_neighbor_centrality(cluster_features)

        # Select top-k images with highest NC scores
        top_k_indices = np.argsort(nc_scores)[-budget:]

        # Map back to original indices
        selected_indices = cluster_indices[top_k_indices]

        return selected_indices

    def select_images(
        self,
        images: List[torch.Tensor],
        budget: int,
        task_name: str = "task",
        show_progress: bool = True
    ) -> List[int]:
        """
        Select representative images from a task.

        Args:
            images: List of image tensors
            budget: Number of images to select
            task_name: Name of the task (for logging)
            show_progress: Whether to show progress bar

        Returns:
            List of indices of selected images
        """
        num_images = len(images)
        budget = min(budget, num_images)

        if budget <= 0:
            return []

        print(f"\nSelecting {budget} images from {num_images} for task '{task_name}'")

        # Step 1: Extract features
        features = self.extract_features(images, show_progress=show_progress)

        # Step 2: Cluster images
        num_clusters = max(1, num_images // 100)
        cluster_labels, _ = self.cluster_images(features, num_clusters)

        # Step 3: Select images from each cluster
        selected_indices = []

        for cluster_id in range(num_clusters):
            # Get images in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_features = features[cluster_mask]

            # Compute budget for this cluster (Equation 6)
            cluster_size = len(cluster_indices)
            cluster_budget = int(budget * cluster_size / num_images)

            # Select from this cluster
            selected = self.select_from_cluster(
                cluster_features,
                cluster_indices,
                cluster_budget
            )
            selected_indices.extend(selected.tolist())

        # Adjust selection to match budget exactly
        selected_indices = self._adjust_selection(
            selected_indices,
            budget,
            features
        )

        print(f"Selected {len(selected_indices)} images for task '{task_name}'")

        return selected_indices

    def _adjust_selection(
        self,
        selected_indices: List[int],
        target_budget: int,
        features: np.ndarray
    ) -> List[int]:
        """
        Adjust selection to match target budget exactly.

        Args:
            selected_indices: Current selection
            target_budget: Target number of images
            features: Feature matrix

        Returns:
            Adjusted selection
        """
        current_size = len(selected_indices)
        difference = target_budget - current_size

        if difference == 0:
            return selected_indices

        if difference > 0:
            # Need to add more images
            all_indices = set(range(features.shape[0]))
            remaining_indices = list(all_indices - set(selected_indices))

            if len(remaining_indices) > 0:
                # Compute NC scores for remaining images
                remaining_features = features[remaining_indices]
                nc_scores = self.compute_neighbor_centrality(remaining_features)

                # Select top images
                num_to_add = min(difference, len(remaining_indices))
                top_indices = np.argsort(nc_scores)[-num_to_add:]
                selected_indices.extend([remaining_indices[i] for i in top_indices])
        else:
            # Need to remove images
            num_to_remove = abs(difference)
            selected_features = features[selected_indices]
            nc_scores = self.compute_neighbor_centrality(selected_features)

            # Remove images with lowest NC scores
            indices_to_keep = np.argsort(nc_scores)[num_to_remove:]
            selected_indices = [selected_indices[i] for i in indices_to_keep]

        return selected_indices


class RandomSelector:
    """Baseline: Random selection without clustering or NC scores."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def select_images(
        self,
        images: List[torch.Tensor],
        budget: int,
        task_name: str = "task",
        show_progress: bool = True
    ) -> List[int]:
        """Randomly select images."""
        num_images = len(images)
        budget = min(budget, num_images)

        selected_indices = np.random.choice(
            num_images,
            size=budget,
            replace=False
        ).tolist()

        print(f"Randomly selected {len(selected_indices)} images for task '{task_name}'")

        return selected_indices
