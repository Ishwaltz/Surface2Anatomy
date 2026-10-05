from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from labels import ORGAN_NAMES, SKELETAL_INDICES, SOFT_TISSUE_INDICES
from sharon.anatomical_prior import MaskedLowRankAnatomyModel

# Canonical Biological Regional Definitions for 107 Primary Targets
def get_canonical_regions(primary_107_indices: List[int]) -> Dict[str, List[int]]:
    """
    Partitions the 107 primary evaluation targets into 5 biologically verified anatomical modules.
    Returns dictionary mapping region_name -> list of local indices in [0, 106].
    """
    regions = {
        "skeletal": [],
        "thoracic": [],
        "upper_abdominal": [],
        "lower_abdominal_pelvic": [],
        "musculoskeletal_vascular": []
    }

    for local_idx, organ_idx in enumerate(primary_107_indices):
        name = ORGAN_NAMES[organ_idx].lower()

        # 1. Skeletal
        if organ_idx in SKELETAL_INDICES:
            regions["skeletal"].append(local_idx)
        # 2. Thoracic Viscera
        elif any(k in name for k in ["lung", "heart", "trachea", "esophagus", "bronch", "thymus", "pulmonary"]):
            regions["thoracic"].append(local_idx)
        # 3. Upper Abdominal Viscera
        elif any(k in name for k in ["liver", "spleen", "kidney", "adrenal", "pancreas", "gallbladder", "stomach"]):
            regions["upper_abdominal"].append(local_idx)
        # 4. Lower Abdominal & Pelvic
        elif any(k in name for k in ["bladder", "prostate", "uterus", "colon", "rectum", "duodenum", "intestine"]):
            regions["lower_abdominal_pelvic"].append(local_idx)
        # 5. Musculoskeletal & Major Vessels
        else:
            regions["musculoskeletal_vascular"].append(local_idx)

    # Ensure no empty regions and full partition
    total_assigned = sum(len(v) for v in regions.values())
    assert total_assigned == len(primary_107_indices), f"Target mismatch: {total_assigned} vs {len(primary_107_indices)}"
    return regions


def compute_residual_correlation(
    residuals_mm: np.ndarray, # (N, K, 3)
    mask: np.ndarray          # (N, K)
) -> np.ndarray:
    """
    Computes empirical target-target residual error correlation matrix (K, K)
    using error vector magnitudes across patients.
    """
    # Error magnitude per target: (N, K)
    err_mag = np.linalg.norm(residuals_mm, axis=-1)
    err_mag = np.nan_to_num(err_mag, nan=0.0)

    # Mean center under mask
    centered = np.zeros_like(err_mag)
    K = err_mag.shape[1]
    for k in range(K):
        m_k = mask[:, k] > 0
        if np.any(m_k):
            centered[m_k, k] = err_mag[m_k, k] - np.mean(err_mag[m_k, k])

    # Compute correlation
    corr = np.corrcoef(centered.T) # (K, K)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def cluster_targets_hierarchical(
    corr_matrix: np.ndarray,
    num_clusters: int = 5
) -> np.ndarray:
    """
    Agglomerative hierarchical clustering on distance matrix D = 1 - abs(corr).
    Returns cluster labels (K,) in [1, num_clusters].
    """
    dist = 1.0 - np.clip(np.abs(corr_matrix), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry
    dist = 0.5 * (dist + dist.T)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, t=num_clusters, criterion="maxclust")
    return clusters


class RegionalLowRankResidualModel:
    """
    Modular Regional Low-Rank Residual Model:
    Decomposes the 107 targets into independent regional modules,
    fitting separate low-rank orthonormal bases U_r for each region.
    """
    def __init__(
        self,
        regions: Dict[str, List[int]],
        regional_dims: Dict[str, int]
    ):
        self.regions = regions
        self.regional_dims = regional_dims
        self.models: Dict[str, MaskedLowRankAnatomyModel] = {}
        for r_name, local_indices in regions.items():
            d_r = regional_dims.get(r_name, 8)
            K_r = len(local_indices)
            self.models[r_name] = MaskedLowRankAnatomyModel(num_targets=K_r, latent_dim=d_r)

    def fit(
        self,
        residuals_all: np.ndarray, # (N, K, 3)
        mask_all: np.ndarray,      # (N, K)
        epochs: int = 600,
        lr: float = 0.03
    ) -> Dict[str, dict]:
        fit_infos = {}
        for r_name, local_indices in self.regions.items():
            res_r = residuals_all[:, local_indices, :]
            mask_r = mask_all[:, local_indices]
            fit_infos[r_name] = self.models[r_name].fit(res_r, mask_r, epochs=epochs, lr=lr)
        return fit_infos

    def project_ground_truth(
        self,
        residuals_all: np.ndarray, # (N, K, 3)
        mask_all: np.ndarray       # (N, K)
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Projects each region's ground truth residuals onto its respective regional basis.
        Returns:
            z_stars_by_region: Dict[r_name -> (N, d_r)]
            reconstructed_residuals: (N, K, 3)
        """
        N, K, _ = residuals_all.shape
        full_recon = np.zeros((N, K, 3), dtype=np.float32)
        z_stars = {}

        for r_name, local_indices in self.regions.items():
            res_r = residuals_all[:, local_indices, :]
            mask_r = mask_all[:, local_indices]
            z_r, recon_r = self.models[r_name].project_ground_truth(res_r, mask_r)
            z_stars[r_name] = z_r
            full_recon[:, local_indices, :] = recon_r

        return z_stars, full_recon
