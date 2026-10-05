import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

class MaskedLowRankAnatomyModel:
    """
    Masked Low-Rank Anatomical Deformation Prior.
    Represents joint internal anatomy as:
        P_i \approx A + U * z_i
    where:
        A \in \mathbb{R}^{3K} is the training mean anatomy
        U \in \mathbb{R}^{3K \times d} is the orthonormal deformation basis
        z_i \in \mathbb{R}^d is the patient-specific anatomical deformation code
    """
    def __init__(self, num_targets: int = 107, latent_dim: int = 16):
        self.K = num_targets
        self.coords_dim = 3 * num_targets
        self.d = latent_dim
        self.A = None            # (3K,) mean anatomy vector
        self.U = None            # (3K, d) orthonormal basis
        self.mode_stds = None    # (d,) standard deviations along modes

    def compute_mean_anatomy(self, train_targets: np.ndarray, train_masks: np.ndarray) -> np.ndarray:
        """
        Computes training-only mean anatomy vector A.
        train_targets: (N, K, 3)
        train_masks: (N, K)
        Returns A: (3K,)
        """
        N, K, _ = train_targets.shape
        A_matrix = np.zeros((K, 3), dtype=np.float32)
        for k in range(K):
            valid_k = train_masks[:, k] > 0.5
            if np.any(valid_k):
                A_matrix[k] = train_targets[valid_k, k].mean(axis=0)
            else:
                A_matrix[k] = 0.0
        self.A = A_matrix.reshape(-1) # (3K,)
        return self.A

    def fit(
        self,
        train_targets: np.ndarray,      # (N, K, 3)
        train_masks: np.ndarray,        # (N, K)
        latent_dim: Optional[int] = None,
        epochs: int = 800,
        lr: float = 0.02,
        lambda_z: float = 1e-4,
        lambda_u: float = 1e-4,
        target_balanced: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, float]:
        """
        Fits U and Z on training data using differentiable masked low-rank factorization.
        Missing coordinates contribute zero loss.
        """
        if latent_dim is not None:
            self.d = latent_dim

        N, K, _ = train_targets.shape
        if self.A is None:
            self.compute_mean_anatomy(train_targets, train_masks)

        # Flatten targets to (N, 3K) and masks to (N, 3K)
        P_train = train_targets.reshape(N, 3 * K)
        M_train = np.repeat(train_masks > 0.5, 3, axis=1).astype(np.float32) # (N, 3K)
        Y_train = P_train - self.A[None, :] # Centered around training mean

        P_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        M_t = torch.tensor(M_train, dtype=torch.float32, device=device)

        # Target weights for balanced loss
        if target_balanced:
            target_counts = train_masks.sum(axis=0).clip(min=1.0) # (K,)
            k_weights = 1.0 / target_counts # (K,)
            k_weights = k_weights / k_weights.mean() # normalized
            weights_3k = np.repeat(k_weights, 3).astype(np.float32) # (3K,)
            W_t = torch.tensor(weights_3k, dtype=torch.float32, device=device).unsqueeze(0) # (1, 3K)
        else:
            W_t = torch.ones((1, 3 * K), dtype=torch.float32, device=device)

        # Initialize U and Z with small random values
        torch.manual_seed(42)
        U_param = nn.Parameter(torch.randn(3 * K, self.d, device=device) * 0.05)
        Z_param = nn.Parameter(torch.randn(N, self.d, device=device) * 0.05)

        opt = torch.optim.AdamW([U_param, Z_param], lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_loss = 1e9
        for ep in range(epochs):
            opt.zero_grad()
            # Predicted centered anatomy: (N, 3K)
            Y_pred = torch.matmul(Z_param, U_param.t())
            diff = (P_t - Y_pred) * M_t
            rec_loss = torch.mean(W_t * (diff ** 2))
            reg_loss = lambda_z * torch.mean(Z_param ** 2) + lambda_u * torch.mean(U_param ** 2)
            total_loss = rec_loss + reg_loss
            total_loss.backward()
            opt.step()
            sched.step()

        # Extract U and Z to CPU
        with torch.no_grad():
            U_raw = U_param.cpu().numpy() # (3K, d)
            Z_raw = Z_param.cpu().numpy() # (N, d)
            Y_pred_raw = np.matmul(Z_raw, U_raw.T)

        # Orthogonalize basis via SVD of the reconstructed variance
        # Y_pred = Z * U^T. SVD of Y_pred gives true principal directions
        # Or SVD of Z_raw @ U_raw^T:
        U_q, S_q, Vh_q = np.linalg.svd(Y_pred_raw, full_matrices=False)
        # Top d components:
        # Vh_q has shape (d, 3K). Vh_q.T has shape (3K, d)
        U_orth = Vh_q[:self.d].T # (3K, d) orthonormal columns
        Z_orth = np.matmul(Y_pred_raw, U_orth) # (N, d)
        mode_stds = np.std(Z_orth, axis=0) # (d,)

        self.U = U_orth.astype(np.float32)
        self.mode_stds = mode_stds.astype(np.float32)

        final_loss = float(rec_loss.item())
        return {
            "final_rec_loss": final_loss,
            "explained_variance": float(np.sum(mode_stds ** 2)),
            "latent_dim": self.d
        }

    def project_ground_truth(
        self,
        targets: np.ndarray,            # (N, K, 3)
        masks: np.ndarray,              # (N, K)
        lambda_reg: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves for the optimal latent code z_i^* given observed ground truth (D0 Oracle):
            z_i^* = argmin_z || M_i \odot (P_i - A - U z) ||^2 + lambda ||z||^2
        Closed-form ridge regression per patient.
        Returns:
            z_star: (N, d)
            P_oracle: (N, K, 3)
        """
        N, K, _ = targets.shape
        P_flat = targets.reshape(N, 3 * K)
        M_flat = np.repeat(masks > 0.5, 3, axis=1) # (N, 3K)
        Y_flat = P_flat - self.A[None, :]

        z_star = np.zeros((N, self.d), dtype=np.float32)
        P_oracle_flat = np.zeros_like(P_flat)

        I_d = np.eye(self.d, dtype=np.float32) * lambda_reg

        for i in range(N):
            obs_idx = np.where(M_flat[i])[0]
            if len(obs_idx) == 0:
                z_star[i] = 0.0
                P_oracle_flat[i] = self.A
                continue
            U_obs = self.U[obs_idx] # (len(obs), d)
            y_obs = Y_flat[i, obs_idx] # (len(obs),)

            # Solve: (U_obs^T U_obs + lambda I) z = U_obs^T y_obs
            A_mat = np.matmul(U_obs.T, U_obs) + I_d
            b_vec = np.matmul(U_obs.T, y_obs)
            z_star[i] = np.linalg.solve(A_mat, b_vec)
            P_oracle_flat[i] = self.A + np.matmul(self.U, z_star[i])

        P_oracle = P_oracle_flat.reshape(N, K, 3)
        return z_star, P_oracle

    def project_predictions(
        self,
        p0: np.ndarray,                 # (N, K, 3) coarse predictions
        weights: Optional[np.ndarray] = None, # (N, K) or None
        lambda_reg: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects coarse predictions into the anatomical subspace (A1 Post-Hoc Projection):
            z_hat = argmin_z || W (P0 - A - U z) ||^2 + lambda ||z||^2
        Returns:
            z_hat: (N, d)
            P_proj: (N, K, 3)
        """
        N, K, _ = p0.shape
        P0_flat = p0.reshape(N, 3 * K)
        Y0_flat = P0_flat - self.A[None, :]

        z_hat = np.zeros((N, self.d), dtype=np.float32)
        P_proj_flat = np.zeros_like(P0_flat)

        I_d = np.eye(self.d, dtype=np.float32) * lambda_reg

        if weights is not None:
            W_flat = np.repeat(weights, 3, axis=1).astype(np.float32)
        else:
            W_flat = np.ones((N, 3 * K), dtype=np.float32)

        for i in range(N):
            w = W_flat[i] # (3K,)
            # U^T W U
            U_w = self.U * w[:, None]
            A_mat = np.matmul(self.U.T, U_w) + I_d
            b_vec = np.matmul(U_w.T, Y0_flat[i])
            z_hat[i] = np.linalg.solve(A_mat, b_vec)
            P_proj_flat[i] = self.A + np.matmul(self.U, z_hat[i])

        P_proj = P_proj_flat.reshape(N, K, 3)
        return z_hat, P_proj

    def soft_projection(
        self,
        p0: np.ndarray,                 # (N, K, 3)
        g: float = 0.5,
        weights: Optional[np.ndarray] = None,
        lambda_reg: float = 1e-3
    ) -> np.ndarray:
        """
        Applies soft projection: P_soft = P0 + g * (P_proj - P0)
        """
        _, P_proj = self.project_predictions(p0, weights=weights, lambda_reg=lambda_reg)
        return p0 + g * (P_proj - p0)

    def get_mode_configurations(self, mode_idx: int, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (A - 2*std*U_r, A, A + 2*std*U_r) for mode visualization.
        Shape: (K, 3) each.
        """
        assert self.U is not None, "Model must be fitted first"
        sigma_r = self.mode_stds[mode_idx]
        u_r = self.U[:, mode_idx] # (3K,)

        A_mean = self.A.reshape(self.K, 3)
        A_minus = (self.A - num_std * sigma_r * u_r).reshape(self.K, 3)
        A_plus = (self.A + num_std * sigma_r * u_r).reshape(self.K, 3)
        return A_minus, A_mean, A_plus
