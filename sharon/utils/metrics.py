import torch
from labels import ORGAN_NAMES


def calculate_physical_error(
    pred_norm: torch.Tensor,    # (B, N, 3) normalized [z, y, x] in [0,1]
    gt_norm: torch.Tensor,      # (B, N, 3) normalized [z, y, x] in [0,1]
    spacing_mm: torch.Tensor,   # (B, 3) physical spacing [dx, dy, dz] in mm
    target_shape: tuple = (128, 128, 128),  # (D, H, W)
) -> torch.Tensor:              # (B, N) error in mm
    device = pred_norm.device
    # Re-order normalized (z, y, x) -> (x, y, z) to match physical spacing (dx, dy, dz)
    pred_xyz = torch.stack([pred_norm[..., 2], pred_norm[..., 1], pred_norm[..., 0]], dim=-1)
    gt_xyz   = torch.stack([gt_norm[..., 2],   gt_norm[..., 1],   gt_norm[..., 0]],   dim=-1)

    # target_shape is (D, H, W) -> spatial resolution for (x, y, z) is (W, H, D)
    shape_xyz = torch.tensor([target_shape[2], target_shape[1], target_shape[0]], dtype=torch.float32, device=device)

    pred_vox_xyz = pred_xyz * shape_xyz
    gt_vox_xyz   = gt_xyz   * shape_xyz

    diff_vox_xyz = pred_vox_xyz - gt_vox_xyz
    diff_mm = diff_vox_xyz * spacing_mm.unsqueeze(1)   # (B, N, 3) * (B, 1, 3) [dx, dy, dz]
    return torch.norm(diff_mm, dim=-1)                # (B, N) Euclidean distance in mm


class OrganErrorAccumulator:
    def __init__(self, num_organs: int):
        self.num_organs = num_organs
        self.sums   = [0.0] * num_organs
        self.counts = [0]   * num_organs

    def update(self, errors_mm: torch.Tensor, found_mask: torch.Tensor):
        """errors_mm: (B,N), found_mask: (B,N) bool."""
        for b in range(errors_mm.shape[0]):
            for i in range(self.num_organs):
                if found_mask[b, i]:
                    self.sums[i]   += errors_mm[b, i].item()
                    self.counts[i] += 1

    def per_organ(self) -> dict:
        return {
            ORGAN_NAMES[i]: (self.sums[i] / self.counts[i] if self.counts[i] else float("nan"))
            for i in range(self.num_organs)
        }

    def overall(self) -> float:
        total, count = sum(self.sums), sum(self.counts)
        return total / count if count else float("nan")

    def reset(self):
        self.sums   = [0.0] * self.num_organs
        self.counts = [0]   * self.num_organs