import torch
from labels import ORGAN_NAMES


def calculate_physical_error(
    pred_norm: torch.Tensor,    # (B, N, 3) normalized [0,1]
    gt_norm: torch.Tensor,      # (B, N, 3)
    spacing_mm: torch.Tensor,   # (B, 3)
    target_shape: tuple = (128, 128, 128),
) -> torch.Tensor:              # (B, N)  error in mm
    shape_t = torch.tensor(target_shape, dtype=torch.float32, device=pred_norm.device)
    pred_vox = pred_norm * shape_t                          # (B,N,3)
    gt_vox   = gt_norm   * shape_t
    diff_vox = pred_vox - gt_vox                            # (B,N,3)
    # spacing_mm: (B,3) → (B,1,3)
    diff_mm  = diff_vox * spacing_mm.unsqueeze(1)
    return torch.norm(diff_mm, dim=-1)                      # (B,N)


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