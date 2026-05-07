import torch
import torch.nn as nn

class PhysicsHead(nn.Module):
    """Predict absorption wavelength; optionally regress in eV then convert to nm."""
    def __init__(self, hidden: int, energy_space: bool = True):
        super().__init__()
        self.energy_space = energy_space
        self.abs = nn.Linear(hidden, 1)
    @staticmethod
    def ev_to_nm(E):
        return 1240.0 / (E.clamp_min(1e-6))

    def forward(self, h):
        if self.energy_space:
            E_abs = self.abs(h)
            lam_abs = self.ev_to_nm(E_abs)
            return lam_abs
        else:
            lam_abs = self.abs(h)
            return lam_abs
