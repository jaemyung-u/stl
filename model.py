import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


def load_mlp(input_dim: int, layers_str: str) -> nn.Sequential:
    """
    Build a simple MLP given the layer configuration string (e.g., '512-128').
    """
    sizes = [input_dim] + list(map(int, layers_str.split('-')))
    layer_list = []
    for i in range(len(sizes) - 2):
        layer_list.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layer_list.append(nn.BatchNorm1d(sizes[i + 1]))
        layer_list.append(nn.ReLU(inplace=True))
    layer_list.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

    return nn.Sequential(*layer_list)


class EquiTrans(nn.Module):
    """
    Hypernetwork-style transformation module.
    """
    def __init__(self, equi_repr_dim: int, trans_repr_dim: int):
        super().__init__()
        self.equitrans_layers = [equi_repr_dim, equi_repr_dim]

        # Calculate parameters needed for each block
        self.num_weights_per_block = [
            self.equitrans_layers[i] * self.equitrans_layers[i + 1]
            for i in range(len(self.equitrans_layers) - 1)
        ]
        self.cumulative_params = [0] + list(np.cumsum(self.num_weights_per_block))

        # Hypernetwork to generate weights
        self.hypernet = nn.Linear(trans_repr_dim, self.cumulative_params[-1], bias=False)

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation predicted by `t` on representation `r`.
        """
        all_weights = self.hypernet(t)  # shape: [B, total_params]
        output = r.unsqueeze(1)

        # Sequentially apply linear blocks
        for i in range(len(self.equitrans_layers) - 1):
            start_idx = self.cumulative_params[i]
            end_idx = start_idx + self.num_weights_per_block[i]

            w_block = all_weights[..., start_idx:end_idx]
            w_reshaped = w_block.view(-1,
                                      self.equitrans_layers[i + 1],
                                      self.equitrans_layers[i])
            output = torch.bmm(output, w_reshaped.transpose(-2, -1))

        return output.squeeze()


class STL(nn.Module):
    """
    Main model that handles:
    - Backbone (ResNet18)
    - Transformation backbone (MLP for trans_repr)
    - EquiTrans module (hypernet)
    - Projectors for inv, equi, trans
    - Forward pass calculating total loss
    """
    def __init__(self, args):
        super().__init__()
        # Backbone
        self.backbone = resnet18(zero_init_residual=True)
        repr_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        # Transform backbone
        trans_repr_dim = int(args.trans_backbone.split('-')[-1])
        self.trans_backbone = load_mlp(2 * repr_dim, args.trans_backbone)
        self.equi_transform = EquiTrans(repr_dim, trans_repr_dim)

        # Projectors
        self.inv_projector = load_mlp(repr_dim, args.projector)
        self.equi_projector = load_mlp(repr_dim, args.projector)
        self.trans_projector = load_mlp(trans_repr_dim, args.trans_projector)

        # Index helpers for rearranging
        batch_size = args.batch_size
        self.even_idxs = 2 * torch.arange(batch_size // 2)
        self.odd_idxs = 2 * torch.arange(batch_size // 2) + 1
        self.shifted_idxs = torch.flatten(torch.stack([self.odd_idxs, self.even_idxs], dim=1))

        # Loss coefficients
        self.inv = args.inv
        self.equi = args.equi
        self.trans = args.trans
        self.temperature = args.temperature

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the InfoNCE loss between two embeddings z1 and z2.
        """
        device = z1.device
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity scores
        scores = torch.matmul(z, z.t()) / self.temperature
        n = z1.size(0)

        # Labels for matching
        labels = torch.cat([
            torch.arange(n, 2 * n, device=device),
            torch.arange(0, n, device=device)
        ])

        # Mask out self-similarity
        diag_mask = torch.eye(2 * n, dtype=torch.bool, device=device)
        scores = scores.masked_fill(diag_mask, float('-inf'))

        return F.cross_entropy(scores, labels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass for self-supervised training.
        Returns total loss and individual losses: (total_loss, inv_loss, equi_loss, trans_loss).
        """
        # Backbone outputs
        y1, y2 = self.backbone(x1), self.backbone(x2)

        # Transformation backbone
        y_trans122 = self.trans_backbone(torch.cat([y1, y2], dim=-1))
        y_trans221 = self.trans_backbone(torch.cat([y2, y1], dim=-1))

        # Split into y_trans1, y_trans2
        y_trans1 = torch.cat([y_trans122[self.even_idxs], y_trans221[self.even_idxs]], dim=0)
        y_trans2 = torch.cat([y_trans122[self.odd_idxs], y_trans221[self.odd_idxs]], dim=0)

        # Equivariant transform
        y_pred1 = self.equi_transform(y2, y_trans221[self.shifted_idxs])
        y_pred2 = self.equi_transform(y1, y_trans122[self.shifted_idxs])

        # Invariance loss
        z_inv1, z_inv2 = self.inv_projector(y1), self.inv_projector(y2)
        inv_loss = self.info_nce_loss(z_inv1, z_inv2)

        # Equivariance loss
        z_equi = torch.cat([self.equi_projector(y1), self.equi_projector(y2)], dim=0)
        z_equi_pred = torch.cat([self.equi_projector(y_pred1), self.equi_projector(y_pred2)], dim=0)
        equi_loss = self.info_nce_loss(z_equi, z_equi_pred)

        # Transformation loss
        z_trans1, z_trans2 = self.trans_projector(y_trans1), self.trans_projector(y_trans2)
        trans_loss = self.info_nce_loss(z_trans1, z_trans2)

        # Total loss
        total_loss = self.inv * inv_loss + self.equi * equi_loss + self.trans * trans_loss
        return total_loss, inv_loss, equi_loss, trans_loss