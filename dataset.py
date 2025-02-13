from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import STL10
from torchvision.transforms import v2

class STL10Align(STL10):
    """
    Custom STL10 dataset for handling aligned and invariant transformations.
    """
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        folds: Optional[int] = None,
        base_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        aligned_transform: Optional[Callable] = None,
        invariant_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            folds=folds,
            transform=base_transform,
            target_transform=target_transform,
            download=download,
        )
        self.base_transform = base_transform
        self.aligned_transform = aligned_transform
        self.invariant_transform = invariant_transform

    def __getitems__(self, possibly_batched_index):
        half_batch_size = len(possibly_batched_index) // 2
        batched_item = []

        for i in range(half_batch_size):
            idx1 = possibly_batched_index[2 * i]
            idx2 = possibly_batched_index[2 * i + 1]

            if self.labels is not None:
                img1, target1 = self.data[idx1], int(self.labels[idx1])
                img2, target2 = self.data[idx2], int(self.labels[idx2])
            else:
                img1, target1 = self.data[idx1], None
                img2, target2 = self.data[idx2], None

            img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
            img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))

            if self.base_transform:
                img11, img12 = self.base_transform(img1), self.base_transform(img1)
                img21, img22 = self.base_transform(img2), self.base_transform(img2)
            else:
                img11, img12 = img1, img1
                img21, img22 = img2, img2

            if self.target_transform:
                target1 = self.target_transform(target1)
                target2 = self.target_transform(target2)

            if self.aligned_transform:
                paired_img1 = torch.stack([img11, img21])
                paired_img2 = torch.stack([img12, img22])

                paired_img1 = self.aligned_transform(paired_img1)
                paired_img2 = self.aligned_transform(paired_img2)

                img11, img12 = paired_img1[0], paired_img2[0]
                img21, img22 = paired_img1[1], paired_img2[1]

            # Invariant transform
            if self.invariant_transform:
                img11, img12 = self.invariant_transform(img11), self.invariant_transform(img12)
                img21, img22 = self.invariant_transform(img21), self.invariant_transform(img22)

            batched_item.append((img11, img12, target1))
            batched_item.append((img21, img22, target2))

        return batched_item


def build_transforms(mean: torch.Tensor, std: torch.Tensor):
    """
    Build base, aligned, and invariant transforms for STL10.
    """
    base_transform = v2.Compose([
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])

    aligned_transform = torch.nn.Sequential(
        v2.RandomResizedCrop(96, scale=(0.2, 1.0)),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
            p=0.8
        ),
        v2.RandomGrayscale(p=0.2)
    )

    invariant_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5
        ),
        v2.Normalize(mean, std)
    ])

    return base_transform, aligned_transform, invariant_transform