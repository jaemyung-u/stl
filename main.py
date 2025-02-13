import argparse
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator

# Local imports
from dataset import STL10Align, build_transforms
from model import STL
from train import train_one_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='STL Training')
    parser.add_argument('--data-dir', default='/data', type=str, help='Dataset directory')
    parser.add_argument('--save-dir', default='/exp', type=str, help='Checkpoint directory')
    parser.add_argument('--num-workers', default=16, type=int, help='Number of data loader workers')
    parser.add_argument('--log-interval', default=100, type=int, help='step interval for wandb logging')

    parser.add_argument('--lr', default=0.03, type=float, help='Initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--batch-size', default=256, type=int, help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Training epochs')

    parser.add_argument('--projector', default='512-128', type=str, help='Representation projector layers')
    parser.add_argument('--trans-backbone', default='128-128', type=str, help='Transformation backbone layers')
    parser.add_argument('--trans-projector', default='128-128', type=str, help='Transformation representation projector layers')

    parser.add_argument('--inv', default=1.0, type=float, help='Invariance loss weight')
    parser.add_argument('--equi', default=1.0, type=float, help='Equivariance loss weight')
    parser.add_argument('--trans', default=0.1, type=float, help='Transformation loss weight')
    parser.add_argument('--temperature', default=0.2, type=float, help='InfoNCE temperature')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='no')
    args.device = accelerator.device

    # Initialize W&B (only on main process)
    if accelerator.is_main_process:
        wandb.init(project='stl', config=args)
        run_id = wandb.run.id
        run_name = f'stl-stl10-{run_id}'
        wandb.run.name = run_name
        wandb.run.save()

    # Model, optimizer, scheduler
    model = STL(args).to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Mean/STD for STL10
    mean = torch.tensor([0.43, 0.42, 0.39])
    std = torch.tensor([0.27, 0.26, 0.27])

    # Build transforms
    base_transform, aligned_transform, invariant_transform = build_transforms(mean, std)

    # Dataset
    dataset = STL10Align(
        root=args.data_dir,
        split='train+unlabeled',
        base_transform=base_transform,
        download=True,
        aligned_transform=aligned_transform,
        invariant_transform=invariant_transform
    )

    # Sampler & DataLoader
    sampler = DistributedSampler(dataset) if accelerator.state.num_processes > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size // accelerator.state.num_processes,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Prepare
    loader, model, optimizer, scheduler = accelerator.prepare(loader, model, optimizer, scheduler)

    # Progress Bar
    progress_bar = tqdm(range(args.epochs), desc='epoch', dynamic_ncols=True, ascii=True)

    # Training Loop
    for epoch in progress_bar:
        avg_total, avg_inv, avg_equi, avg_trans = train_one_epoch(
            loader=loader,
            model=model,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
            args=args,
            log_interval=args.log_interval
        )

        if hasattr(optimizer, "optimizer"):
            lr = optimizer.optimizer.param_groups[0]['lr']
        else:
            lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        progress_bar.set_postfix({
            "lr": f"{lr:.4f}",
            "loss": f"{avg_total:.2f}",
            "inv": f"{avg_inv:.2f}",
            "equi": f"{avg_equi:.2f}",
            "trans": f"{avg_trans:.2f}",
        })

    # Save the final backbone
    if accelerator.is_main_process:
        checkpoint_path = f"{args.save_dir}/checkpoint_{run_name}.pth"
        torch.save(model.backbone.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    wandb.finish()

if __name__ == '__main__':
    main()