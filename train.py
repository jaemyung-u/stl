import wandb
from tqdm import tqdm
from accelerate import Accelerator


def train_one_epoch(
    loader,
    model,
    optimizer,
    accelerator: Accelerator,
    epoch: int,
    args,
    log_interval: int = 100,
):
    """
    Single epoch training loop.
    Returns:
        avg_total_loss, avg_inv_loss, avg_equi_loss, avg_trans_loss
    """
    model.train()
    total_steps = len(loader)

    running_total_loss = 0.0
    running_inv_loss = 0.0
    running_equi_loss = 0.0
    running_trans_loss = 0.0

    progress_bar = tqdm(loader, desc='step ', leave=False, dynamic_ncols=True, ascii=True)
    for step, (x1, x2, _) in enumerate(progress_bar, start=epoch * total_steps):
        x1, x2 = x1.to(args.device), x2.to(args.device)
        optimizer.zero_grad()

        total_loss, inv_loss, equi_loss, trans_loss = model.forward(x1, x2)
        accelerator.backward(total_loss)
        optimizer.step()

        running_total_loss += total_loss.item()
        running_inv_loss += inv_loss.item()
        running_equi_loss += equi_loss.item()
        running_trans_loss += trans_loss.item()

        if hasattr(optimizer, "optimizer"):
            lr = optimizer.optimizer.param_groups[0]['lr']
        else:
            lr = optimizer.param_groups[0]['lr']

        progress_bar.set_postfix({
            "lr": f"{lr:.4f}",
            "loss": f"{total_loss.item():.2f}",
            "inv": f"{inv_loss.item():.2f}",
            "equi": f"{equi_loss.item():.2f}",
            "trans": f"{trans_loss.item():.2f}"
        })

        if (step % log_interval == 0) and accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "lr": lr,
                "loss": total_loss.item(),
                "inv": inv_loss.item(),
                "equi": equi_loss.item(),
                "trans": trans_loss.item()
            }, step=step+1)

    avg_total_loss = running_total_loss / total_steps
    avg_inv_loss = running_inv_loss / total_steps
    avg_equi_loss = running_equi_loss / total_steps
    avg_trans_loss = running_trans_loss / total_steps

    return avg_total_loss, avg_inv_loss, avg_equi_loss, avg_trans_loss