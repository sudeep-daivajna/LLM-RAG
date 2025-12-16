from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader 



@dataclass
class TrainConfig:
    epochs: int = 10
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: Optional[float] = None   #to prevent exploding gradient
    amp: bool = False #32 bits

@torch.no_grad()
def accuracy(logits: torch.tensor, y:torch.tensor) -> float:
    preds = logits.argmax(dim = -1) # 0 : across rows, 1/-1 : across columns in (batchsize, logits{model predicted values before activation})
    return (preds == y).float().mean().item()

def train_one_epoch(
        model : nn.Module,
        loader : DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        cfg: TrainConfig,
) -> Dict[str,float]:
    model.train()
    device = cfg.device

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp) #helps with automatic mixed precision training, it helps scale the loss to prevent underflow when using float16 precision. Basically when calculating the gradient, if we are using float16, the small values can get rounded to zero (underflow), leading to inaccurate gradients. The GradScaler scales up the loss value before backpropagation, which helps keep the gradient values in a representable range for float16. After backpropagation, it then unscales the gradients back down to their original range before the optimizer step.

    for step, (x,y) in enumerate(loader, start=1):
        x,y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True) #if the gradient becomes a tensor of 0's (if the weight is a matrix) then we change it to None, its just to save memory during training

        with torch.cuda.amp.autocast(enabled = cfg.amp):
            logits = model(x)   #logits are the raw predictions from the model before applying activation function
            loss = loss_fn(logits,y)
        
        scaler.scale(loss).backward() #scales the loss, and calls backward on the scaled loss to create scaled gradients. keep in mind, scaling the loss scales the gradients too since gradients are computed as the derivative of the loss w.r.t model parameters

        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer) #unscale the gradients of the optimizer's assigned params in order to clip them
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) #clips the gradients to prevent exploding gradients. We pass the parameters of the model to clip their gradients
        
        scaler.step(optimizer) # the actual update step, where we update the model parameters
        scaler.update() # if the gradients were too large or too small this time around, the scaler will tweak that scaling factor so that next time, the gradients are in a more comfortable range. Basically, it helps keep things stable and efficient over the course of your training.

        batch_loss = loss.item()
        running_loss += batch_loss
        running_acc += accuracy(logits, y)
        n_batches += 1

        if cfg.log_every and step % cfg.log_every == 0:
            print(f"  step {step:5d} | loss {batch_loss:.4f}")

        return {
            "loss" : running_loss / max(n_batches, 1), # we use max to prevent division by zero
            "acc"  : running_acc / max(n_batches, 1)
        }


@torch.no_grad()
def evaluate(    
    model : nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: str
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits,y)

        running_loss += loss.item()
        running_acc += accuracy(logits,y)
        n_batches += 1
    
    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1),
    }

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    cfg: TrainConfig,
) -> Dict[str, list]:
    model.to(cfg.device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])

        line = f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f}" #02d means pad with 0's to make it 2 digits

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, cfg.device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            line += f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f}"

        print(line)

    return history











    