import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from shared.utils.train import TrainConfig, fit

import matplotlib.pyplot as plt

def make_classification_data(n = 4000):
    torch.manual_seed(42)
    x0 = torch.randn(n//2, 2) + torch.tensor([-2.0, 0.0])
    x1 = torch.randn(n//2, 2) + torch.tensor([2.0, 0.0])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n//2), torch.ones(n//2)]).long()
    return x, y

def main():
    x,y = make_classification_data()
   
    ds = TensorDataset(x,y)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = nn.Sequential(
        nn.Linear(2,32),
        nn.ReLU(),
        nn.Linear(32,2)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

    cfg = TrainConfig(epochs=15, log_every=100, grad_clip=1.0, amp=False)
    history = fit(model, train_loader, val_loader, optimizer, loss_fn, cfg)

    plt.figure()
    plt.plot(history["train_loss"], label = "train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()


if __name__ == "__main__":
    main()  