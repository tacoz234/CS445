"""CIFAR-10 Challenge!

You have 75 minutes to maximize test-set accuracy on CIFAR-10
(https://www.cs.toronto.edu/~kriz/cifar.html) using a feedforward
(MLP) neural network.  Good luck!

Once this script is running you can access the associated TensorBoard
logs by running:

    tensorboard --logdir=runs

Then open http://localhost:6006/ in a web browser.

*** ADVICE FOR TUNING ***

1) Increase the capacity (width and/or depth) of your network until
   you start to see overfitting.

2) Tune your hyperparameters to address overfitting.
   Possible hyperparameters to explore:
   - Number of training epochs (track validation error)
   - Network size and shape
   - L2 regularization / Weight Decay
   - Dropout
   - Batch normalization

3) Once you think you are doing as well as possible on the validation
   data, execute the testing code.
   IF YOU DO THIS MORE THAN ONCE, YOU ARE "CHEATING": tuning on the
   testing data invalidates your model evaluation.

Note that, since this is image data, we could do significantly better
by using image-specific approaches including convolutional neural
networks and data augmentation. **Don’t do these things.** The point
of this exercise is to practice tuning neural network hyperparameters,
not to explore optimizations that are specific to a particular type of
data.

"""

# References:
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt


# ----------------------------
# Device selection
# ----------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")


# ----------------------------
# Reproducibility (optional but recommended)
# ----------------------------
def seed_all(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(42)
g = torch.Generator().manual_seed(42)


# ----------------------------
# Helpers
# ----------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_cifar_images(dataset, classes):
    """Show a 5x5 grid of (unnormalized) images from a normalized dataset.

    Assumes Normalize(mean=.5, std=.5) per channel; if you change normalization,
    adjust this unnormalize accordingly.
    """
    plt.figure(figsize=(7, 7))
    for i in range(25):
        img, label = dataset[i]  # C x H x W tensor
        img = img / 2 + 0.5      # unnormalize for mean=.5, std=.5
        npimg = img.numpy()
        plt.subplot(5, 5, i + 1)
        plt.title(classes[label], fontsize=8)
        plt.axis("off")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    # plt.show()


def plot_weights(weights: np.ndarray):
    num_show = min(100, weights.shape[0])
    rows = int(np.ceil(np.sqrt(num_show)))

    if weights.ndim == 2:
        # Likely MLP: (num_units, features)
        plt.figure(figsize=(8, 8))
        for i in range(num_show):
            vec = weights[i, :]
            if vec.size == 3 * 32 * 32:
                img = vec.reshape(3, 32, 32).transpose((1, 2, 0))
                mn, mx = np.min(img), np.max(img)
                rng = mx - mn if mx > mn else 1.0
                img = (img - mn) / rng
                plt.subplot(rows, rows, i + 1)
                plt.axis("off")
                plt.imshow(img)
            else:
                # Fallback: show as square grayscale
                side = int(np.ceil(np.sqrt(vec.size)))
                pad = side * side - vec.size
                padded = np.pad(vec, (0, pad), mode="constant")
                img = padded.reshape(side, side)
                mn, mx = np.min(img), np.max(img)
                rng = mx - mn if mx > mn else 1.0
                img = (img - mn) / rng
                plt.subplot(rows, rows, i + 1)
                plt.axis("off")
                plt.imshow(img, cmap="viridis", interpolation="nearest")
        plt.tight_layout()
        # plt.show()

    elif weights.ndim == 4:
        # Conv2d: (out_channels, in_channels, kH, kW)
        out_ch, in_ch, kH, kW = weights.shape
        plt.figure(figsize=(8, 8))
        for i in range(min(num_show, out_ch)):
            kernel = weights[i]  # (in_ch, kH, kW)
            if in_ch == 3:
                # RGB visualization: (kH, kW, 3)
                img = np.transpose(kernel, (1, 2, 0))
                mn, mx = np.min(img), np.max(img)
                rng = mx - mn if mx > mn else 1.0
                img = (img - mn) / rng
                # Upscale small kernels for readability
                img = np.kron(img, np.ones((12, 12, 1)))
                plt.subplot(rows, rows, i + 1)
                plt.axis("off")
                plt.imshow(img, interpolation="nearest")
            else:
                # Average across channels to visualize as grayscale
                img = kernel.mean(axis=0)  # (kH, kW)
                mn, mx = np.min(img), np.max(img)
                rng = mx - mn if mx > mn else 1.0
                img = (img - mn) / rng
                img = np.kron(img, np.ones((12, 12)))
                plt.subplot(rows, rows, i + 1)
                plt.axis("off")
                plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.tight_layout()
        # plt.show()
    else:
        # Generic fallback
        plt.figure(figsize=(8, 8))
        for i in range(num_show):
            arr = np.ravel(weights[i])
            side = int(np.ceil(np.sqrt(arr.size)))
            pad = side * side - arr.size
            arr = np.pad(arr, (0, pad), mode="constant")
            img = arr.reshape(side, side)
            mn, mx = np.min(img), np.max(img)
            rng = mx - mn if mx > mn else 1.0
            img = (img - mn) / rng
            plt.subplot(rows, rows, i + 1)
            plt.axis("off")
            plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.tight_layout()
        # plt.show()


def train(dataloader, model, loss_fn, optimizer):
    """Perform a single epoch of training."""
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward + loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            print(".", end="", flush=True)

    print(f"  Epoch time: {time.time() - start_time:.2f}s")
    return total_loss / len(dataloader)


def evaluate(dataloader, model, loss_fn):
    """Evaluation helper."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).float().sum().item()

    return test_loss / num_batches, correct / size


# ----------------------------
# Main experiment
# ----------------------------
def main():
    batch_size = 128
    epochs = 40
    learning_rate = 1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ----------------------------
    # Data
    # ----------------------------
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(
        train_set, [0.9, 0.1], generator=g
    )

    pin = (device == "cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=pin, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=pin, persistent_workers=True
    )

    classes = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    # Show some sample images
    plot_cifar_images(train_set, classes)

    # ----------------------------
    # Model
    # ----------------------------
    # Starter MLP (students should widen/deepen, add dropout/bn, etc.)
    model = nn.Sequential(
        nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(40, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    print(model)
    print("Total parameters:", count_parameters(model))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    # ----------------------------
    # TensorBoard
    # ----------------------------
    writer = SummaryWriter()
    imgs, _ = next(iter(train_loader))
    writer.add_graph(model, imgs.to(device))

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss_epoch = train(train_loader, model, loss_fn, optimizer)

        train_loss, train_acc = evaluate(train_loader, model, loss_fn)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)

        print(
            f"  Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
        )

        # Store data for TensorBoard visualization
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        torch.save(model, "latest_model.pt")

    # ----------------------------
    # Visualize first-layer weights
    # ----------------------------
    wts = model[0].weight.detach().cpu().numpy()
    plot_weights(wts)

    writer.close()


if __name__ == "__main__":
    main()
