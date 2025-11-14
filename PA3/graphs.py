import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from sf_classifiers import MLP
from sf_classifier_examples import noisy_xor


def run_training_runs(activation, hidden_sizes, learning_rate, epochs, runs, n_points):
    losses_per_run = []
    for _ in range(runs):
        x, y = noisy_xor(n_points)
        clf = MLP(2, hidden_sizes, activation=activation)
        losses = clf.train(x, y, learning_rate=learning_rate,
                           epochs=epochs, verbose=False)
        losses_per_run.append(losses)
    return losses_per_run


def plot_runs(losses_per_run, epochs, title, learning_rate):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, losses in enumerate(losses_per_run):
        ax.plot(range(1, epochs + 1), losses, alpha=0.8, label=f"Run {i+1}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title}\nLearning rate: {learning_rate}",
                 fontsize=11, y=0.98)
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.subplots_adjust(top=0.88)
    return fig


def create_figures(learning_rate=0.05, epochs=50, runs=10, n_points=200):
    pdf = PdfPages("learning_curves.pdf")

    print("Training 3-layer sigmoid runs...")
    losses_sigmoid_3 = run_training_runs(
        activation="sigmoid", hidden_sizes=[10], learning_rate=learning_rate,
        epochs=epochs, runs=runs, n_points=n_points
    )
    fig1 = plot_runs(
        losses_sigmoid_3, epochs,
        "Learning curves for 3-layer MLP with sigmoid (10 runs). Each line is a run; either activation is effective for a 3-layer network.",
        learning_rate
    )
    pdf.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)

    print("Training 3-layer relu runs...")
    losses_relu_3 = run_training_runs(
        activation="relu", hidden_sizes=[10], learning_rate=learning_rate,
        epochs=epochs, runs=runs, n_points=n_points
    )
    fig2 = plot_runs(
        losses_relu_3, epochs,
        "Learning curves for 3-layer MLP with relu (10 runs). Each line is a run; either activation is effective for a 3-layer network.",
        learning_rate
    )
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)

    deep_hidden = [10, 10, 10, 10, 10]

    print("Training 5-hidden-layer sigmoid runs...")
    losses_sigmoid_5 = run_training_runs(
        activation="sigmoid", hidden_sizes=deep_hidden, learning_rate=learning_rate,
        epochs=epochs, runs=runs, n_points=n_points
    )
    fig3 = plot_runs(
        losses_sigmoid_5, epochs,
        "Learning curves for 5-hidden-layer MLP with sigmoid (10 units per layer, 10 runs). Learning question: does relu learn better in deeper networks?",
        learning_rate
    )
    pdf.savefig(fig3, bbox_inches='tight')
    plt.close(fig3)

    print("Training 5-hidden-layer relu runs...")
    losses_relu_5 = run_training_runs(
        activation="relu", hidden_sizes=deep_hidden, learning_rate=learning_rate,
        epochs=epochs, runs=runs, n_points=n_points
    )
    fig4 = plot_runs(
        losses_relu_5, epochs,
        "Learning curves for 5-hidden-layer MLP with relu (10 units per layer, 10 runs). Learning question: does relu learn better in deeper networks?",
        learning_rate
    )
    pdf.savefig(fig4, bbox_inches='tight')
    plt.close(fig4)

    pdf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--npoints", type=int, default=200)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    runs = args.runs
    npoints = args.npoints

    if args.fast:
        epochs = min(epochs, 50)
        npoints = min(npoints, 200)

    print(
        f"Settings: lr={lr}, epochs={epochs}, runs={runs}, npoints={npoints}")
    create_figures(learning_rate=lr, epochs=epochs,
                   runs=runs, n_points=npoints)
