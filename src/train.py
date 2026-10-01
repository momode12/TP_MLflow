import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch


class NeuralNetwork(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def main(args):
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root="data/raw", train=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="data/raw", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device :", device)

    model = NeuralNetwork(args.hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment("Fashion-MNIST")

    with mlflow.start_run(run_name=args.name):
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("hidden_neurons", args.hidden)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Loss: {average_loss:.4f}")
            mlflow.log_metric("train_loss", average_loss, step=epoch)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy : {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, name="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--name", type=str, default=None)
    main(parser.parse_args())