from torchvision import datasets
from torchvision.transforms import ToTensor

datasets.FashionMNIST(root="data/raw", train=True, download=True, transform=ToTensor())
datasets.FashionMNIST(root="data/raw", train=False, download=True, transform=ToTensor())
print("Dataset téléchargé.")