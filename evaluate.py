import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor

from models.modern_lenet import ModernLeNet


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images.float()).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ModernLeNet().to(device)
    weights = torch.load('results/checkpoints/best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(weights)

    test_dataset = mnist.MNIST(root='test', train=False, transform=ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    acc = evaluate(model, test_loader, device)
    print(f'test accuracy: {acc:.4f}')

    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/evaluation.json', 'w') as metrics_file:
        json.dump({'checkpoint': 'best_model.pt', 'test_accuracy': acc}, metrics_file, indent=2)
