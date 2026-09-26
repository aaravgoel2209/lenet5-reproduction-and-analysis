import csv
from pathlib import Path

import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from models.modern_lenet import ModernLeNet
from evaluate import evaluate


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_dataset = mnist.MNIST(root=project_dir / 'train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root=project_dir / 'test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = ModernLeNet().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    best_acc = -1.0

    metrics_dir = project_dir / 'results' / 'metrics'
    checkpoint_dir = project_dir / 'results' / 'checkpoints'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Each run starts a new metrics file and replaces the best checkpoint.
    with (metrics_dir / 'training_metrics.csv').open('w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(['epoch', 'train_loss', 'test_accuracy'])

        for current_epoch in range(all_epoch):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for train_x, train_label in train_loader:
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                sgd.zero_grad()
                predict_y = model(train_x.float())
                loss = loss_fn(predict_y, train_label.long())
                loss.backward()
                sgd.step()
                total_loss += loss.item() * train_label.size(0)
                total_samples += train_label.size(0)

            train_loss = total_loss / total_samples
            # This baseline uses the test set for model selection; see README.
            acc = evaluate(model, test_loader, device)
            writer.writerow([current_epoch + 1, train_loss, acc])
            metrics_file.flush()
            print('epoch: {} loss: {:.4f} test accuracy: {:.4f}'.format(
                current_epoch + 1, train_loss, acc), flush=True)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), checkpoint_dir / 'best_model.pt')

    print('Model finished training')
