import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from pathlib import Path

from model import SelfPruningNet


def get_loaders(batch_size=128):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=tf_train)
    test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=tf_test)
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2),
            DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2))


def test_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    return 100 * correct / total


def train(lam, epochs=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nlambda={lam}  device={device}")

    train_loader, test_loader = get_loaders()
    model = SelfPruningNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    ce    = nn.CrossEntropyLoss()

    history = {'lam': lam, 'acc': [], 'sparsity': [], 'loss': []}

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()

            loss = ce(model(imgs), lbls) + lam * model.sparsity_loss()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        acc      = test_accuracy(model, test_loader, device)
        sparsity = model.overall_sparsity() * 100

        history['acc'].append(round(acc, 2))
        history['sparsity'].append(round(sparsity, 2))
        history['loss'].append(round(running_loss / len(train_loader), 4))

        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} | loss {running_loss/len(train_loader):.4f} "
                  f"| acc {acc:.2f}% | sparsity {sparsity:.1f}%")

    # save gate values for the plot
    gates = []
    for layer in [model.l1, model.l2, model.l3]:
        gates += layer.get_gates().cpu().numpy().flatten().tolist()

    history['final_acc']      = acc
    history['final_sparsity'] = sparsity
    history['gates']          = gates

    Path(f'./checkpoints/lambda_{lam}').mkdir(parents=True, exist_ok=True)
    with open(f'./checkpoints/lambda_{lam}/history.json', 'w') as f:
        json.dump(history, f)

    return history


if __name__ == '__main__':
    train(lam=1e-4, epochs=5)