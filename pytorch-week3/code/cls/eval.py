import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from dataset import get_cifar10_loaders
from model import resnet18_cifar
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

def evaluate(save_dir='runs/cls/'):
    os.makedirs(save_dir, exist_ok=True)
    _, test_loader = get_cifar10_loaders()
    model = resnet18_cifar().to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir,'resnet18_best.pth')))
    model.eval()

    y_true, y_pred = [], []
    correct_imgs, incorrect_imgs = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            for i in range(len(targets)):
                img = inputs[i].cpu()
                if preds[i]==targets[i]:
                    correct_imgs.append(img)
                else:
                    incorrect_imgs.append(img)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='viridis')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(save_dir,'confusion_matrix.png'))

    # Prediction grids
    correct_grid = make_grid(correct_imgs[:25], nrow=5, normalize=True)
    save_image(correct_grid, os.path.join(save_dir,'preds_grid.png'))

    incorrect_grid = make_grid(incorrect_imgs[:25], nrow=5, normalize=True)
    save_image(incorrect_grid, os.path.join(save_dir,'miscls_grid.png'))

if __name__ == "__main__":
    evaluate()
