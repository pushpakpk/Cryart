import torch
import torch.nn.functional as F
import numpy as np
import cv2
from model import resnet18_cifar
from dataset import get_cifar10_loaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activations(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(save_activations)
        target_layer.register_backward_hook(save_gradients)

    def __call__(self, x, class_idx=None):
        x = x.unsqueeze(0).to(device)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        loss = out[0,class_idx]
        self.model.zero_grad()
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0,2,3])
        activations = self.activations[0]
        for i in range(len(pooled_grads)):
            activations[i] *= pooled_grads[i]
        heatmap = torch.sum(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        return heatmap

def save_gradcam_images(save_dir='runs/cls/'):
    os.makedirs(save_dir, exist_ok=True)
    _, test_loader = get_cifar10_loaders(batch_size=1)
    model = resnet18_cifar().to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir,'resnet18_best.pth')))
    model.eval()

    target_layer = model.layer4[1].conv2  # last conv layer
    gradcam = GradCAM(model, target_layer)

    for i, (img, label) in enumerate(test_loader):
        if i>=5: break  # generate 5 sample images
        heatmap = gradcam(img[0], class_idx=label.item())
        heatmap = cv2.resize(heatmap, (32,32))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        orig = np.transpose((img[0].numpy()*255).astype(np.uint8), (1,2,0))
        cam = heatmap*0.4 + orig
        cv2.imwrite(os.path.join(save_dir,f'gradcam_{i}.png'), cam)

if __name__ == "__main__":
    save_gradcam_images()
