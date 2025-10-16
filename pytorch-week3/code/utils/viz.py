import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

def plot_curves(losses, save_dir):
    plt.figure()
    plt.plot(losses, label='Train Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'curves_mt.png'))

def plot_attention(attention, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(attention.detach().cpu().numpy(), cmap='viridis')
    plt.savefig(save_path)
