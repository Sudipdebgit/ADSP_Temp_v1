#!/usr/bin/python

import numpy as np
import torch
import matplotlib.pyplot as plt

from bss_dataset import STFTDataset
from bss_model import ImprovedBSS


def plot_spec(true_mag, pred_mag, title, eps=1e-8):
    # average channels if stereo
    if true_mag.ndim == 3:
        true_mag = true_mag.mean(axis=0)
    if pred_mag.ndim == 3:
        pred_mag = pred_mag.mean(axis=0)

    true_db = 20 * np.log10(true_mag + eps)
    pred_db = 20 * np.log10(pred_mag + eps)
    diff_db = true_db - pred_db

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    im0 = ax[0].imshow(true_db, origin="lower", aspect="auto")
    ax[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(pred_db, origin="lower", aspect="auto")
    ax[1].set_title("Predicted")
    plt.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(diff_db, origin="lower", aspect="auto", cmap="coolwarm")
    ax[2].set_title("True − Predicted")
    plt.colorbar(im2, ax=ax[2])

    for a in ax:
        a.set_xlabel("Time frames")
        a.set_ylabel("Frequency bins")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # ===== config =====
    checkpoint_path = "checkpoints/best_model.pth"
    meta_path = "dataset_stft/meta.jsonl"
    sample_index = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== load model =====
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ===== load data =====
    dataset = STFTDataset(meta_path, supervised_ratio=1.0, normalize=True)
    sample = dataset[sample_index]

    mix_mag = sample["mix_mag"].numpy()
    s1_mag_true = sample["s1_mag"].numpy()
    s2_mag_true = sample["s2_mag"].numpy()

    # ===== inference =====
    with torch.no_grad():
        mix_mag_batch = torch.from_numpy(mix_mag).unsqueeze(0).float().to(device)
        pred_s1_mag, pred_s2_mag, _ = model(mix_mag_batch)

    pred_s1_mag = pred_s1_mag.squeeze(0).cpu().numpy()
    pred_s2_mag = pred_s2_mag.squeeze(0).cpu().numpy()

    # ===== plot =====
    plot_spec(s1_mag_true, pred_s1_mag, "Source 1: True vs Predicted")
    plot_spec(s2_mag_true, pred_s2_mag, "Source 2: True vs Predicted")


if __name__ == "__main__":
    main()
