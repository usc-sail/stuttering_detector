import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
import numpy as np
import cv2
import librosa
from einops import rearrange
import glob
import textgrid
from transformers import AutoProcessor, AutoModel
import noisereduce as nr
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from datasets import BinaryMultimodalStutteringDataset
from models import StutteringDetector
from utils import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16)
parser.add_argument("--epochs", default=30)
parser.add_argument("--articulator", default="LA")
args = parser.parse_args()

batch_size = int(args.batch_size)
epochs = args.epochs

la = {
  3: [61,74,25,128],
  4: [56,74,29,128],
  5: [56,78,27,128],
  6: [56,78,37,128],
  7: [56,78,34,128],
  8: [56,82,24,128],
  10: [56,82,33,128]
}
tt = {
  3: [57,74,22,33],
  4: [62,78,31,38],
  5: [64,82,26,38],
  6: [62,82,36,45],
  7: [64,82,29,38],
  8: [62,78,24,34],
  10: [62,78,34,42]
}
tb = {
  3: [55,70,34,44],
  4: [60,74,41,52],
  5: [62,78,41,52],
  6: [60,78,45,58],
  7: [62,78,45,57],
  8: [59,78,35,48],
  10: [59,75,43,55]
}
vl = {
  3: [52,72,53,66],
  4: [57,83,64,74],
  5: [57,81,60,74],
  6: [57,78,64,77],
  7: [57,79,60,77],
  8: [57,84,60,77],
  10: [57,78,65,77]
}
tr = {
  3: [76,92,50,65],
  4: [85,102,62,76],
  5: [83,102,58,76],
  6: [81,102,58,80],
  7: [79,98,58,80],
  8: [81,98,58,80],
  10: [81,96,58,80]
}
lx = {
  3: [100,128,48,64],
  4: [105,128,53,73],
  5: [102,128,53,73],
  6: [102,128,53,76],
  7: [102,128,57,79],
  8: [102,128,54,79],
  10: [98,128,54,79]
}

patch = 0
if args.articulator == "LA":
    patch = la
elif args.articulator == "TT":
    patch = tt
elif args.articulator == "TB":
    patch = tb
elif args.articulator == "VL":
    patch = vl
elif args.articulator == "TR":
    patch = tr
elif args.articulator == "LX":
    patch = lx


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Audio encoder
MODEL_ID = "microsoft/wavlm-base-plus"
wavlm = AutoModel.from_pretrained(MODEL_ID)

#My main model
model = StutteringDetector(wavlm, "video", num_classes=2).to(device)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)

counts = np.array([
    392205,
    220214
], dtype=np.float64)

weights = counts.sum() / (2 * counts)
class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

#Load training and testing datasets
train_set = torch.load('binary_train_set.pt', weights_only=False)
test_set = torch.load('binary_test_set.pt', weights_only=False)

trainloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
testloader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

for epoch in range(epochs):
    print("Epoch", epoch)
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    for index, batch in enumerate(trainloader):

        audio = batch["audio"].to(device) # [B, T]
        video = batch["video"][:,:,0:1,:,:] # [B, T, 1, H, W]
        subject = batch["video"][:,0,1,0,0].int()
        for sample in range(subject.shape[0]):
            video[sample, :, :, patch[subject[sample].item()][0]:patch[subject[sample].item()][1],patch[subject[sample].item()][2]:patch[subject[sample].item()][3]] = 0
        video = video.to(device)
        optimizer.zero_grad()
        logits = model(audio, video) # [B, 150, 2]

        predictions = torch.argmax(logits, dim=-1)

        target_labels = []

        for label_seq in batch["labels"]:
            label_seq = label_seq.to(device)
            target_labels.append(label_seq)

        target_labels = torch.stack(target_labels)
        all_predictions.append(
            predictions.cpu().numpy().reshape(-1)
        )

        all_targets.append(
            target_labels.cpu().numpy().reshape(-1)
        ) 
        loss = criterion(
            logits.reshape(-1, 2),
            target_labels.reshape(-1)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        running_loss += loss.item()

        if index % 15 == 0:
            loss, current, size = loss.item(), index * batch_size + len(audio), len(trainloader.dataset)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(
        f"Training loss = {running_loss / len(trainloader):.4f}"
    )
    
    running_loss = 0.0
    model.eval()
    all_predictions = []
    all_targets = []
    for index, batch in enumerate(testloader):
        with torch.no_grad():
            audio = batch["audio"].to(device) # [B, T]
            video = batch["video"][:,:,0:1,:,:] # [B, T, 1, H, W]
            subject = batch["video"][:,0,1,0,0].int()
            for sample in range(subject.shape[0]):
                video[sample, :, :, patch[subject[sample].item()][0]:patch[subject[sample].item()][1],patch[subject[sample].item()][2]:patch[subject[sample].item()][3]] = 0
            video = video.to(device)

            logits = model(audio, video) # [B, 150, 2]
            predictions = torch.argmax(logits, dim=-1)
        target_labels = []

        for label_seq in batch["labels"]:
            label_seq = label_seq.to(device)

            target_labels.append(label_seq)
        target_labels = torch.stack(target_labels)
        loss = criterion(
            logits.reshape(-1, 2),
            target_labels.reshape(-1)
        )
        running_loss += loss.item()
        all_predictions.append(
            predictions.cpu().numpy().reshape(-1)
        )

        all_targets.append(
            target_labels.cpu().numpy().reshape(-1)
        ) 
    print(
        f"Testing loss = {running_loss / len(testloader):.4f}"
    )
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[0, 1]
    )
    print(cm)
    print(classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1],
        target_names=["No disfluency", "Disfluency"],
        digits=3
    ))
