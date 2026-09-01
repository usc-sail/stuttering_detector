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

from datasets import MulticlassMultimodalStutteringDataset
from models import StutteringDetector
from utils import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--modality", default="video")
parser.add_argument("--batch_size", default=16)
parser.add_argument("--epochs", default=30)
args = parser.parse_args()

batch_size = int(args.batch_size)
modality = args.modality
epochs = args.epochs

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Audio encoder
MODEL_ID = "microsoft/wavlm-base-plus"
wavlm = AutoModel.from_pretrained(MODEL_ID)

#My main model
model = StutteringDetector(wavlm, modality, num_classes=4).to(device)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)

counts = np.array([
    388266,
    54498,
    101508,
    70875
], dtype=np.float64)

weights = counts.sum() / (4 * counts)
class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

#Load training and testing datasets
train_set = torch.load('multiclass_train_set.pt', weights_only=False)
test_set = torch.load('multiclass_test_set.pt', weights_only=False)

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
        video = batch["video"].to(device) # [B, T, 1, H, W]

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
            logits.reshape(-1, 4),
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
            video = batch["video"].to(device) # [B, T, 1, H, W]
            logits = model(audio, video) # [B, 150, 2]
            predictions = torch.argmax(logits, dim=-1)
        target_labels = []

        for label_seq in batch["labels"]:
            label_seq = label_seq.to(device)

            target_labels.append(label_seq)
        target_labels = torch.stack(target_labels)
        loss = criterion(
            logits.reshape(-1, 4),
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
        labels=[0, 1, 2, 3]
    )
    print(cm)
    print(classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=["No disfluency", "Prolongation", "Repetition", "Complex"],
        digits=3
    ))
