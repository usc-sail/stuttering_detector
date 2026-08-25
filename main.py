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

from datasets import MultimodalStutteringDataset
from models import StutteringDetector
from utils import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--modality", default="video")
parser.add_argument("--batch_size", default=16)
parser.add_argument("--epochs", default=30)
args = parser.parse_args()

batch_size = args.batch_size
modality = args.modality
epochs = args.epochs

print("Loading dataset... Please be patient!")
audios = []
videos = []
labels = []

root = '/data1/span_data/stuttering/'
subjects = ["PWS3", "PWS4", "PWS6", "PWS8"]

for subject in subjects:
    textgrids = sorted(glob.glob(os.path.join(root, subject, "textgrid", "*.TextGrid")))
    for index in range(len(textgrids)):
        tg = textgrid.TextGrid.fromFile(textgrids[index])
        try:
            wav, sr = librosa.load(os.path.join(root, subject, "wav_denoised", textgrids[index].rsplit('/', 1)[-1][:-9] + ".wav"), sr=16000)
            cap = cv2.VideoCapture(os.path.join(root, subject, "avi_resampled", textgrids[index].rsplit('/', 1)[-1][:-9] + ".avi"))
            frames = []
            counter = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if counter % 2 == 0:
                    frames.append(cv2.resize(frame, (128, 128))[:,:,0:1])
                counter += 1
            video = torch.tensor(np.array(frames), dtype=torch.float32)
            video = rearrange(video, 't h w c -> t c h w')  # Rearrange to (T, C, H, W)
            cap.release()
            for tier in tg:
                if tier.name == "words":
                    timelabels = []
                    timestep = 0
                    while timestep <= tier.maxTime:
                        for palabra in tier:
                            if timestep >= palabra.minTime and timestep <= palabra.maxTime:
                                if "flue" in palabra.mark:
                                    timelabels.append(1)
                                elif "disf" in palabra.mark:
                                    timelabels.append(2)
                                else:
                                    timelabels.append(0)
                        timestep += 0.02
                    if 1 in timelabels or 2 in timelabels:
                        labels.append(timelabels)
                        videos.append(video)
                        audios.append(wav)
        except:
            continue
        

combinadas = list(zip(audios, videos, labels))
random.shuffle(combinadas)
audios, videos, labels = map(list, zip(*combinadas))

training_audios = audios[:85]
training_videos = videos[:85]
training_labels = labels[:85]
testing_audios = audios[85:]
testing_videos = videos[85:]
testing_labels = labels[85:]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_ID = "microsoft/wavlm-base-plus"
wavlm = AutoModel.from_pretrained(MODEL_ID)
model = StutteringDetector(wavlm, modality).to(device)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-2,
)

counts = Counter()

for label_seq in training_labels:
    counts.update(label_seq)

counts = np.array([
    counts[0],
    counts[1],
    counts[2]
], dtype=np.float64)

weights = counts.sum() / (3 * counts)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

train_set = MultimodalStutteringDataset(
    training_audios,
    training_videos,
    training_labels,
    window_sec=3.0,
    stride_sec=3.0,
)

test_set = MultimodalStutteringDataset(
    testing_audios,
    testing_videos,
    testing_labels,
    window_sec=3.0,
    stride_sec=3.0,
)

batch_size = 16

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
        logits = model(audio, video) # [B, 150, 3]

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
            logits.reshape(-1, 3),
            target_labels.reshape(-1)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        running_loss += loss.item()

        if index % 5 == 0:
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
            
            logits = model(audio, video) # [B, 150, 3]
            predictions = torch.argmax(logits, dim=-1)
        target_labels = []

        for label_seq in batch["labels"]:
            label_seq = label_seq.to(device)

            target_labels.append(label_seq)
        target_labels = torch.stack(target_labels)
        loss = criterion(
            logits.reshape(-1, 3),
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
        labels=[0, 1, 2]
    )
    print(cm)
    print(classification_report(
        all_targets,
        all_predictions,
        labels=[0, 1, 2],
        target_names=["Silence", "Fluent", "Disfluent"],
        digits=3
    ))
