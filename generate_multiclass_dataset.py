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

print("Loading dataset... Please be patient!")
audios = []
videos = []
labels = []

root = '/data1/span_data/stuttering/'
subjects = ["PWS3", "PWS4", "PWS5", "PWS6", "PWS7", "PWS8", "PWS10"]

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
                if tier.name == "disfluency":
                    timelabels = []
                    timestep = 0
                    while timestep <= tier.maxTime:
                        for palabra in tier:
                            if timestep >= palabra.minTime and timestep <= palabra.maxTime:
                                if palabra.mark == "":
                                    timelabels.append(0)
                                elif "pro" in palabra.mark and "rep" in palabra.mark:
                                    timelabels.append(2)
                                elif "pro" in palabra.mark:
                                    timelabels.append(1)
                                elif "rep" in palabra.mark:
                                    timelabels.append(2)
                                else:
                                    timelabels.append(3)
                        timestep += 0.02
                    if 1 in timelabels or 2 in timelabels or 3 in timelabels:
                        labels.append(timelabels)
                        videos.append(video)
                        audios.append(wav)
        except:
            continue

combinadas = list(zip(audios, videos, labels))
random.shuffle(combinadas)
audios, videos, labels = map(list, zip(*combinadas))

training_audios = audios[:334]
training_videos = videos[:334]
training_labels = labels[:334]
testing_audios = audios[334:]
testing_videos = videos[334:]
testing_labels = labels[334:]

counts = Counter()
for label_seq in training_labels:
    counts.update(label_seq)

#Load training and testing datasets
train_set = MulticlassMultimodalStutteringDataset(
    training_audios,
    training_videos,
    training_labels,
    window_sec=3.0,
    stride_sec=3.0,
)
test_set = MulticlassMultimodalStutteringDataset(
    testing_audios,
    testing_videos,
    testing_labels,
    window_sec=3.0,
    stride_sec=3.0,
)

torch.save(train_set, 'multiclass_train_set.pt')
torch.save(test_set, 'multiclass_test_set.pt')