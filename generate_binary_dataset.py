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
import argparse

from datasets import BinaryMultimodalStutteringDataset
from models import StutteringDetector
from utils import collate_fn

print("Loading dataset... Please be patient!")
audios = []
videos = []
labels = []

root = '/data1/span_data/stuttering/'
subjects = ["PWS3", "PWS4", "PWS5", "PWS6", "PWS7", "PWS8", "PWS10"]

la = {
  "PWS3": [61,74,25,128],
  "PWS4": [56,74,29,128],
  "PWS5": [56,78,27,128],
  "PWS6": [56,78,37,128],
  "PWS7": [56,78,34,128],
  "PWS8": [56,82,24,128],
  "PWS10": [56,82,33,128]
}
tt = {
  "PWS3": [57,74,22,33],
  "PWS4": [62,78,31,38],
  "PWS5": [64,82,26,38],
  "PWS6": [62,82,36,45],
  "PWS7": [64,82,29,38],
  "PWS8": [62,78,24,34],
  "PWS10": [62,78,34,42]
}
tb = {
  "PWS3": [55,70,34,44],
  "PWS4": [60,74,41,52],
  "PWS5": [62,78,41,52],
  "PWS6": [60,78,45,58],
  "PWS7": [62,78,45,57],
  "PWS8": [59,78,35,48],
  "PWS10": [59,75,43,55]
}
vl = {
  "PWS3": [52,72,53,66],
  "PWS4": [57,83,64,74],
  "PWS5": [57,81,60,74],
  "PWS6": [57,78,64,77],
  "PWS7": [57,79,60,77],
  "PWS8": [57,84,60,77],
  "PWS10": [57,78,65,77]
}
tr = {
  "PWS3": [76,92,50,65],
  "PWS4": [85,102,62,76],
  "PWS5": [83,102,58,76],
  "PWS6": [81,102,58,80],
  "PWS7": [79,98,58,80],
  "PWS8": [81,98,58,80],
  "PWS10": [81,96,58,80]
}
lx = {
  "PWS3": [100,128,48,64],
  "PWS4": [105,128,53,73],
  "PWS5": [102,128,53,73],
  "PWS6": [102,128,53,76],
  "PWS7": [102,128,57,79],
  "PWS8": [102,128,54,79],
  "PWS10": [98,128,54,79]
}

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
                    frame = cv2.resize(frame, (128, 128))[:,:,0:2]
                    if subject == "PWS3":
                        frame[:,:,1] = 3
                    elif subject == "PWS4":
                        frame[:,:,1] = 4
                    elif subject == "PWS5":
                        frame[:,:,1] = 5
                    elif subject == "PWS6":
                        frame[:,:,1] = 6
                    elif subject == "PWS7":
                        frame[:,:,1] = 7
                    elif subject == "PWS8":
                        frame[:,:,1] = 8
                    elif subject == "PWS10":
                        frame[:,:,1] = 10
                    frames.append(frame)
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
                                else:
                                    timelabels.append(1)
                        timestep += 0.02
                    if 1 in timelabels and 0 in timelabels:
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
print(counts[0])
print(counts[1])

#Load training and testing datasets
train_set = BinaryMultimodalStutteringDataset(
    training_audios,
    training_videos,
    training_labels,
    window_sec=3.0,
    stride_sec=3.0,
)
test_set = BinaryMultimodalStutteringDataset(
    testing_audios,
    testing_videos,
    testing_labels,
    window_sec=3.0,
    stride_sec=3.0,
)

torch.save(train_set, 'binary_train_set.pt')
torch.save(test_set, 'binary_test_set.pt')