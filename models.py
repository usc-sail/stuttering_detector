import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class StutteringDetector(nn.Module):
    def __init__(
        self,
        wavlm,
        modality,
        hidden_size=128,
        num_classes=3,
        freeze_wavlm=True,
    ):
        super().__init__()
        self.modality = modality
        self.wavlm = wavlm

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False

        # WavLM-Base-Plus has hidden_size = 768
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=13, stride=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1521, 1024)

        self.conv2 = nn.Conv1d(1024, 64, kernel_size=15, stride=1, padding=7)
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.pool1 = nn.MaxPool1d(2)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        lstm_input = 256
        if self.modality == "video":
            lstm_input = 64
        if self.modality == "both":
            lstm_input = 320
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, audio, video, attention_mask=None):
        if self.modality == "audio":
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.wavlm(
                    input_values=audio,
                    attention_mask=attention_mask,
                )
            
            x = outputs.last_hidden_state #[B, T, 768]
            
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                size=150,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
    
            x = self.projection(x) #[B, T, 256]
    
            x, _ = self.lstm(x) #[B, T, 256]
    
            x = self.dropout(x)
    
            logits = self.classifier(x) #[B, T, 3]
    
            return logits
        if self.modality == "video":
            b = video.shape[0]
            t = video.shape[1]
            x = rearrange(video, 'b t c h w -> (b t) c h w')
            x = F.relu(self.conv1(x))
            x = self.batchnorm1(x) 
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))

            x = rearrange(x, '(b t) e -> b t e', b=b, t=t) 
            x = rearrange(x, 'b t e -> b e t')
            x = F.relu(self.conv2(x))
            x = self.batchnorm2(x)
            x = rearrange(x, 'b e t -> b t e')
    
            x, _ = self.lstm(x) #[B, T, 256]
    
            x = self.dropout(x)
    
            logits = self.classifier(x) #[B, T, 3]
    
            return logits
        if self.modality == "both":
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.wavlm(
                    input_values=audio,
                    attention_mask=attention_mask,
                )
            
            x1 = outputs.last_hidden_state #[B, T, 768]
            
            x1 = torch.nn.functional.interpolate(
                x1.transpose(1, 2),
                size=150,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
    
            x1 = self.projection(x1) #[B, T, 256]

            b = video.shape[0]
            t = video.shape[1]
            x2 = rearrange(video, 'b t c h w -> (b t) c h w')
            x2 = F.relu(self.conv1(x2))
            x2 = self.batchnorm1(x2) 
            x2 = torch.flatten(x2, start_dim=1)
            x2 = F.relu(self.fc1(x2))

            x2 = rearrange(x2, '(b t) e -> b t e', b=b, t=t) 
            x2 = rearrange(x2, 'b t e -> b e t')
            x2 = F.relu(self.conv2(x2))
            x2 = self.batchnorm2(x2)
            x2 = rearrange(x2, 'b e t -> b t e')
            x = torch.cat((x1, x2), 2)
    
            x, _ = self.lstm(x) #[B, T, 256]
    
            x = self.dropout(x)
    
            logits = self.classifier(x) #[B, T, 3]
            return logits