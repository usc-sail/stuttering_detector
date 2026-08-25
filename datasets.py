import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class MultimodalStutteringDataset(Dataset):
    def __init__(
        self,
        audios,
        videos,
        labels,
        audio_sr=16000,
        video_fps=50,
        window_sec=3.0,
        stride_sec=3.0,
        drop_incomplete=True,
    ):
        assert len(audios) == len(videos) == len(labels)

        self.audio_sr = audio_sr
        self.video_fps = video_fps

        self.window_audio_samples = int(
            window_sec * audio_sr
        )

        self.window_video_frames = int(
            window_sec * video_fps
        )

        self.stride_audio_samples = int(
            stride_sec * audio_sr
        )

        self.stride_video_frames = int(
            stride_sec * video_fps
        )

        self.windows = []

        for audio, video, label_seq in zip(
            audios, videos, labels
        ):
            audio = np.asarray(audio, dtype=np.float32)

            # video: [T, 1, H, W]
            video = torch.as_tensor(video)

            # labels: [T]
            label_seq = np.asarray(
                label_seq,
                dtype=np.int64
            )

            # Number of complete 20-ms / 50-Hz frames
            if video.shape[0] > len(label_seq):
                video = video[:len(label_seq),:,:,:]
            elif video.shape[0] < len(label_seq):
                label_seq = label_seq[:video.shape[0]]
            
            n_video_frames = video.shape[0]
            n_labels = len(label_seq)

            if n_video_frames != n_labels:
                raise ValueError(
                    f"Video has {n_video_frames} frames "
                    f"but labels have {n_labels} entries."
                )
            if audio.shape[0]/audio_sr > video.shape[0]/video_fps:
                audio = audio[:int(video.shape[0]/video_fps*audio_sr)]

            # Audio duration expressed in 50-Hz frames
            expected_audio_frames = len(audio) / (
                audio_sr / video_fps
            )

            if abs(expected_audio_frames - n_video_frames) > 1.0:
                raise ValueError(
                    f"Audio/video duration mismatch: "
                    f"audio={len(audio)/audio_sr:.3f}s, "
                    f"video={n_video_frames/video_fps:.3f}s"
                )

            start_video = 0
            start_audio = 0

            while True:

                end_video = (
                    start_video +
                    self.window_video_frames
                )

                end_audio = (
                    start_audio +
                    self.window_audio_samples
                )

                # Check whether this is a complete window
                if (
                    end_video > n_video_frames
                    or end_audio > len(audio)
                ):
                    if drop_incomplete:
                        break

                    # Otherwise, don't include partial windows
                    break

                video_window = video[
                    start_video:end_video
                ]

                audio_window = audio[
                    start_audio:end_audio
                ]

                label_window = label_seq[
                    start_video:end_video
                ]

                self.windows.append({
                    "audio": audio_window,
                    "video": video_window,
                    "labels": label_window,
                })

                start_video += self.stride_video_frames
                start_audio += self.stride_audio_samples

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]

        return {
            "audio": torch.tensor(
                item["audio"],
                dtype=torch.float32
            ),

            # [50, 1, H, W]
            "video": item["video"].float(),

            # [50]
            "labels": torch.tensor(
                item["labels"],
                dtype=torch.long
            )
        }