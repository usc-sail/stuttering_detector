import torch
import numpy as np

def collate_fn(batch):
    # --------------------------------------------------
    # Audio
    # --------------------------------------------------
    max_audio_len = max(
        item["audio"].shape[0]
        for item in batch
    )

    audio_batch = []

    # --------------------------------------------------
    # Video
    # --------------------------------------------------
    max_video_len = max(
        item["video"].shape[0]
        for item in batch
    )

    # Assume all videos have the same H and W
    H = batch[0]["video"].shape[-2]
    W = batch[0]["video"].shape[-1]

    video_batch = []

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------
    max_label_len = max(
        item["labels"].shape[0]
        for item in batch
    )

    label_batch = []

    for item in batch:

        # ---------- Audio ----------
        audio = item["audio"]

        if audio.shape[0] < max_audio_len:
            audio = F.pad(
                audio,
                (0, max_audio_len - audio.shape[0])
            )

        audio_batch.append(audio)

        # ---------- Video ----------
        video = item["video"]
        # [T, 1, H, W]

        if video.shape[0] < max_video_len:
            padding_frames = torch.zeros(
                (
                    max_video_len - video.shape[0],
                    1,
                    H,
                    W
                ),
                dtype=video.dtype
            )

            video = torch.cat(
                [video, padding_frames],
                dim=0
            )

        video_batch.append(video)

        # ---------- Labels ----------
        labels = item["labels"]

        if labels.shape[0] < max_label_len:
            # Use -100 so padded labels can be ignored
            # by CrossEntropyLoss(ignore_index=-100)
            padding_labels = torch.full(
                (
                    max_label_len - labels.shape[0],
                ),
                -100,
                dtype=labels.dtype
            )

            labels = torch.cat(
                [labels, padding_labels],
                dim=0
            )

        label_batch.append(labels)

    return {
        "audio": torch.stack(audio_batch),
        "video": torch.stack(video_batch),
        "labels": torch.stack(label_batch),
    }