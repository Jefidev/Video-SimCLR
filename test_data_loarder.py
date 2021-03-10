from datasets.simclr_video_loader import SimCLRLoader
import pandas as pd

from torchvision import datasets, transforms
import torch
import cv2
import numpy as np

from transforms.video_transforms import (
    ChangeVideoShape,
    ResizeVideo,
    RandomCropVideo,
    CenterCropVideo,
    I3DPixelsValue,
    RandomTrimVideo,
    TrimVideo,
    PadVideo,
    RandomColorJitterVideo,
)

def write_video(video, path):
    name = path
    fourc = cv2.VideoWriter_fourcc(*"DIVX")

    s = (video.shape[2], video.shape[1])
    out = cv2.VideoWriter(name, fourc, 25.0, s)

    for frame in video:
        image = frame * 255
        r, g, b = cv2.split(np.float32(image))
        image = cv2.merge([b, g, r])
        out.write(image.astype(np.uint8))
    out.release()


batch_size = 3
cumulation = 86  # accum gradient
nbr_frames = 48
nb_workers = 3

prefix = "/run/media/jeromefink/2.0 TB Hard/RAW-LSFB/videos"
path = "/run/media/jeromefink/2.0 TB Hard/RAW-LSFB/simclr_data.csv"

data = pd.read_csv(path)

# Transforms
vid1_transforms = transforms.Compose(
    [
        PadVideo(nbr_frames),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
    ]
)

# Add random color drop and jitter
vid2_transforms = transforms.Compose(
    [
        PadVideo(nbr_frames),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        RandomColorJitterVideo(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ]
)

simclr = SimCLRLoader(data, nbr_frames, vid1_transforms, vid2_transforms, prefix)
print(len(data))

# Dataloader

dataloader = torch.utils.data.DataLoader(
    simclr, batch_size=1, shuffle=True
)

cpt = 1
for data in dataloader:

    vid1 = data[0][0]
    vid2 = data[1][0]

    p1 = f"./show_transfo/{cpt}_1.avi"
    p2 = f"./show_transfo/{cpt}_2.avi"

    write_video(vid1, p1)
    write_video(vid2, p2)

    cpt += 1
