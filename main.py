import argparse
from pytorch_metric_learning.losses import NTXentLoss
import torch

from models.pytorch_i3d_checkpointed import InceptionI3d
from datasets.simclr_video_loader import SimCLRLoader
from models.simclr import SimCLR
from modules.lars import LARS
import pandas as pd
from torchvision import datasets, transforms
from torch.autograd import Variable
import mlflow

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

# Thanks to : https://github.com/Spijkervet/SimCLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

KINETIC_PATH = "./checkpoints/rgb_imagenet.pt"
PREFIX = "/run/media/jeromefink/2.0 TB Hard/RAW-LSFB/videos"

batch_size = 12
cumulation = 170  # accum gradient
nbr_frames = 48
nb_workers = 2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to the input video directory")
#parser.add_argument("-o", "--output", help="Path to the output directory")
parser.add_argument("-l", "--load", help="Indicate to load model weight")
parser.add_argument("-p", "--projection", help="Projection dimension of the projection head", type=int)
parser.add_argument("-f", "--features", help="Number of features of the last layer of the encoder", type=int)
parser.add_argument("-a", "--all", action="store_true", help="If present tells the program to load the whole simclr model from saved file")

args = parser.parse_args()

model_weights = args.load
input = args.input
n_features = args.features
n_projection = args.projection
all = args.all

# Saving mlflow params

params_ml_flow = {
    "batch_size": batch_size,
    "frames": nbr_frames,
    "input": input,
    "temperature": 0.1
}

# Prepare data loader
data = pd.read_csv(input)

# Transforms
vid1_transforms = transforms.Compose(
    [
        PadVideo(nbr_frames),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        I3DPixelsValue(),
        ChangeVideoShape("CTHW"),
    ]
)

# Add random color drop and jitter
vid2_transforms = transforms.Compose(
    [
        PadVideo(nbr_frames),
        ResizeVideo(270, interpolation="linear"),
        RandomCropVideo((224, 224)),
        RandomColorJitterVideo(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        I3DPixelsValue(),
        ChangeVideoShape("CTHW"),
    ]
)

simclr = SimCLRLoader(data, nbr_frames, vid1_transforms, vid2_transforms, PREFIX)

# Dataloader

dataloader = torch.utils.data.DataLoader(
    simclr, batch_size=batch_size, shuffle=True, num_workers=nb_workers
)


# Load pre-trained I3D

if model_weights == None:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(KINETIC_PATH))
    i3d.replace_logits(n_features)
    print("RGB kinetic loaded")

    model = SimCLR(i3d, n_projection, n_features)
elif all:
    i3d = InceptionI3d(400, in_channels=3)
    model = SimCLR(i3d, n_projection, n_features)
    model.load_state_dict(model_weights)
    print("Loading the whole SimCLR model")
else:
    i3d = InceptionI3d(n_features, in_channels=3)
    i3d.load_state_dict(torch.load(model_weights))
    print("Previous weights loaded")

    model = SimCLR(i3d, n_projection, n_features)

# Construct SimCLR model

model.to(device)

# Loss function
criterion = NTXentLoss(temperature=0.10)

# LARS optimizer

learning_rate = 0.3 * (batch_size * cumulation) / 256
optimizer = LARS(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.0005,
    exclude_from_weight_decay=["batch_normalization", "bias"],
    device=device
)

# "decay the learning rate with the cosine decay schedule without restarts"
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 100, eta_min=0, last_epoch=-1
)


mlflow.set_experiment("SimCLR")

with mlflow.start_run(run_name="SimCLR test") as run:
    mlflow.log_params(params_ml_flow)
    loss_array = []

    # Training 
    for epoch in range(100):
        print(f"EPOCH - {epoch+1}")
        cpt = 1
        epoch_size = len(dataloader)

        model.train()

        for xi, xj in dataloader:

            xi = xi.type(torch.FloatTensor).to(device)
            xj = xj.type(torch.FloatTensor).to(device)
            xi.requires_grad = True
            xj.requires_grad = True

            zi, zj = model(xi, xj)

            embeddings = torch.cat((zi, zj))
            indices = torch.arange(0, zi.size(0), device=zi.device)
            labels = torch.cat((indices, indices))

            loss = criterion(embeddings, labels)
            loss.backward()

            # Collecting loss
            print (f"{cpt}/{epoch_size}   --  Loss : {loss.item()}", end="\r")
            loss_array.append(float(loss.item()))

            # Cumulation
            if cpt % cumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Saving the models
                print("saving  \t \t \t", end="\r")
                torch.save(
                    model.state_dict(), "./checkpoints/full_smiclr.pt"
                )

                torch.save(
                    model.encoder.state_dict(), "./checkpoints/i3d_simclr.pt"
                )

                # log mlflow
                avg = sum(loss_array) / len(loss_array)
                mlflow.log_metric("avg_loss", avg)
                loss_array = []


            cpt += 1



