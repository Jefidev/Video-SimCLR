from torch.utils.data import Dataset
from typing import Tuple, Dict
import torch
import cv2
import numpy as np
import random


class SimCLRLoader(Dataset):
  def __init__(self, data, nb_frame, transforms_vid1, transforms_vid2, path_prefix=""):
    """
    Expected data format : list of video ?
    """

    self.data = data
    self.nb_frame = nb_frame
    self.prefix = path_prefix
    self.transforms_vid1 = transforms_vid1
    self.transforms_vid2 = transforms_vid2


    self.mapping = self.get_idx_mapping()
  

  def __len__(self):
    return len(self.mapping)

  def __getitem__(self, idx):

    vid_info = self.mapping[idx]
    corrected_idx = idx - vid_info["offset"]
    
    start_frame = corrected_idx * (self.nb_frame // 2)
    end_frame = start_frame + self.nb_frame

    vid1 = self.load_clip(vid_info["path"], start_frame, end_frame)
    vid2 = vid1.copy()

    vid1 = self.transforms_vid1(vid1)
    vid2 = self.transforms_vid2(vid2)


    return vid1, vid2


  def get_idx_mapping(self):
    steps = self.nb_frame // 2
    mapping = {}
    cpt = 0
    offset = 0

    for idx, vid_info in self.data.iterrows():
      path = vid_info["path"]
      frames = vid_info["nb_frames"]

      vid_path = f"{self.prefix}/{path}"

      nbr_clips = (frames // steps) - 1

      for idx in range(nbr_clips):
        mapping[cpt] = {
          "path": vid_path,
          "offset": offset
        }

        cpt += 1

      offset += nbr_clips

    return mapping


  def load_clip(self, path, start, end):

    frame_array = []

    capture = cv2.VideoCapture(path)
    # Begin at starting frame
    capture.set(cv2.CAP_PROP_POS_FRAMES,start)

    success, frame = capture.read()

    while success:
      b, g, r = cv2.split(frame)
      frame = cv2.merge([r, g, b])
      frame_array.append(frame / 255)

      success, frame = capture.read()
      
      frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
      
      if frame_number >= end:
        success = False


    return np.array(frame_array)
    