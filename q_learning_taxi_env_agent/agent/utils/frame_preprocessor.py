import cv2
import numpy as np
import torch

from .compute_device import ComputeDevice


class FramePreprocessor:
    def __init__(self):
        self.device = ComputeDevice.get_device()

    def preprocess(self, frame):
        frame = cv2.resize(frame, (128,128), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))

        frame = torch.tensor(frame, dtype=torch.float32, device=self.device).unsqueeze(0)
        return frame