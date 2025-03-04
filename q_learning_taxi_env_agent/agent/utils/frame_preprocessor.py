from collections import deque

import cv2
import numpy as np
import torch

from .compute_device import ComputeDevice


class FramePreprocessor:
    def __init__(self, stack_size: int = 4):
        self.device = ComputeDevice.get_device()
        self.stack_size = stack_size
        self.frames = deque(maxlen = self.stack_size)

    def preprocess_frames(self, frames):
        preprocessed_frames = []
        for frame in frames:
            frame = self.preprocess_frame(frame)
            preprocessed_frames.append(frame)

        return preprocessed_frames


    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0

        return self.stack_frame(frame)

    def stack_frame(self, frame):
        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        return np.stack(self.frames, axis=0)


