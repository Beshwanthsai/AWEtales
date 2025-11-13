"""
Diarization Module
Handles speaker diarization using PyAnnote.
"""

import numpy as np
from pyannote.audio import Pipeline
import torch
from typing import List, Dict

class Diarizer:
    def __init__(self, config: dict):
        self.config = config
        # Initialize PyAnnote pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                 use_auth_token=config.get("hf_token"))

    def diarize(self, audio: np.ndarray) -> List[Dict]:
        # Placeholder: Convert numpy array to pyannote format
        # PyAnnote expects file path or Audio object
        # For now, assume audio is processed
        # This needs proper implementation
        diarization = self.pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": self.config.get("sample_rate", 16000)})

        # Convert to list of dicts
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        return segments