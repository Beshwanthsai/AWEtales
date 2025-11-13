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

    def diarize(self, audio_path: str) -> List[Dict]:
        # Use PyAnnote with file path
        diarization = self.pipeline(audio_path)

        # Convert to list of dicts
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        return segments