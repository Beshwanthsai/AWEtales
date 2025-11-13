"""
Post Processing Module
Handles punctuation restoration and final output formatting.
"""

import os
import numpy as np
from typing import List, Dict, Any
import soundfile as sf
from deepmultilingualpunctuation import PunctuationModel

class PostProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.punct_model = PunctuationModel()

    def process(self, transcripts: List[Dict], target_audio: np.ndarray) -> Dict[str, Any]:
        # Punctuation restoration
        processed_transcripts = self.restore_punctuation(transcripts)

        # Save target audio
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, "target_speaker.wav")
        sf.write(target_path, target_audio, self.config.get("sample_rate", 16000))

        return {
            "transcripts": processed_transcripts,
            "target_audio_path": target_path
        }

    def restore_punctuation(self, transcripts: List[Dict]) -> List[Dict]:
        for t in transcripts:
            t["text"] = self.punct_model.restore_punctuation(t["text"])
        return transcripts