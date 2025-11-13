"""
ASR Module
Handles automatic speech recognition using Whisper.
"""

import whisper
from typing import List, Dict

class ASRModule:
    def __init__(self, config: dict):
        self.config = config
        self.model = whisper.load_model("base")  # Or larger model

    def transcribe_segments(self, audio: np.ndarray, diarization_segments: List[Dict]) -> List[Dict]:
        transcripts = []
        for segment in diarization_segments:
            start = int(segment["start"] * self.config.get("sample_rate", 16000))
            end = int(segment["end"] * self.config.get("sample_rate", 16000))
            segment_audio = audio[start:end]

            # Transcribe
            result = self.model.transcribe(segment_audio, fp16=False)
            text = result["text"]

            transcripts.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": text.strip(),
                "confidence": 0.95  # Placeholder
            })

        return transcripts