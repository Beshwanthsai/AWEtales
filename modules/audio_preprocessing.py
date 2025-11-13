import librosa
import numpy as np
from audio_separator import Separator
from pyannote.audio import Pipeline
import torch
import soundfile as sf
import tempfile
import os

class AudioPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        # Initialize separator for denoising
        self.separator = Separator(model_file_dir="/tmp/audio_separator_models")
        self.separator.load_model("UVR-MDX-NET-Inst_HQ_4")
        # Initialize VAD pipeline
        self.vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                                     use_auth_token=config.get("hf_token"))

    def process(self, audio_path: str) -> np.ndarray:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Denoise using UVR-MDX-Net
        # For denoising, use the separator on the audio
        # But separator expects file, so save temp
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            temp_path = f.name
        try:
            # Separate (denoise by removing non-vocal)
            output_files = self.separator.separate(temp_path)
            # Assume output is denoised vocal
            if output_files:
                denoised_audio, _ = librosa.load(output_files[0], sr=self.sample_rate)
            else:
                denoised_audio = audio
        finally:
            os.unlink(temp_path)
            for f in output_files:
                if os.path.exists(f):
                    os.unlink(f)

        # VAD
        vad_result = self.vad_pipeline({"waveform": torch.tensor(denoised_audio).unsqueeze(0), "sample_rate": self.sample_rate})
        vad_segments = []
        for segment in vad_result.get_timeline():
            vad_segments.append((segment.start, segment.end))

        # For simplicity, return the denoised audio
        return denoised_audio