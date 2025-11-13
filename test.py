"""
Test script to run the unified neural pipeline with dummy audio.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from unified_neural_pipeline import UnifiedNeuralPipeline

def create_dummy_audio(filename, duration=10, sr=16000):
    # Create a simple sine wave as dummy audio
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    sf.write(filename, audio, sr)
    print(f"Created dummy audio: {filename}")

def main():
    # Create dummy files
    mixture_file = "test_mixture.wav"
    target_file = "test_target.wav"
    output_dir = "test_output"

    create_dummy_audio(mixture_file, duration=10)
    create_dummy_audio(target_file, duration=3)

    # Config
    config = {
        "sample_rate": 16000,
        "output_dir": output_dir,
        "hf_token": os.getenv("HF_TOKEN")  # Assume set
    }

    # Run pipeline
    pipeline = UnifiedNeuralPipeline(config)
    result = pipeline.process(mixture_file, target_file)

    print("Pipeline completed!")
    print("Result:", result)

    # Cleanup
    os.unlink(mixture_file)
    os.unlink(target_file)

if __name__ == "__main__":
    main()