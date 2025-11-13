# Unified Neural Pipeline for Target Speaker Identification and Multispeaker ASR

This project implements a unified neural pipeline for target speaker identification and multispeaker automatic speech recognition (ASR), capable of isolating a specific speaker's voice from multi-speaker conversations and providing timestamped transcripts.

## Features

- **Target Speaker Separation**: Isolates the target speaker's voice using a reference clip.
- **Speaker Diarization**: Identifies and segments different speakers in the audio.
- **Automatic Speech Recognition**: Transcribes speech to text with high accuracy.
- **Punctuation Restoration**: Adds punctuation and casing to transcripts.
- **Real-time Streaming Support**: Handles both offline and streaming audio inputs.
- **Web/API Interface**: Provides REST and WebSocket endpoints.

## Architecture

The pipeline consists of the following stages:

1. **Audio Preprocessing**: Endpoint detection, denoising, and voice activity detection.
2. **Speaker Separation**: Isolates the target speaker using advanced neural models.
3. **Diarization**: Segments audio by speaker.
4. **ASR**: Transcribes each segment to text.
5. **Post-processing**: Restores punctuation and formats output.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AWEtales
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Hugging Face token for PyAnnote (if required):
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Usage

Run the pipeline on audio files:

```bash
python unified_neural_pipeline.py --mixture mixture_audio.wav --target target_sample.wav --output_dir output
```

This will generate:
- `output/target_speaker.wav`: Clean separated voice of the target speaker.
- `output/diarization.json`: Timestamped transcripts with speaker labels.

## Modules

- `audio_preprocessing.py`: Handles initial audio processing.
- `speaker_separation.py`: Performs target speaker isolation.
- `diarization.py`: Manages speaker segmentation.
- `asr.py`: Handles speech-to-text conversion.
- `post_processing.py`: Finalizes transcripts and outputs.

## Evaluation Criteria

- **System Design**: Modular, scalable architecture.
- **Code Quality**: Readable, maintainable code with proper documentation.
- **Performance**: High accuracy in diarization, separation, and ASR.
- **Innovation**: Optimized for real-time processing.
- **Deployment**: Ready for production use with APIs.

## License

© 2025 Awetales - Confidential