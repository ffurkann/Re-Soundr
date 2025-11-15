# Re-Soundr: Automated Video Dubbing Pipeline

Re-Soundr is a modular pipeline that takes a video, extracts dialogue, processes speakers, translates the text, and regenerates speech using AI-powered models. It is designed to be fully extensible so each stage can be improved or replaced independently.

## Features

### 1. Audio Extraction
- Extracts audio tracks from any video format.
- Supports optional background separation.
- Implemented in `video_input/extract_audio.py` and `video_input/seperate_background.py`.

### 2. Speaker Recognition (AI)
- Uses speaker embeddings to detect who is speaking.
- Enables multi-speaker dubbing workflows.
- Logic implemented in `video_input/get_speaker.py`.

### 3. Dialogue Segmentation
- Splits audio into meaningful, processable segments.
- Uses silence detection and custom heuristics.
- Implemented in `video_input/get_segments.py`.

### 4. Speech-to-Text (AI)
- Converts audio segments into text using an AI transcription model.
- Implemented in `video_input/get_text.py`.

### 5. Translation (AI)
- Translates the extracted text into the target language.
- Supports:
  - Pretrained transformer models
  - Custom fine-tuned translation models
- Implemented in `translation/pre_trained.py` and `translation/model_train_and_test.py`.

### 6. Voice Cloning / Text-to-Speech (AI)
- Regenerates translated text as speech using cloned voices.
- Architecture inspired by XTTS/YourTTS systems.
- Implemented in `dub/voice_clone.py`.

### 7. Final Reconstruction
- Intended to recombine generated audio with the original video.
- Lip-sync engine and video rebuilding are planned future components.
