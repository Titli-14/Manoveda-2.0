import os
from pydub import AudioSegment
from pydub.utils import which

# Explicitly set ffmpeg and ffprobe paths
AudioSegment.converter = which("ffmpeg") or r"C:\Users\TITLI DUTTA\Desktop\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = which("ffprobe") or r"C:\Users\TITLI DUTTA\Desktop\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

def convert_to_wav(file_path):
    """
    Converts any audio file to standard WAV (16-bit PCM, mono, 22050Hz).
    Returns path to converted file or None if conversion fails.
    """
    try:
        # Create new filename
        wav_path = os.path.splitext(file_path)[0] + "_converted.wav"

        # Load audio with pydub
        audio = AudioSegment.from_file(file_path)

        # Standardize format
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

        # Export as WAV
        audio.export(wav_path, format="wav")
        return wav_path

    except Exception as e:
        print(f"[convert_to_wav] Failed for {file_path}: {e}")
        return None

