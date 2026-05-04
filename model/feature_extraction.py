import librosa
import numpy as np

# Fixed feature order: 13 MFCC + 12 Chroma + 3 Spectral Contrast + 2 Spectral Features = 30
FEATURE_NAMES = (
    [f"mfcc_{i+1}" for i in range(13)] +
    [f"chroma_{i+1}" for i in range(12)] +
    [f"spectral_contrast_{i+1}" for i in range(3)] +
    ["spectral_centroid", "spectral_bandwidth"]
)

def extract_features(file_path):
    """
    Extracts 30 audio features in a fixed order:
    - 13 MFCC
    - 12 Chroma
    - 3 Spectral Contrast (first 3 bands)
    - Spectral Centroid
    - Spectral Bandwidth
    Returns:
        dict: feature_name -> value
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # 1️⃣ MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 2️⃣ Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 3️⃣ Spectral Contrast (first 3 bands)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
    spec_contrast_selected = spec_contrast[:3]

    # 4️⃣ Other spectral features
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Combine all features in fixed order (30 values)
    feature_values = list(mfccs_mean) + list(chroma_mean) + list(spec_contrast_selected) + \
                     [spec_centroid, spec_bandwidth]

    # Map feature names to values
    feature_dict = dict(zip(FEATURE_NAMES, feature_values))
    return feature_dict
