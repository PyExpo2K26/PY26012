import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64   

def analyze_audio(file_path):
  
    try:
        y, sr = librosa.load(file_path, duration=10, sr=None)
        
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        spectrum_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        mfcc_var = np.var(mfccs, axis=1).mean()
        contrast_mean = np.mean(spectral_contrast)
        score = 0.0
        if mfcc_var < 30:
            score = 0.8 # High probability of being fake
        elif mfcc_var < 50:
            score = 0.4
        else:
            score = 0.1
            
        return score, spectrum_base64

    except Exception as e:
        print(f"Audio Analysis Error: {e}")
        return 0.0, ""
