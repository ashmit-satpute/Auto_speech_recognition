import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
from transformer_model import get_asr_transformer
from IPython.display import display, Audio
from text_gen import create_mel_spectrogram, generate_text
import numpy as np
import librosa
import matplotlib.pyplot as plt


freq = 16000
duration = 5

print('RECORDING')
rec = sd.rec(int(duration*freq), samplerate=freq, channels=1)
write('recording0.mp3', freq, rec)

sd.wait()
rec *= 8
write("recording0.mp3", freq, rec)
print(np.squeeze(rec, axis=-1).shape)

print(generate_text(get_asr_transformer(), create_mel_spectrogram(rec/37268, freq)))
