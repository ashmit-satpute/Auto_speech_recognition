import tensorflow as tf
import librosa
import numpy as np

SAMPLE_RATE = 22050
MAX_INPUT_LENGTH_IN_SEC = 15
OUTPUT_SEQUENCE_LENGTH = 357 # This is used by the decoder's positional encoding
# BATCH_SIZE = 64
# BUFFER_SIZE = tf.data.AUTOTUNE
# EPOCHS = 30
# LEARNING_RATE = 1e-6
D_MODEL = 256  # Embedding & Transformer dimension
NUM_HEADS = 8
NUM_LAYERS = 2 # Assuming NUM_ENCODER_LAYERS and NUM_DECODER_LAYERS are both 2 based on your code
DFF = 512  # Hidden layer size in feed-forward network
DROPOUT_RATE = 0.1
VOCAB_SIZE = 32
VOCABULARY = {'<pad>': 0, '<sos>': 1, ' ': 2, 'e': 3, 't': 4, 'o': 5, 'a': 6, 'i': 7, 'h': 8, 'n': 9, 's': 10, 'r': 11, 'd': 12, 'l': 13, 'u': 14, 'y': 15, 'w': 16, 'm': 17, 'g': 18, 'c': 19, 'f': 20, 'b': 21, 'p': 22, 'k': 23, "'": 24, 'v': 25, 'j': 26, 'x': 27, 'q': 28, 'z': 29,  '<eos>': 30, '<vnd>':31}

# Reverse vocabulary for decoding token IDs back to characters
ID_TO_CHAR = {idx: char for char, idx in VOCABULARY.items()}

# --- Helper Functions ---

def create_mel_spectrogram(audio_data, sr):
    audio = np.squeeze(audio_data, axis=-1)
    # audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)


    # Pad/truncate audio to MAX_INPUT_LENGTH_IN_SEC
    max_input_length_in_sec = 15
    sample_rate = 16000
    target_length_samples = max_input_length_in_sec * sample_rate
    if len(audio) < target_length_samples:
       audio = np.pad(audio, (0, 15*16000 - len(audio)), mode='constant', constant_values=0)

    n_fft = 400
    hop_length = 160
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=193,
        fmin=0.0,
        fmax=8000.0,
        window='hann'  # Explicitly specify Hann window, though it's librosa's default for melspec
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)  # ref=1.0 for log(power)
    log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / (np.std(log_mel_spectrogram) + 1e-6)
    log_mel_spectrogram = tf.transpose(log_mel_spectrogram, perm=[1, 0])
    return np.expand_dims(log_mel_spectrogram, axis=0)

def generate_text(model, audio_spectrogram):
    """
    Performs greedy decoding to generate text from a spectrogram.
    """
    # Start with the <sos> token
    decoder_input = tf.expand_dims([VOCABULARY['<sos>']], 0)

    output_tokens = []

    # Iterate until <eos> is predicted or max sequence length is reached
    for i in range(OUTPUT_SEQUENCE_LENGTH):
        predictions, _ = model(audio_spectrogram, decoder_input, training=False)

        # Select the last token from the sequence and get the most probable char
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # If <eos> is predicted, stop
        if predicted_id == VOCABULARY['<eos>']:
            break

        # Add the predicted token to the output and continue
        output_tokens.append(predicted_id.numpy()[0][0])
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    # Convert token IDs back to characters
    decoded_text = ''.join([ID_TO_CHAR[token_id] for token_id in output_tokens])
    return decoded_text