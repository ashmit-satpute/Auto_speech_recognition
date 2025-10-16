# Auto_speech_recognition
This project enables Automatic Speech Recognition on  15 seconds audio streams/clips, using transformer model-based of whisper AI architecture, for transcibing speech into text.

FILE STRUCTURE:
ASR_TRANSFORMER_FINAL_weights.h5 -> weights of the transformer model trained yeilding an accurcy of ⁓50%.
inference.py -> runs inference on the 5 sec audio recorded from your device.
text_gen.py -> preprocesses audio data, generates output using model and returns the post processed output.
transformer_model.py -> contains ASRTransformer class that initializes the transformer modeel.
