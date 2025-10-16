import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


# SAMPLE_RATE = 22050
# MAX_INPUT_LENGTH_IN_SEC = 5
OUTPUT_SEQUENCE_LENGTH = 357
# BATCH_SIZE = 64
# BUFFER_SIZE = tf.data.AUTOTUNE
# EPOCHS = 30
# LEARNING_RATE = 1e-6
# D_MODEL = 256  # Embedding & Transformer dimension
# NUM_HEADS = 8
# NUM_LAYERS = 2
# NUM_DECODER_LAYERS = 2
# DFF = 512  # Hidden layer size in feed-forward network
# DROPOUT_RATE = 0.1
# VOCAB_SIZE = 31
VOCABULARY = {'<pad>': 0, '<sos>': 1, ' ': 2, 'e': 3, 't': 4, 'o': 5, 'a': 6, 'i': 7, 'h': 8, 'n': 9, 's': 10, 'r': 11, 'd': 12, 'l': 13, 'u': 14, 'y': 15, 'w': 16, 'm': 17, 'g': 18, 'c': 19, 'f': 20, 'b': 21, 'p': 22, 'k': 23, "'": 24, 'v': 25, 'j': 26, 'x': 27, 'q': 28, 'z': 29,  '<eos>': 30}


# SAMPLE_RATE = 22050
# MAX_INPUT_LENGTH_IN_SEC = 5
# OUTPUT_SEQUENCE_LENGTH = 119
# BATCH_SIZE = 64
# BUFFER_SIZE = tf.data.AUTOTUNE
# EPOCHS = 30
# LEARNING_RATE = 1e-6
# D_MODEL = 256  # Embedding & Transformer dimension
# NUM_HEADS = 8
# NUM_LAYERS = 2
# NUM_DECODER_LAYERS = 2
# DFF = 512  # Hidden layer size in feed-forward network
# DROPOUT_RATE = 0.1
# VOCAB_SIZE = 31
# VOCABULARY = {'<pad>': 0, '<sos>': 1, ' ': 2, 'e': 3, 't': 4, 'o': 5, 'a': 6, 'i': 7, 'h': 8, 'n': 9, 's': 10, 'r': 11, 'd': 12, 'l': 13, 'u': 14, 'y': 15, 'w': 16, 'm': 17, 'g': 18, 'c': 19, 'f': 20, 'b': 21, 'p': 22, 'k': 23, "'": 24, 'v': 25, 'j': 26, 'x': 27, 'q': 28, 'z': 29,  '<eos>': 30}


def scaled_dot_product(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e4)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(k)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask, training=False):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        self_attn_output, self_attn_weight = self.mha1(x, x, x, look_ahead_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(x + self_attn_output)

        cross_attn_output, cross_attn_weight = self.mha2(out1, enc_output, enc_output, padding_mask)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3, self_attn_weight, cross_attn_weight


class Conv1DStem(keras.layers.Layer):
    def __init__(self, d_model):
        super(Conv1DStem, self).__init__()
        self.conv1 = keras.layers.Conv1D(64, 3, padding='same', activation='gelu')
        self.conv2 = keras.layers.Conv1D(128, 3, strides=2, padding='same', activation='gelu')
        self.projection_layer = keras.layers.Dense(d_model)

    def call(self, spectrogram):
        spectrogram = self.conv1(spectrogram)
        spectrogram = self.conv2(spectrogram)
        spectrogram_features = self.projection_layer(spectrogram)  # shape = [batch, timeframe/2, d_model]
        return spectrogram_features


class Encoder(keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, num_layers, rate=0.1):
        super(Encoder, self).__init__()
        self.num_head = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(1501 // 2 + 1, self.d_model)
        # self.embeding = tf.keras.layers.Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate=0.1) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
        self.conv_stem = Conv1DStem(d_model)

    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (
                    2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, training=False):
        x = self.conv_stem(x)
        seq_length = x.shape[1]

        enc_padded_mask = tf.cast(tf.math.equal(tf.reduce_sum(tf.abs(x), axis=-1), 0.0), tf.float32)
        enc_padded_mask = enc_padded_mask[:, tf.newaxis, tf.newaxis, :]

        # x = self.embeding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=enc_padded_mask, training=training)

        return x


class Decoder(keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, num_layers, vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(OUTPUT_SEQUENCE_LENGTH, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate=0.1) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
        self.final_layer = keras.layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (
                    2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        seq_length = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x, training=training)

        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output=enc_output, look_ahead_mask=look_ahead_mask,
                                                   padding_mask=padding_mask, training=training)
            attention_weights[f"decoder_layer{i + 1}_block1"] = block1
            attention_weights[f"decoder_layer{i + 1}_block2"] = block2

        final_output = self.final_layer(x)
        return final_output, attention_weights


class ASRTransformer(keras.Model):
    def __init__(self, d_model, dff, num_heads, vocab_size, num_enc_layers, num_dec_layers, rate=0.1):
        super(ASRTransformer, self).__init__()
        self.encoder = Encoder(d_model, dff, num_heads, num_enc_layers,
                               rate=0.1)  # d_model, dff, num_heads, num_layers, rate=0.1
        self.decoder = Decoder(d_model, dff, num_heads, num_dec_layers, vocab_size,
                               rate=0.1)  # d_model, dff, num_heads, num_layers, vocab_size, rate=0.1

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_spec_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(tf.reduce_sum(tf.abs(seq), axis=-1), 0.0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq_len):
        # seq_len = 118
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask

    def call(self, inp, tar, training=False):
        dec_padding_mask = self.create_padding_mask(tar)

        enc_output = self.encoder(inp, training=training)

        enc_output_padding_mask = self.create_spec_padding_mask(enc_output)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output=enc_output, look_ahead_mask=look_ahead_mask,
                                                     padding_mask=enc_output_padding_mask, training=training)
        return dec_output, attention_weights

def get_asr_transformer():
    SAMPLE_RATE = 22050
    MAX_INPUT_LENGTH_IN_SEC = 15
    OUTPUT_SEQUENCE_LENGTH = 357
    BATCH_SIZE = 64
    #BUFFER_SIZE = tf.data.AUTOTUNE
    #EPOCHS = 58
    #LEARNING_RATE = 1e-4
    D_MODEL = 256  # Embedding & Transformer dimension
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DFF = 512  # Hidden layer size in feed-forward network
    DROPOUT_RATE = 0.1
    VOCAB_SIZE = 32

    transformer_model = ASRTransformer(D_MODEL, DFF, NUM_HEADS, VOCAB_SIZE, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                               DROPOUT_RATE)  # d_model, dff, num_heads, vocab_size, num_layers, rate=0.1
    DUMMY_MEL_FRAMES = 1501
    DUMMY_FREQ_BINS = 193
    inp = tf.random.uniform((BATCH_SIZE, DUMMY_MEL_FRAMES, DUMMY_FREQ_BINS), dtype=tf.float32)
    out = tf.random.uniform((BATCH_SIZE, OUTPUT_SEQUENCE_LENGTH - 1), minval=1, maxval=VOCAB_SIZE, dtype=tf.int32)
    # predicitions
    preds, _ = transformer_model(inp, out, training=False)
    transformer_model.load_weights('Transformer_36.weights.h5')
    return transformer_model