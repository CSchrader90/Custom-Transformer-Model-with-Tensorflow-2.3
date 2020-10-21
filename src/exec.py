import tensorflow as tf
import numpy as np
import argparse
import string
from word_embeddings import embeddings

# Self-Attention constants
ENCODER_ATTENTION_HEAD_DIM = 50
DECODER_ATTENTION_HEAD_DIM = 100
NUMBER_HEADS = 6

# Encoders' FFN architecture
FFN_INTERNAL_DIM = 200

# Encoder/Decoder stacks
NUMBER_ENCODERS = 6
NUMBER_DECODERS = 6

# Optimiser
LEARNING_RATE = 0.01

# Output vocab directory
OUTPUT_VOCAB_DIR = "../lib/python3.7/site-packages/word_embeddings/fastTextFinnish100d.csv"


# Input/Output training sentences
ENGLISH_SENTENCES = "./english_sentences.txt"
FINNISH_SENTENCES = "./finnish_sentences.txt"


def attention_eval(key, value, query, dim):
    softmax = tf.nn.softmax(tf.matmul(query, tf.transpose(key)/np.sqrt(dim)))

    return tf.matmul(softmax, value)


class SelfAttentionHead:
    def __init__(self, attention_dim):
        self._KeyMat = tf.Variable(initializer((attention_dim, attention_dim)), name="KeyMat", trainable=True, dtype=tf.float32)
        self._QueryMat = tf.Variable(initializer((attention_dim, attention_dim)), name="QueryMat", trainable=True, dtype=tf.float32)
        self._ValueMat = tf.Variable(initializer((attention_dim, attention_dim)), name="ValueMat", trainable=True, dtype=tf.float32)
        weights.append([self._KeyMat, self._QueryMat, self._ValueMat])

    def __call__(self, X):
        key = tf.matmul(X, self._KeyMat)
        value = tf.matmul(X, self._ValueMat)
        query = tf.matmul(X, self._QueryMat)

        return attention_eval(key, value, query, ENCODER_ATTENTION_HEAD_DIM)


class MultiHeadAttention:
    def __init__(self, attention_dim, num_heads):
        self._AttentionHeads = [SelfAttentionHead(attention_dim) for i in range(num_heads)]
        self._WO = tf.Variable(initializer((attention_dim*num_heads, attention_dim)), trainable=True, dtype=tf.float32)
        weights.append([self._WO])

    def __call__(self, x):
        z_out = [head(x) for head in self._AttentionHeads]
        comb_z = z_out[0]
        for i in z_out[1:]:
            comb_z = tf.concat([comb_z, i], axis=1)

        return tf.matmul(comb_z, self._WO)


class FeedForwardNetwork:
    # Output layer dimension set to input layer dimension
    def __init__(self, input_dim, internal_dim):
        self._W1 = tf.Variable(initializer((input_dim, internal_dim)), trainable=True, dtype=tf.float32)
        self._b1 = tf.Variable(initializer((1, internal_dim)), trainable=True, dtype=tf.float32)
        self._W2 = tf.Variable(initializer((internal_dim, input_dim)), trainable=True, dtype=tf.float32)
        self._b2 = tf.Variable(initializer((1, input_dim)), trainable=True, dtype=tf.float32)
        weights.append([self._W1, self._b1, self._W2, self._b2])

    def __call__(self, X):
        relu_out = tf.add(tf.nn.leaky_relu(tf.matmul(X, self._W1)), self._b1)

        return tf.add(tf.matmul(relu_out, self._W2), self._b2)


def add_and_normalise(X, Z):
    comb = tf.add(X, Z)
    std = tf.math.reduce_std(comb, axis=1, keepdims=True)
    mean = tf.reduce_mean(comb, axis=1, keepdims=True)

    return tf.divide(tf.subtract(comb, mean), std)


class Encoder:
    def __init__(self, attention_dim, number_heads, ffn_internal_dim):
        self._self_attention = MultiHeadAttention(attention_dim, number_heads)
        self._feed_forward = FeedForwardNetwork(attention_dim, ffn_internal_dim)

    def __call__(self, input):
        z1 = self._self_attention(input)
        z2 = add_and_normalise(input, z1)
        z3 = self._feed_forward(z2)
        z4 = add_and_normalise(z2, z3)

        return z4


# Decoding
class Decoder:
    def __init__(self, attention_dim, num_heads, ffn_internal_dim):
        self._self_attention = MultiHeadAttention(attention_dim, num_heads)
        self._query_weights = tf.Variable(initializer((attention_dim, attention_dim)), trainable=True,
                                          dtype=tf.float32)
        self._enc_dec_key_w = tf.Variable(initializer((ENCODER_ATTENTION_HEAD_DIM, DECODER_ATTENTION_HEAD_DIM)),
                                          trainable=True, dtype=tf.float32)
        self._enc_dec_val_w = tf.Variable(initializer((ENCODER_ATTENTION_HEAD_DIM, DECODER_ATTENTION_HEAD_DIM)),
                                          trainable=True, dtype=tf.float32)
        self._feed_forward = FeedForwardNetwork(attention_dim, ffn_internal_dim)
        weights.append([self._query_weights, self._enc_dec_key_w, self._enc_dec_val_w])

    def __call__(self, input, attention_dim, encoder_output):
        # Self-attention layer
        z1 = self._self_attention(input)
        z2 = add_and_normalise(input, z1)

        # Encoder-decoder layer
        query = tf.matmul(z2, self._query_weights)
        key = tf.matmul(encoder_output, self._enc_dec_key_w)
        val = tf.matmul(encoder_output, self._enc_dec_val_w)
        enc_dec_att = attention_eval(key, val, query, attention_dim)
        z3 = add_and_normalise(z2, enc_dec_att)

        # Feed-forward layer
        z4 = self._feed_forward(z3)
        z5 = add_and_normalise(z4, z3)

        return z5


# Final layers after Decoder Stack
class FinalLayers:
    def __init__(self, input_dim, output_dim):
        self._W = tf.Variable(initializer((input_dim, output_dim)), trainable=True, dtype=tf.float32)
        self._b = tf.Variable(initializer((1, output_dim)), trainable=True, dtype=tf.float32)
        weights.append([self._W, self._b])

    def __call__(self, x):
        # linear layer
        ll_out = tf.add(tf.matmul(tf.reshape(tf.reduce_sum(x, axis=0), (1, x.shape[1])), self._W), self._b)
        # softmax layer
        return tf.nn.softmax(ll_out)


class Transformer(tf.Module):
    def __init__(self, encoder_attention_head_dim, num_heads, ffn_internal_dim, decoder_attention_head_dim, num_encoders, num_decoders):
        super().__init__()
        # Create Encoder stack
        self._encoder_Stack = [Encoder(encoder_attention_head_dim, num_heads, ffn_internal_dim) for i in
                         range(num_encoders)]

        # Create Decoder Stack
        self._decoder_Stack = [Decoder(decoder_attention_head_dim, num_heads, ffn_internal_dim) for i in
                         range(num_decoders)]
        self._last_decoder_output = tf.Variable(zero_init((1, decoder_attention_head_dim)), trainable=True, dtype=tf.float32)
        self._final_Layers = FinalLayers(embeddings.OUTPUT_EMBEDDING_SIZE, embeddings.OUTPUT_VOCAB_SIZE)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, ENCODER_ATTENTION_HEAD_DIM), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32)])
    def __call__(self, encoded_input_sentence, out_length):

        for enc_idx in range(0, NUMBER_ENCODERS):
            if enc_idx == 0:
                encoder_output = self._encoder_Stack[enc_idx](encoded_input_sentence)
            else:
                encoder_output = self._encoder_Stack[enc_idx](encoder_output)
        last_encoder_output = encoder_output

        output_word_idx = 1  # 1-based
        decoder_posit = embeddings.posit_encode(output_word_idx, embeddings.OUTPUT_EMBEDDING_SIZE)
        decoder_input = tf.add(self._last_decoder_output, decoder_posit)

        decoder_output_sentence = tf.TensorArray(tf.float32, size=out_length, element_shape=(1, embeddings.OUTPUT_VOCAB_SIZE))
        decoder_output = initializer((1, DECODER_ATTENTION_HEAD_DIM), dtype=tf.float32)
        for out_sen_idx in range(out_length):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(decoder_output, tf.TensorShape([None, DECODER_ATTENTION_HEAD_DIM]))])
            for dec_num in range(NUMBER_DECODERS):
                if out_sen_idx == 0:
                    decoder_output = self._decoder_Stack[dec_num](decoder_input, DECODER_ATTENTION_HEAD_DIM, last_encoder_output)
                else:
                    decoder_output = self._decoder_Stack[dec_num](decoder_output, DECODER_ATTENTION_HEAD_DIM, last_encoder_output)
            last_decoder_output = decoder_output

            # pass through linear layer & softmax
            last_layer_output = self._final_Layers(last_decoder_output)
            decoder_output_sentence.write(out_sen_idx, last_layer_output)

            # feed word back to input of decoder stack
            output_word_idx += 1
            decoder_posit = embeddings.posit_encode(output_word_idx, embeddings.OUTPUT_EMBEDDING_SIZE)
            new_input = tf.add(last_decoder_output, decoder_posit[output_word_idx - 1, :])
            decoder_input = tf.concat([decoder_input, new_input], 0)

        return decoder_output_sentence.concat()


def loss(target, predicted):
    return tf.losses.kullback_leibler_divergence(tf.reshape(target, (1, -1)), tf.reshape(predicted, (1, -1)))


def output_dict_idx(dictionary, search_word):
    if search_word in dictionary.keys():
        return dictionary[search_word]
    else:
        return 0


# Variable Initialisers
initializer = tf.initializers.glorot_uniform()
zero_init = tf.zeros_initializer()

# fetch training data
train_in_data = open(ENGLISH_SENTENCES)
train_out_data = open(FINNISH_SENTENCES)

train_in_sentences = [line.split() for line in train_in_data]
train_out_sentences = [line.split() for line in train_out_data]

assert len(train_in_sentences) == len(train_out_sentences)
num_sentences = len(train_in_sentences)
for sentence in range(num_sentences):
    for word in range(len(train_in_sentences[sentence])):
        train_in_sentences[sentence][word] = train_in_sentences[sentence][word].lower().translate(str.maketrans('', '', string.punctuation))
    for word in range(len(train_out_sentences[sentence])):
        train_out_sentences[sentence][word] = train_out_sentences[sentence][word].lower().translate(str.maketrans('', '', string.punctuation))

# input embeddings
word_embeddings, word_dict = embeddings.load_input_embeddings()

# output embeddings
output_vocab_list, output_dict = embeddings.get_output_dict(OUTPUT_VOCAB_DIR)

weights = []
optimizer = tf.optimizers.Adam(LEARNING_RATE)

transformer = Transformer(ENCODER_ATTENTION_HEAD_DIM, NUMBER_HEADS, FFN_INTERNAL_DIM, DECODER_ATTENTION_HEAD_DIM, NUMBER_ENCODERS, NUMBER_DECODERS)
for sentence_idx in range(num_sentences):

    # Encoding input sentence
    input_sentence = train_in_sentences[sentence_idx]
    enc_posit_encodings = tf.constant(embeddings.posit_encode(len(input_sentence), embeddings.INPUT_EMBEDDING_SIZE))
    embedded_input_sentence = embeddings.glove_embeddings(word_embeddings, word_dict, input_sentence,
                                                    embeddings.INPUT_EMBEDDING_SIZE)
    encoded_input = tf.cast(tf.add(embedded_input_sentence, enc_posit_encodings), dtype=tf.float32)

    # Encoding target sentence
    output_sentence = train_out_sentences[sentence_idx]
    dec_posit_encodings = tf.constant(embeddings.posit_encode(len(output_sentence), embeddings.OUTPUT_EMBEDDING_SIZE))
    embedded_output_sentence = tf.one_hot(output_dict_idx(output_dict, output_sentence[0]), depth=embeddings.OUTPUT_VOCAB_SIZE, dtype=tf.float32)

    for out_word_idx in range(1, len(output_sentence)):
        cur_word = output_sentence[out_word_idx]
        embedded_output_sentence = tf.concat([embedded_output_sentence, tf.one_hot(output_dict_idx(output_dict, output_sentence[sentence_idx]), depth=embeddings.OUTPUT_VOCAB_SIZE)], axis=0)

    embedded_output_sentence = tf.reshape(embedded_output_sentence, (1, -1))
    with tf.GradientTape() as tape:
        transformer_output = transformer(encoded_input, len(output_sentence))

    cur_loss = loss(embedded_output_sentence, transformer_output)
    grads = tape.gradient(cur_loss, weights)

    for i in range(len(grads)):
        optimizer.apply_gradients(zip(grads[i], weights[i]))
