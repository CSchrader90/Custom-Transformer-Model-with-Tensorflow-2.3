import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import argparse
import string
from word_embeddings import embeddings

#Self-Attention constants
ENCODER_ATTENTION_HEAD_DIM = 50
DECODER_ATTENTION_HEAD_DIM = 100
NUMBER_HEADS = 6

#Encoders' FFN architecture
FFN_INTERNAL_DIM = 200

#Encoder/Decoder stacks
NUMBER_ENCODERS = 6
NUMBER_DECODERS = 6

def attention_eval(key, value, query, dim):
    softmax = tf.nn.softmax(tf.matmul(query, tf.transpose(key)/np.sqrt(dim)))

    return tf.matmul(softmax, value)

class SelfAttentionHead:
    def __init__(self, attention_dim):
        self._KeyMat = tf.Variable(initializer((attention_dim, attention_dim)), name="KeyMat", trainable=True, dtype=tf.float32)
        self._QueryMat = tf.Variable(initializer((attention_dim, attention_dim)), name="QueryMat", trainable=True, dtype=tf.float32)
        self._ValueMat = tf.Variable(initializer((attention_dim, attention_dim)), name="ValueMat", trainable=True, dtype=tf.float32)

    def eval(self, X):
        key = tf.matmul(X, self._KeyMat)
        value = tf.matmul(X, self._ValueMat)
        query = tf.matmul(X, self._QueryMat)

        return attention_eval(key, value, query, ENCODER_ATTENTION_HEAD_DIM)

class MultiHeadAttention(SelfAttentionHead):
    def __init__(self, attention_dim, num_heads):
        self._AttentionHeads = [SelfAttentionHead(attention_dim) for i in range(num_heads)]
        self._WO = tf.Variable(initializer((attention_dim*num_heads, attention_dim)), trainable=True, dtype=tf.float32)
    def eval(self, x):
        z_out = [head.eval(x) for head in self._AttentionHeads]
        comb_z = z_out[0]
        for i in z_out[1:]:
            comb_z = tf.concat([comb_z, i], axis=1)

        return tf.matmul(comb_z, self._WO)

class FeedForwardNetwork:
    #Output layer dimension set to input layer dimension
    def __init__(self, input_dim, internal_dim):
        self._W1 = tf.Variable(initializer((input_dim, internal_dim)), trainable=True, dtype=tf.float32)
        self._b1 = tf.Variable(initializer((1, internal_dim)), trainable=True, dtype=tf.float32)
        self._W2 = tf.Variable(initializer((internal_dim, input_dim)), trainable=True, dtype=tf.float32)
        self._b2 = tf.Variable(initializer((1, input_dim)), trainable=True, dtype=tf.float32)

    def eval(self, X):
        relu_out = tf.add(tf.nn.relu(tf.matmul(X, self._W1)), self._b1)

        return tf.add(tf.matmul(relu_out, self._W2), self._b2)

def add_and_normalise(X, Z):
    comb = tf.add(X, Z)
    std = tf.reshape(tfp.stats.stddev(comb, sample_axis=1), (comb.shape[0], -1))
    mean = tf.reshape(tf.reduce_mean(comb, axis=1), (comb.shape[0], -1))

    return tf.divide(tf.subtract(comb, mean), std)

class Encoder(MultiHeadAttention, FeedForwardNetwork):
    def __init__(self, attention_dim, number_heads, ffn_internal_dim):
        self._self_attention = MultiHeadAttention(attention_dim, number_heads)
        self._feed_forward = FeedForwardNetwork(attention_dim, ffn_internal_dim)

    def eval(self, input):
        z1 = self._self_attention.eval(input)
        z2 = add_and_normalise(input, z1)
        z3 = self._feed_forward.eval(z2)
        z4 = add_and_normalise(z2, z3)

        return z4

# Decoding
class Decoder(MultiHeadAttention, FeedForwardNetwork):
    def __init__(self, attention_dim, num_heads, ffn_internal_dim):
        self._self_attention = MultiHeadAttention(attention_dim, num_heads)
        self._query_weights = tf.Variable(initializer((attention_dim, attention_dim)), trainable=True,
                                          dtype=tf.float32)
        self._enc_dec_key_w = tf.Variable(initializer((ENCODER_ATTENTION_HEAD_DIM, DECODER_ATTENTION_HEAD_DIM)),
                                          trainable=True, dtype=tf.float32)
        self._enc_dec_val_w = tf.Variable(initializer((ENCODER_ATTENTION_HEAD_DIM, DECODER_ATTENTION_HEAD_DIM)),
                                          trainable=True, dtype=tf.float32)
        self._feed_forward = FeedForwardNetwork(attention_dim, ffn_internal_dim)

    def eval(self, input, attention_dim, encoder_output):
        # Self-attention layer
        z1 = self._self_attention.eval(input)
        z2 = add_and_normalise(input, z1)

        # Encoder-decoder layer
        query = tf.matmul(z2, self._query_weights)
        key = tf.matmul(encoder_output, self._enc_dec_key_w)
        val = tf.matmul(encoder_output, self._enc_dec_val_w)
        enc_dec_att = attention_eval(key, val, query, attention_dim)
        z3 = add_and_normalise(z2, enc_dec_att)

        # Feed-forward layer
        z4 = self._feed_forward.eval(z3)
        z5 = add_and_normalise(z4, z3)

        return z5

#Final layers after Decoder Stack
class FinalLayers:
    def __init__(self, input_dim, output_dim):
        self._W = tf.Variable(initializer((input_dim, output_dim)))
        self._b = tf.Variable(initializer((1, output_dim)))

    def eval(self, x):
        # linear layer
        ll_out = tf.add(tf.matmul(tf.reshape(tf.reduce_sum(x, axis=0), (1, x.shape[1])), self._W), self._b)
        # softmax layer
        return tf.nn.softmax(ll_out)

#Variable Initialisers
initializer = tf.initializers.glorot_uniform()
zero_init  = tf.zeros_initializer()

#parse input sentence
parser = argparse.ArgumentParser()
parser.add_argument('input_string',  type=str)
args = parser.parse_args()
input_sentence = args.input_string

#embed input sentence
word_embeddings, word_dict = embeddings.load_input_embeddings()
input_sentence = input_sentence.translate(str.maketrans('', '', string.punctuation))
split_sentence = [w.lower() for w in input_sentence.split()] #tokenize sentence and remove capitalisation
embed_sen = embeddings.glove_embeddings(word_embeddings, word_dict, split_sentence, embeddings.EMBEDDING_SIZE)

#get the positional encodings
posit_encodings = embeddings.posit_encode(len(input_sentence.split()), embeddings.EMBEDDING_SIZE)

#Add sentence embedding with positional encoding
encoded_input = tf.cast(tf.add(embed_sen, posit_encodings), dtype=tf.float32)

#Create Encoder stack
Encoder_Stack = [Encoder(ENCODER_ATTENTION_HEAD_DIM, NUMBER_HEADS, FFN_INTERNAL_DIM) for i in range(NUMBER_ENCODERS)]

#Feed through Encoder stack
for i in range(0, NUMBER_ENCODERS):
    if i == 0:
        output = Encoder_Stack[i].eval(encoded_input)
    else:
        output = Encoder_Stack[i].eval(output)

Encoder_Stack_output = output

#Feed through Decoder stack
Decoder_Stack = [Decoder(DECODER_ATTENTION_HEAD_DIM, NUMBER_HEADS, FFN_INTERNAL_DIM) for i in range(NUMBER_DECODERS)]
Final_Layers = FinalLayers(embeddings.FINNISH_EMBEDDING_SIZE, embeddings.OUTPUT_VOCAB_SIZE)

output_word_idx = 1 #1-based

last_decoder_output = tf.Variable(zero_init((1, DECODER_ATTENTION_HEAD_DIM)), trainable=True, dtype=tf.float32)
decoder_posit = embeddings.posit_encode(output_word_idx, embeddings.OUTPUT_EMBEDDING_SIZE)
decoder_input = tf.add(last_decoder_output, decoder_posit)

word = ""
while word is not None:
    for i in range(NUMBER_DECODERS):
        if i == 0:
            decoder_output = Decoder_Stack[i].eval(decoder_input, DECODER_ATTENTION_HEAD_DIM, Encoder_Stack_output)
        else:
            decoder_output = Decoder_Stack[i].eval(decoder_output, DECODER_ATTENTION_HEAD_DIM, Encoder_Stack_output)
    last_decoder_output = decoder_output

    #pass through linear layer & softmax
    last_layer_output = Final_Layers.eval(last_decoder_output)
    word = tf.argmax(last_layer_output, 1)

    #IMPLEMENT DICTIONARY LOOKUP

    #feed word back to input of decoder stack
    output_word_idx += 1
    decoder_posit = embeddings.posit_encode(output_word_idx, embeddings.OUTPUT_EMBEDDING_SIZE)
    new_input = tf.add(decoder_output, decoder_posit[output_word_idx - 1, :])
    decoder_input = tf.concat([decoder_input, new_input], 0)