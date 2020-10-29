# Custom-Transformer-Model-with-Tensorflow-2.3
Custom Transformer Model with Tensorflow 2.3 for machine translation



## Prerequisites

Tensorflow 2.3

Python 3.7

## About

An implementation of the Transformer Deep Learning Architecture that can be easily altered to define the dimensions of attention mechanisms within the encoder/decoder stacks + the number of encoders and decoders in each of their respective stacks.


The tf.function decorator is used with an input signature for the __call__(self, encoded_input, output_length) method. This instructs Tensorflow to build the graph and will avoid retracing for different length inputs/outputs.


## Running

Set the parameters within exec.py to define the exact architecture of the transformer + the input/output sources for embedded vocabularies and trainings sentences.

Within embeddings.py, the values for the input/output dimensions must be defined as well as the output vocabulary size which is used to define the final output layer.


## Further work

This implementation uses a one-hot encoding across the output vocabulary for the final output layer. I plan on reducing this by training this to an output embedding, drastically reducing dimensionality and speeding up training by preserving relative information between words.


This will then require an efficient means of nearest neighbour search to infer the output word after the training phase


The loss will stop accumulating during the training phase once the predicted output sentence reaches the length of the target output sentence. This should be adjusted to recognise end of sentence.
