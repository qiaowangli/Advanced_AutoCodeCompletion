#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
"""
Last edited by   : Roy
Last edited time : 14/11/2021
Version Status: stable
TO DO: Skipgram implementaion with tf
"""

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim,neg_example):
    super(Word2Vec, self).__init__()
    self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=neg_example+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

#reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/skipgrams

def generate_training_data(sequences, window_size, num_ns, vocab_size):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # # Build the sampling table for vocab_size tokens.
  # sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in sequences:

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=10,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def word_embedding(tokenized_subsequence, lookup_table_dict):
    """
    @ input : tokenized_dict --> input dict just replace words with int representation
                and lookup_table_dict --> {1, [R = 'Program, F = 12],...}
    @ output: word embedding table
    """
    targets, contexts, labels=generate_training_data(tokenized_subsequence,2,10,len(lookup_table_dict))
    AUTOTUNE = tf.data.AUTOTUNE
    targets = np.array(targets)
    contexts = np.array(contexts)[:,:,0]
    labels = np.array(labels)
    BATCH_SIZE = 1024
    BUFFER_SIZE = 20000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # acclerating the processing speed 
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # constructing embedding table 
    embedding_dim = 100
    # the size of embedding table would be len(lookup_table_dict) * embedding_dim
    word2vec = Word2Vec(len(lookup_table_dict), embedding_dim,10)
    word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    # fitting model
    word2vec.fit(dataset, epochs=5, callbacks=[tensorboard_callback])
    # output embedding table
    return word2vec.get_layer('w2v_embedding').get_weights()[0]
