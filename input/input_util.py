"""
Set of utilities for the ai_lit input packages.
"""

from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import tensorflow as tf

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def get_pretrained_vectors(vocab, pretrained_model, embedding_length, binary=True):
    """ Load a pre-trained vector model of the vocabulary and return a lookup
  tensorflow constant for the desired vocab words with the pretrained vectors.
  """
    word_vectors = KeyedVectors.load_word2vec_format(pretrained_model, binary=binary)
    np_model = np.array([]);
    for word in vocab:
        if word in word_vectors:
            np_model = np.append(np_model, word_vectors[word])
        else:
            np_model = np.append(np_model, np.array(0 * embedding_length))
    return tf.constant(
        np_model, dtype=tf.float32, shape=[len(vocab), embedding_length], name="embedding_weights")


def get_vocab_lookup(vocab):
    """ Return an index-based tensor vocab lookup which can be used
    to turn a sequence of terms into indexes for training/inference.
    """
    sorted_vocab_list = sorted(set(vocab.elements()))
    sorted_vocab_list = [term for term in sorted_vocab_list if term is not None]
    sorted_vocab_list.insert(0, PAD_TOKEN)
    sorted_vocab_list.insert(1, UNK_TOKEN)
    sorted_vocab_list = tf.constant(sorted_vocab_list)
    vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
        mapping=sorted_vocab_list, default_value=0)
    return vocab_lookup


def lookup_word(word, sorted_vocab_list):
    """ Look up the index of the vocab word.
    """
    if word in sorted_vocab_list:
        return sorted_vocab_list.index(word)
    else:
        return sorted_vocab_list.index(UNK_TOKEN)
