"""
Set of utilities for the ai_lit input packages.
"""

from gensim.models.keyedvectors import KeyedVectors

import gensim
import numpy as np
import re
import tensorflow as tf

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

UNKNOWN_AUTHOR = "unknown"
UNKNOWN_TITLE = "unknown"

# regexes used for splitting and parsing text body files
para_split = re.compile(r"\n+")
chapter_split = re.compile(r"^\s*chapter\s+.*$", re.IGNORECASE | re.MULTILINE)
part_split = re.compile(r"^\s*part\s+.*$", re.IGNORECASE | re.MULTILINE)
section_split = re.compile(r"^\s*section\s+.*$", re.IGNORECASE | re.MULTILINE)
num_section_split = re.compile(r"^\s*[0-9]+\.?.*$", re.IGNORECASE | re.MULTILINE)
rom_num_section_split = re.compile(r"^\s*[ivxlc]+\.?.*$", re.IGNORECASE | re.MULTILINE)


def train_word2vec(corpus, trained_model_file, dimensions=100, binary=True):
    """
    Train a new word2vec model for the provided text content.
    Note: this method is really slow.
    :param corpus: The list of sentences (or titles or etc.) of text content.
    :param trained_model_file: The file to write the word2vec model to.
    :param dimensions: The dimensions of the trained vectors.
    :param binary: True to write in the C-based binary format. False to use pickle.
    :return: The word2vec model for the corpus.
    """
    model = gensim.models.Word2Vec(corpus, size=dimensions)
    model.wv.save_word2vec_format(trained_model_file, binary=binary)
    return model


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


def get_vocab_lookup(sorted_vocab_list):
    """ Return an index-based tensor vocab lookup which can be used
    to turn a sequence of terms into indexes for training/inference.
    """
    vocab_lookup_map = tf.constant(sorted_vocab_list)
    vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
        mapping=vocab_lookup_map, default_value=0)
    return vocab_lookup


def get_sorted_vocab(vocab_counter):
    """
    Convert a counter-based vocabulary into a sorted vocab list (with special tokens added).
    :param vocab_counter: The counter vocabulary to convert.
    :return: A list of the vocabulary for inclusion in ML solutions.
    """
    sorted_vocab_list = sorted(set(vocab_counter.elements()))
    sorted_vocab_list = [term for term in sorted_vocab_list if term is not None]
    sorted_vocab_list = sorted(sorted_vocab_list)
    if PAD_TOKEN not in sorted_vocab_list:
        sorted_vocab_list.insert(0, PAD_TOKEN)
    if UNK_TOKEN not in sorted_vocab_list:
        sorted_vocab_list.insert(1, UNK_TOKEN)
    return sorted_vocab_list


def lookup_word(word, sorted_vocab_list):
    """ Look up the index of the vocab word.
    """
    if word in sorted_vocab_list:
        return sorted_vocab_list.index(word)
    else:
        return sorted_vocab_list.index(UNK_TOKEN)


def parse_chapters(body):
    """
    Parse the provided body file into chapters and return the chapters.
    """
    if re.search(chapter_split, body):
        return re.split(chapter_split, body)
    elif re.search(part_split, body):
        return re.split(part_split, body)
    elif re.search(section_split, body):
        return re.split(section_split, body)
    elif re.search(num_section_split, body):
        return re.split(num_section_split, body)
    elif re.search(rom_num_section_split, body):
        return re.split(rom_num_section_split, body)
    else:
        return [body]

def parse_paragraphs(body):
    """
    Parse the provided text by paragraphs. In this case, new lines.
    """
    return re.split(para_split, body)