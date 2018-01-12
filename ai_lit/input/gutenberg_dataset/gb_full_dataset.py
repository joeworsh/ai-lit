"""
This dataset builds and accesses a TFRecords based dataset which contains the
unbounded full content of each book as a sequence of indices and the associated atomic genre which
is targeted for the classification task.
"""

import nltk
import os
import tensorflow as tf

from ai_lit.input import input_util
from ai_lit.input import dataset_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_dataset_util

FLAGS = dataset_util.FLAGS

TRAIN_TITLE_RECORDS = 'train_gb_full.tfrecords'
TEST_TITLE_RECORDS = 'test_gb_full.tfrecrods'


def get_training_dataset(workspace, class_count, doc_length=None):
    """
    Get the GB full training dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param class_count: The number of classes to be classified.
    :param doc_length: The set document length of the full dataset.
    :return: The label and value record tensors from the dataset.
    """
    return get_dataset(workspace, TRAIN_TITLE_RECORDS, class_count)


def get_testing_dataset(workspace, class_count, doc_length=None):
    """
    Get the GB full testing dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param class_count: The number of classes to be classified.
    :param doc_length: The set document length of the full dataset.
    :return: The label and value record tensors from the dataset.
    """
    return get_dataset(workspace, TEST_TITLE_RECORDS, class_count)


def get_dataset(workspace, tf_file, class_count, doc_length=None):
    """
    Retrieve the defining GB full dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param tf_file: The TFRecords file to parse.
    :param class_count: The number of classes to be classified.
    :param doc_length: The set document length of the full dataset.
    :return: The label and value record tensors from the dataset.
    """

    def _parse_function(example_proto):
        context_features = {
            "y": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        seq_features = {
            'x': tf.FixedLenSequenceFeature([], tf.int64)
        }
        context_parsed, seq_parsed = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features=context_features,
            sequence_features=seq_features)
        y = tf.one_hot(context_parsed["y"][0], class_count, dtype=tf.int64)
        x = seq_parsed["x"]
        if doc_length is not None:
            x = x[:doc_length]
        return y, x

    # filter to ensure only complete batches are included in this dataset
    filter_batch = lambda y, x: tf.equal(tf.shape(x)[0], FLAGS.batch_size)

    tf_filepath = os.path.join(workspace, tf_file)
    dataset = tf.contrib.data.TFRecordDataset(tf_filepath)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_queue_capacity)
    if doc_length is not None:
        dataset = dataset.padded_batch(FLAGS.batch_size, padded_shapes=([class_count], [doc_length]))
    else:
        dataset = dataset.padded_batch(FLAGS.batch_size, padded_shapes=([class_count], [-1]))
    dataset = dataset.filter(filter_batch)
    dataset = dataset.repeat(FLAGS.epochs)
    return dataset


def compile_dataset(subjects, dataset_dir, workspace, max_vocab_size=5000, test_split=0.3):
    """
    Assemble all needed components in order to build the GB full dataset.
    :param subjects: The subjects to target.
    :param dataset_dir: The directory of the dataset.
    :param workspace: The directory of the workspace where all content is stored.
    :param max_vocab_size: The maximum number of vocabulary to keep in the dataset.
    :param test_split: The split between training and test.
    :return: Nothing.
    """
    if not gb_input.is_input_initialized(workspace):
        if os.path.exists(workspace):
            tf.gfile.DeleteRecursively(workspace)
        train_inputs, test_inputs, vocab, _ = gb_input.compile_input(subjects, dataset_dir, workspace, test_split)
    else:
        train_inputs, test_inputs, _, vocab = gb_input.get_inputs(workspace, max_vocab_size)

    build_tf_records(train_inputs, test_inputs, vocab, subjects, workspace)


def build_tf_records(train, test, vocab, subjects, workspace):
    """
    Construct the TFRecords file for the provided train and test breakout of GutenbergIndices.
    :param train: List of GutenbergIndex training records.
    :param test: List of GutenbergIndex testing records.
    :param vocab: The counter of vocabulary found from this corpus.
    :param subjects: The set of classes which are targeted in these records.
    :param workspace: The directory path where the records will be stored.
    :return:
    """

    # ensure the TF records get written to the current workspace
    train_records_file = os.path.join(workspace, TRAIN_TITLE_RECORDS)
    if tf.gfile.Exists(train_records_file):
        tf.gfile.Remove(train_records_file)
    train_writer = tf.python_io.TFRecordWriter(train_records_file)

    test_records_file = os.path.join(workspace, TEST_TITLE_RECORDS)
    if tf.gfile.Exists(test_records_file):
        tf.gfile.Remove(test_records_file)
    test_writer = tf.python_io.TFRecordWriter(test_records_file)

    sorted_vocab_list = input_util.get_sorted_vocab(vocab)

    # write a record for each book title into the training set or test based on the distribution
    for gidx in train:
        write_record(gidx, sorted_vocab_list, subjects, train_writer)
    for gidx in test:
        write_record(gidx, sorted_vocab_list, subjects, test_writer)

    train_writer.close()
    test_writer.close()


def write_record(gidx, vocab, subjects, tf_writer):
    """
    Write the provided Gutenberg Index into the targeted TFRecords writer.
    :param gidx: The Gutenberg Index to write.
    :param vocab: The set of vocabulary used to index each term.
    :param subjects: The set of subjects targeted by the classification problem.
    :param tf_writer: The TFRecords writer to insert the new record into.
    """
    terms = [input_util.lookup_word(term, vocab) for term in nltk.word_tokenize(gidx.body)]
    y = gb_dataset_util.get_atomic_subject(gidx.subjects, subjects)

    # build the title term sequence example
    ex = tf.train.SequenceExample()
    fl_tokens = ex.feature_lists.feature_list["x"]
    for term in terms:
        fl_tokens.feature.add().int64_list.value.append(term)

    # add the label vector to the example
    ex.context.feature["y"].int64_list.value.append(subjects.index(y))

    tf_writer.write(ex.SerializeToString())
