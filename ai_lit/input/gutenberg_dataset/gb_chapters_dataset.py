"""
This dataset builds and accesses a TFRecords based dataset which contains the
individual chapters of each book as unique records and the associated atomic genre which
is targeted for the classification task.
"""

import json
import nltk
import os
import tensorflow as tf

from ai_lit.input import input_util
from ai_lit.input import dataset_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_dataset_util

FLAGS = dataset_util.FLAGS

TRAIN_TITLE_RECORDS = 'train_gb_chapters.tfrecords'
TEST_TITLE_RECORDS = 'test_gb_chapters.tfrecrods'
CHAPTER_INDEX_JSON = 'chapter_index.json'

# when building the dataset, this will map the id (index) of a TFRecord back to the book
# title and author to which it belongs
title_map = []


def get_training_dataset(workspace, class_count, vocab, chapter_length):
    """
    Get the GB chapter training dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param class_count: The number of classes to be classified.
    :param vocab: The set of unique terms used for this dataset.
    :param chapter_length: The fixed length of each chapter extracted from the dataset.
    :return: The label and value record tensors from the dataset.
    """
    return get_dataset(workspace, TRAIN_TITLE_RECORDS, class_count, vocab, chapter_length)


def get_testing_dataset(workspace, class_count, vocab, chapter_length):
    """
    Get the GB chapter testing dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param class_count: The number of classes to be classified.
    :param chapter_length: The fixed length of each chapter extracted from the dataset.
    :return: The label and value record tensors from the dataset.
    """
    return get_dataset(workspace, TEST_TITLE_RECORDS, class_count, vocab, chapter_length)


def get_dataset(workspace, tf_file, class_count, vocab, chapter_length):
    """
    Retrieve the defining GB chapter dataset.
    :param workspace: The workspace directory where the TFRecords are kept.
    :param tf_file: The TFRecords file to parse.
    :param class_count: The number of classes to be classified.
    :param chapter_length: The fixed length of each chapter extracted from the dataset.
    :return: The label, value, book_id, and chapter index record tensors from the dataset.
    """

    vocab_cap = len(vocab) - 1

    def _parse_function(example_proto):
        context_features = {
            "y": tf.FixedLenFeature([1], dtype=tf.int64),
            "book_id": tf.FixedLenFeature([1], dtype=tf.int64),
            "chapter_idx": tf.FixedLenFeature([1], dtype=tf.int64)
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
        if chapter_length is not None and chapter_length > 0:
            x = tf.cond(tf.less(chapter_length, tf.shape(x)[0]), lambda: x[:chapter_length], lambda: x)
        x = tf.clip_by_value(x, 0, vocab_cap)
        book_id = context_parsed["book_id"]
        chapter_idx = context_parsed["chapter_idx"]
        return y, x, book_id, chapter_idx

    # filter to ensure only complete batches are included in this dataset
    filter_batch = lambda y, x, book_id, chapter_idx: tf.equal(tf.shape(x)[0], FLAGS.batch_size)

    tf_filepath = os.path.join(workspace, tf_file)
    dataset = tf.contrib.data.TFRecordDataset(tf_filepath)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_queue_capacity)
    dataset = dataset.padded_batch(FLAGS.batch_size, padded_shapes=([class_count], [chapter_length], [1], [1]))
    dataset = dataset.filter(filter_batch)
    dataset = dataset.repeat(FLAGS.epochs)
    return dataset


def extract_dataset(workspace, class_count, vocab, chapter_length, train=True):
    """
    Organize and extract the chapter based dataset into a list of lists of integers (word ids).
    :param train:
    :param chapter_length:
    :param vocab:
    :param class_count:
    :param workspace:
    :return: list of list of ints and the title map from file
    """
    FLAGS.epochs = 1
    FLAGS.batch_size = 1
    records = {}
    with tf.Graph().as_default():
        if train:
            dataset = get_training_dataset(workspace, class_count, vocab, chapter_length)
        else:
            dataset = get_testing_dataset(workspace, class_count, vocab, chapter_length)
        data_iter = dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = data_iter.get_next()
        tf.no_op()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(),
                           tf.tables_initializer())
        with tf.Session() as session:
            init_op.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            try:
                while True:
                    # get next batch from the list
                    batch_y, batch_x, batch_book_ids, batch_chap_idx = session.run(
                        [labels, bodies, book_ids, chapter_idx])
                    for label, body, book_id, chap_idx in zip(batch_y, batch_x, batch_book_ids, batch_chap_idx):
                        if book_id[0] not in records:
                            records[book_id[0]] = {}
                        records[book_id[0]][chap_idx[0]] = body
            except tf.errors.OutOfRangeError:
                print("Training examples exhausted")

            coord.request_stop()
            coord.join(threads)
    sorted_records = {book_id: [records[book_id][chap_idx] for chap_idx in sorted(records[book_id])] for book_id in
                      records}
    chapter_index_file = os.path.join(workspace, CHAPTER_INDEX_JSON)
    with open(chapter_index_file, 'r') as f:
        title_map_file = json.load(f)
    return sorted_records, title_map_file


def compile_dataset(subjects, dataset_dir, workspace, max_vocab_size=5000, test_split=0.3):
    """
    Assemble all needed components in order to build the GB chapters dataset.
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
    :return: nothing
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
        write_chapter_records(gidx, sorted_vocab_list, subjects, train_writer)
    for gidx in test:
        write_chapter_records(gidx, sorted_vocab_list, subjects, test_writer)

    train_writer.close()
    test_writer.close()

    chapter_index_file = os.path.join(workspace, CHAPTER_INDEX_JSON)
    if tf.gfile.Exists(chapter_index_file):
        tf.gfile.Remove(chapter_index_file)
    with open(chapter_index_file, 'w') as f:
        json.dump(title_map, f, indent=4)


def write_chapter_records(gidx, vocab, subjects, tf_writer):
    """
    Write the provided Gutenberg Index into the targeted TFRecords writer.
    :param gidx: The Gutenberg Index to write.
    :param vocab: The set of vocabulary used to index each term.
    :param subjects: The set of subjects targeted by the classification problem.
    :param tf_writer: The TFRecords writer to insert the new record into.
    """
    title_map.append(gidx.title + '+' + gidx.author)
    this_id = len(title_map) - 1
    current_chapter = 0
    for chapter in input_util.parse_chapters(gidx.body):
        terms = [input_util.lookup_word(term, vocab) for term in nltk.word_tokenize(chapter)]

        # chapter must have at least 20 words to count as a chapter
        if len(terms) >= 20:
            y = gb_dataset_util.get_atomic_subject(gidx.subjects, subjects)

            # build the title term sequence example
            ex = tf.train.SequenceExample()
            fl_tokens = ex.feature_lists.feature_list["x"]
            for term in terms:
                fl_tokens.feature.add().int64_list.value.append(term)

            # add the label vector to the example
            ex.context.feature["y"].int64_list.value.append(subjects.index(y))

            # add the reverse lookup of the book id so that chapters can be matched together
            ex.context.feature["book_id"].int64_list.value.append(this_id)

            # add the current chapter number so that vectors can be reassembled at a later time
            ex.context.feature["chapter_idx"].int64_list.value.append(current_chapter)
            current_chapter = current_chapter + 1

            tf_writer.write(ex.SerializeToString())
