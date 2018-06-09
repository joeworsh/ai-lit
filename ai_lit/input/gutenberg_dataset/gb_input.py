"""
The input reader to read the Gutenberg Project structured Gutenberg dataset.
This dataset can be downloaded from the Gutenberg Project website.
"""

from ai_lit.input import input_util
from glob2 import glob
from zipfile import ZipFile

import collections
import json
import nltk
import os
import random
import sys
import tensorflow as tf
import xml.etree.ElementTree as et

# namespaces used to parse gutenberg index XML files
gutenberg_idx_ns = {'dcterms': 'http://purl.org/dc/terms/',
                    'pgterms': 'http://www.gutenberg.org/2009/pgterms/',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'dcam': 'http://purl.org/dc/dcam/'}

train_index_list = 'train_index_list.json'
test_index_list = 'test_index_list.json'
vocab_file = 'vocab.json'
subject_file = 'subjects.json'
word2vec_model = 'gb_word2vec_100.bin'


class GutenbergIndex(object):
    """
    A class defining the contents of a Gutenberg index file.
    """

    def __init__(self, index, indexfile, title, author, subjects):
        self.index = index
        self.indexfile = indexfile
        self.title = title
        self.author = author
        self.subjects = subjects


class GbIndexEncoder(json.JSONEncoder):
    """
    Special encoder used to write the GutenbergIndex class to JSON.
    """

    def default(self, o):
        if isinstance(o, GutenbergIndex):
            return o.__dict__
        return json.JSONEncoder.default(self, o)


def decode_gb_idx(dct):
    """
    Decoder used to parse GutenbergIndices from JSON.
    :param dct: The JSON object
    :return: The new Gutenberg Index
    """
    if 'index' in dct and 'indexfile' in dct:
        gb_idx = GutenbergIndex(dct['index'], dct['indexfile'], dct['title'], dct['author'], dct['subjects'])
        gb_idx.body = dct['body']
        gb_idx.bodyfile = dct['bodyfile']
        gb_idx.body_length = dct['body_length']
        gb_idx.title_length = dct['title_length']
        return gb_idx
    return dct


def is_input_initialized(workspace):
    """
    Determine if the input workspace is already setup and initialized.
    :param workspace: The workspace directory to check.
    :return: True if it is initialized, false if not.
    """
    train_inputs_path = os.path.join(workspace, train_index_list)
    test_inputs_path = os.path.join(workspace, test_index_list)
    subjects_file_path = os.path.join(workspace, subject_file)
    vocab_file_path = os.path.join(workspace, vocab_file)
    model_path = os.path.join(workspace, word2vec_model)

    return (os.path.exists(train_inputs_path) and
            os.path.exists(test_inputs_path) and
            os.path.exists(subjects_file_path) and
            os.path.exists(vocab_file_path) and
            os.path.exists(model_path))


def compile_input(subjects, dataset_dir, workspace, test_split=0.3):
    """
    Compile and build all input files for the Gutenberg corpus. This method
    will perform all necessary operations to assemble the entire Gutenberg Input.
    :param subjects: The set of subjects to target for the input set.
    :param dataset_dir: The directory where the dataset is stored.
    :param workspace: The workspace directory where assembled input is stored.
    :param test_split: The split between training and test data.
    :return: train_inputs, test_inputs, vocab and word2vec_model
    """
    if tf.gfile.Exists(workspace):
        raise Exception(
            'This directory already exists. Nothing will be assembled. Please delete the directory and try again.')

    body_file_map = map_body_files(dataset_dir)
    train_inputs, test_inputs, vocab = build_inputs(subjects, body_file_map, dataset_dir, workspace, test_split)
    w2v_model = build_word2vec(train_inputs, workspace)
    return train_inputs, test_inputs, vocab, w2v_model


def build_word2vec(gb_indices, workspace):
    """
    Train the provided Gutenberg Indices (can be training, testing or both) with a word2vec model.
    :param gb_indices: The GB indices to train over.
    :param workspace: The workspace to place the word2vec model in.
    :return: The word2vec model.
    """
    sentences = []
    for gb_idx in gb_indices:
        sentences.extend(nltk.sent_tokenize(gb_idx.body))
        sentences.append(gb_idx.title)
    model_path = os.path.join(workspace, word2vec_model)
    return input_util.train_word2vec(sentences, model_path)


def get_inputs(workspace, max_vocab_size):
    """
    Load the inputs from file that contain the known vocabulary of the dataset
    and the set of Gutenberg index files which are relevant to this dataset.
    If the files do not exist, this method will create those files.
    :param workspace:
    The directory of the workspace where all built data is stored.
    :param max_vocab_size:
    The maximum size of the vocabulary which will be kept with this input.
    :return:
    The training input list, the testing input list, the list of included subjects
    and the vocabulary, as a counter, found in this dataset.
    """
    train_inputs = []
    test_inputs = []
    subjects = []
    vocab = collections.Counter()

    train_inputs_path = os.path.join(workspace, train_index_list)
    test_inputs_path = os.path.join(workspace, test_index_list)
    subjects_file_path = os.path.join(workspace, subject_file)
    vocab_file_path = os.path.join(workspace, vocab_file)

    if (os.path.exists(train_inputs_path) and
            os.path.exists(test_inputs_path) and
            os.path.exists(subjects_file_path) and
            os.path.exists(vocab_file_path)):
        with open(train_inputs_path, 'r') as f:
            train_inputs = json.load(f, object_hook=decode_gb_idx)

        with open(test_inputs_path, 'r') as f:
            test_inputs = json.load(f, object_hook=decode_gb_idx)

        with open(subjects_file_path, 'r') as f:
            subjects = json.load(f)

        with open(vocab_file_path, 'r') as f:
            vocab = collections.Counter(json.load(f))

        # only allow a certain amount of words to be present in the vocabulary
        vocab = collections.Counter(dict(vocab.most_common(max_vocab_size)))

    return train_inputs, test_inputs, subjects, vocab


def get_subjects(workspace):
    """
    Retrieve the recorded set of subjects for GB classification.
    :param workspace: The workspace where the subjects are stored.
    :return: The list of subjects. Raises an error if the file does not exist.
    """
    subjects_file_path = os.path.join(workspace, subject_file)
    assert os.path.exists(subjects_file_path)
    with open(subjects_file_path, 'r') as f:
        subjects = json.load(f)
    return subjects


def get_vocabulary(workspace):
    """
    Retrieve the vocabulary, in counter form, for the GB input data.
    :param workspace: The workspace where the vocabulary is stored.
    :return: The counter of the vocabulary terms with their length.
    """
    vocab_file_path = os.path.join(workspace, vocab_file)
    assert os.path.exists(vocab_file_path)
    with open(vocab_file_path, 'r') as f:
        vocab = collections.Counter(json.load(f))
    return vocab


def build_inputs(subjects, body_file_map, dataset_dir, workspace, test_split):
    """ Read through all Gutenberg inputs and build an index file list
    and a vocabulary for all entries which have a subject which is contained
    in the list of desired subjects to be included in this dataset.
    """
    train_idx_path = os.path.join(workspace, train_index_list)
    test_idx_path = os.path.join(workspace, test_index_list)
    vocab_file_path = os.path.join(workspace, vocab_file)
    subject_file_path = os.path.join(workspace, subject_file)

    # initialize the workspace directory for this run
    if tf.gfile.Exists(workspace):
        raise Exception("This directory already exists. Nothing to build.")
    tf.gfile.MakeDirs(workspace)

    train_inputs = []
    test_inputs = []
    vocab = collections.Counter()

    # step through all Gutenberg index files searching for files
    # with the desired subjects and building the known vocabulary
    print("Indexing all Gutenberg Index files.")
    index_files = glob(os.path.join(dataset_dir, 'cache', '**/*.rdf'))
    total_count = len(index_files)
    current_idx = 0
    for idxfile in index_files:
        sys.stdout.write("GB Processing... %s%%\r" % ((current_idx / total_count) * 100))
        sys.stdout.flush()
        with open(idxfile, 'r', encoding='ISO-8859-1') as f:
            idxbody = f.read().encode('utf-8')
            gidx = parse_index_file(idxfile, idxbody)
            if any(s in gidx.subjects for s in subjects):
                if gidx.index in body_file_map:
                    gidx.bodyfile = body_file_map[gidx.index]
                    gidx.body = extract_body(gidx.bodyfile)

                    # determine if this record will belong to the training set or the test set
                    if random.random() < test_split:
                        test_inputs.append(gidx)
                    else:
                        train_inputs.append(gidx)

                        # note: the vocabulary is only created based off of the training inputs, not the test inputs
                        tokens = nltk.word_tokenize(gidx.body)
                        vocab.update(tokens)
                        gidx.body_length = len(tokens)

                        title_tokens = nltk.word_tokenize(gidx.title)
                        gidx.title_length = len(title_tokens)
        current_idx = current_idx + 1

    # write the vocab to file
    with open(vocab_file_path, 'w') as f:
        json.dump(vocab, f, sort_keys=True, indent=4)

    with open(train_idx_path, 'w') as f:
        json.dump(train_inputs, f, cls=GbIndexEncoder, indent=4)

    with open(test_idx_path, 'w') as f:
        json.dump(test_inputs, f, cls=GbIndexEncoder, indent=4)

    with open(subject_file_path, 'w') as f:
        json.dump(subjects, f, indent=4)

    return train_inputs, test_inputs, vocab


def parse_index_file(index_filename, index_body):
    """ Read the given Gutenberg index file and parse all available information.
    Returns a class containing the title, author and subjects.
    """
    # index files are stored in XML, parse out important information
    root = et.fromstring(index_body)
    indexfile = index_filename
    index = os.path.splitext(indexfile)[0]
    index = os.path.basename(index)[2:]

    title_node = root.find('pgterms:ebook/dcterms:title', gutenberg_idx_ns)
    title = title_node.text.lower() if title_node is not None else input_util.UNKNOWN_AUTHOR

    creator_node = root.find('pgterms:ebook/dcterms:creator', gutenberg_idx_ns)
    name_node = creator_node.find('pgterms:agent/pgterms:name', gutenberg_idx_ns) if creator_node is not None else None
    author = name_node.text.lower() if name_node is not None else input_util.UNKNOWN_TITLE

    idxsubjects = []
    for subject in root.findall('pgterms:ebook/dcterms:subject', gutenberg_idx_ns):
        if subject.find('rdf:Description/dcam:memberOf', gutenberg_idx_ns).attrib[
            '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'] == 'http://purl.org/dc/terms/LCSH':
            idxsubjects.append(subject.find('rdf:Description/rdf:value', gutenberg_idx_ns).text)

    return GutenbergIndex(index, indexfile, title, author, idxsubjects)


def map_body_files(data_dir):
    """ Map the found body files in the data directory to the indexes
    that they represent.
    """
    print("Mapping all body files by index.")
    body_files = glob(os.path.join(data_dir, '**/*.zip'), recursive=True)
    print("Searching through", len(body_files), "zip files.")
    body_file_map = {}
    for bodyfile in body_files:
        idx = os.path.splitext(bodyfile)[0]
        idx = os.path.basename(idx)
        body_file_map[idx] = bodyfile
    print("Found", len(body_file_map), "GB files.")
    return body_file_map


def extract_body(bodyfile):
    """ Extract the text content of a Gutenberg works zip file and return it
    as a string.
    """
    with ZipFile(bodyfile, 'r') as z:
        for n in z.namelist():
            return z.open(n).read().decode('ISO-8859-1').lower()
