"""
The input reader to read the Gutenberg Project structured Gutenberg dataset.
This dataset can be downloaded from the Gutenberg Project website.
"""

from glob2 import glob
from pymaybe import maybe
from zipfile import ZipFile

import collections
import nltk
import os
import pickle
import random
import sys
import tensorflow as tf
import xml.etree.ElementTree as et

# namespaces used to parse gutenberg index XML files
gutenberg_idx_ns = {'dcterms': 'http://purl.org/dc/terms/',
                    'pgterms': 'http://www.gutenberg.org/2009/pgterms/',
                    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'dcam': 'http://purl.org/dc/dcam/'}

train_index_list = 'train_index_list.pkl'
test_index_list = 'test_index_list.pkl'
vocab_file = 'vocab.pkl'
subject_file = 'subjects.pkl'


class GutenbergIndex(object):
    """ A class defining the contents of a Gutenberg index file.
    """

    def __init__(self, index, indexfile, title, author, subjects):
        self.index = index
        self.indexfile = indexfile
        self.title = title
        self.author = author
        self.subjects = subjects


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
        with open(train_inputs_path, 'rb') as f:
            train_inputs = pickle.load(f)

        with open(test_inputs_path, 'rb') as f:
            test_inputs = pickle.load(f)

        with open(subjects_file_path, 'rb') as f:
            subjects = pickle.load(f)

        with open(vocab_file_path, 'rb') as f:
            vocab = pickle.load(f)

        # only allow a certain amount of words to be present in the vocabulary
        vocab = collections.Counter(dict(vocab.most_common(max_vocab_size)))

    return train_inputs, test_inputs, subjects, vocab


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
    print()
    for idxfile in index_files:
        sys.stdout.write("Processing... %s%%\r" % ((current_idx / total_count) * 100))
        with open(idxfile, 'r', encoding='ISO-8859-1') as f:
            idxbody = f.read().encode('utf-8')
            gidx = parse_index_file(idxfile, idxbody, dataset_dir)
            if any(s in gidx.subjects for s in subjects):
                if gidx.index in body_file_map:
                    # determine if this record will belong to the training set or the test set
                    if random.random() < test_split:
                        test_inputs.append(idxfile)
                    else:
                        train_inputs.append(idxfile)

                        # note: the vocabulary is only created based off of the training inputs, not the test inputs
                        gidx.bodyfile = body_file_map[gidx.index]
                        gidx.body = extract_body(gidx.bodyfile)
                        vocab.update(nltk.word_tokenize(gidx.body))
                else:
                    print("Could not find a body file for index " + str(gidx.index) + "\r\n")

    # write the vocab to file
    with open(vocab_file_path, 'wb') as f:
        pickle.dump(vocab, f)

    with open(train_idx_path, 'wb') as f:
        pickle.dump(train_inputs, f)

    with open(test_idx_path, 'wb') as f:
        pickle.dump(test_inputs, f)

    with open(subject_file_path, 'wb') as f:
        pickle.dump(subjects, f)

    return train_inputs, test_inputs, vocab


def parse_index_file(index_filename, index_body, data_dir):
    """ Read the given Gutenberg index file and parse all available information.
    Returns a class containing the title, author and subjects.
    """
    # index files are stored in XML, parse out important information
    root = et.fromstring(index_body)
    indexfile = index_filename
    index = os.path.splitext(indexfile)[0]
    index = os.path.basename(index)[2:]
    title = maybe(root.find('pgterms:ebook/dcterms:title', gutenberg_idx_ns)).text
    author = maybe(
        maybe(root.find('pgterms:ebook/dcterms:creator', gutenberg_idx_ns)).find('pgterms:agent/pgterms:name',
                                                                                 gutenberg_idx_ns)).text
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
    body_files = tf.gfile.Glob(os.path.join(data_dir, 'www.gutenberg.lib.md.us', '**/*.zip'))
    body_file_map = {}
    for bodyfile in body_files:
        idx = os.path.splitext(bodyfile)[0]
        idx = os.path.basename(idx)
        body_file_map[idx] = bodyfile
    return body_file_map


def extract_body(bodyfile):
    """ Extract the text content of a Gutenberg works zip file and return it
    as a string.
    """
    with ZipFile(bodyfile, 'r') as z:
        for n in z.namelist():
            return z.open(n).read().decode('ISO-8859-1').lower()
