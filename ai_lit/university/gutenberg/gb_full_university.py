"""
An AI Lit university base class for training and evaluating the Gutenberg Full text dataset.
"""

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_full_dataset
from ai_lit.university import ai_lit_university

import tensorflow as tf

from enum import Enum

# training parameters
tf.flags.DEFINE_integer("document_length", 5000,
                        "The padding and clipped length of each document for CNN processing.")
tf.flags.DEFINE_integer("vocab_count", 5000,
                        "The top number of vocabulary terms to keep for training.")

FLAGS = tf.flags.FLAGS


class TextWindow(Enum):
    """
    An enumeration to tell the university where it should pull its window of text from the full text body
    of a Gutenberg record.
    """
    beginning = 1
    middle = 2
    end = 3
    random = 4


class GbFullUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training models on the Gutenberg Full text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc, text_window=TextWindow.beginning, starting_idx=0):
        """
        Initialize the GB Full university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        :param text_window: The window (beginning, middle, end) where the text is pulled from.
        :param starting_idx: When using text_window.end, this parameter will define the starting
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace)
        self.dataset_wkspc = dataset_wkspc
        self.subjects = gb_input.get_subjects(self.dataset_wkspc)
        self.vocab = input_util.get_sorted_vocab(gb_input.get_vocabulary(self.dataset_wkspc))
        self.vocab = self.vocab[:FLAGS.vocab_count + 1]
        self.text_window = text_window
        self.starting_idx = starting_idx

    def get_training_data(self):
        """
        Supply the training data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        if self.text_window is TextWindow.beginning:
            training_dataset = gb_full_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, end_index=FLAGS.document_length)
        elif self.text_window is TextWindow.end:
            training_dataset = gb_full_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, start_index=-FLAGS.document_length)
        elif self.text_window is TextWindow.random:
            training_dataset = gb_full_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, random_crop=True,
                                                                    crop_size=FLAGS.document_length)
        else:
            training_dataset = gb_full_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, start_index=self.starting_idx,
                                                                    end_index=self.starting_idx + FLAGS.document_length)

        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies = train_iterator.get_next()
        return labels, bodies

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        if self.text_window is TextWindow.beginning:
            testing_dataset = gb_full_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab, end_index=FLAGS.document_length)
        elif self.text_window is TextWindow.end:
            testing_dataset = gb_full_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab, start_index=-FLAGS.document_length)
        elif self.text_window is TextWindow.random:
            testing_dataset = gb_full_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab, random_crop=True,
                                                                  crop_size=FLAGS.document_length)
        else:
            testing_dataset = gb_full_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab, start_index=self.starting_idx,
                                                                  end_index=self.starting_idx + FLAGS.document_length)

        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies = test_iterator.get_next()
        return labels, bodies
