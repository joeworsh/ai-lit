"""
An AI Lit university base class for training and evaluating the Gutenberg individual chapter text dataset.
"""

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_chapters_dataset
from ai_lit.university import ai_lit_university

import tensorflow as tf

# training parameters
tf.flags.DEFINE_integer("chapter_length", 2500,
                        "The padding and clipped length of each chapter for processing.")
tf.flags.DEFINE_integer("vocab_count", 5000,
                        "The top number of vocabulary terms to keep for training.")

FLAGS = tf.flags.FLAGS


class GbChapterUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training models on the Gutenberg individual chapters text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB chapters university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace)
        self.dataset_wkspc = dataset_wkspc
        self.subjects = gb_input.get_subjects(self.dataset_wkspc)
        self.vocab = input_util.get_sorted_vocab(gb_input.get_vocabulary(self.dataset_wkspc))
        self.vocab = self.vocab[:FLAGS.vocab_count]

    def get_training_data(self):
        """
        Supply the training data for GB chapters text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_chapters_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, FLAGS.chapter_length)
        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = train_iterator.get_next()
        return labels, bodies

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        testing_dataset = gb_chapters_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                              self.vocab, FLAGS.chapter_length)
        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = test_iterator.get_next()
        return labels, bodies
