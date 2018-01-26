"""
An AI Lit university base class for training and evaluating the Gutenberg title-only dataset.
"""

import tensorflow as tf

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_titles_dataset
from ai_lit.university import ai_lit_university

# training parameters
tf.flags.DEFINE_integer("vocab_count", 5000,
                        "The top number of vocabulary terms to keep for training.")

FLAGS = tf.flags.FLAGS


class GbTitlesUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training models on the Gutenberg titles only dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB Titles university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        """
        super().__init__(model_dir, workspace)
        self.dataset_wkspc = dataset_wkspc
        self.subjects = gb_input.get_subjects(self.dataset_wkspc)
        self.vocab = input_util.get_sorted_vocab(gb_input.get_vocabulary(self.dataset_wkspc))
        self.vocab = self.vocab[:FLAGS.vocab_count + 1]

    def get_training_data(self):
        """
        Supply the training data for GB titles text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_titles_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab)

        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies = train_iterator.get_next()
        return labels, bodies

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB titles text dataset.
        :return: Labels and bodies tensors for input.
        """
        testing_dataset = gb_titles_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                self.vocab)

        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies = test_iterator.get_next()
        return labels, bodies
