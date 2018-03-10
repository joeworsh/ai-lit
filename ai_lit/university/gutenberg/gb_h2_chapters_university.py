"""
An AI Lit university base class for training and evaluating the Gutenberg
individual h2 hierarchical chapter text dataset.
"""

import tensorflow as tf

from ai_lit.input.gutenberg_dataset import gb_h2_chapters_dataset
from ai_lit.university.gutenberg import gb_chapters_university

FLAGS = tf.flags.FLAGS


class GbH2ChapterUniversity(gb_chapters_university.GbChapterUniversity):
    """
    This is an AI Lit university for training models on the Gutenberg individual h2 chapters text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc, max_chapter_len=50, max_para_len=50):
        """
        Initialize the GB H2 chapters university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace, dataset_wkspc)
        self.max_chapter_len = max_chapter_len
        self.max_para_len = max_para_len

    def get_training_data(self):
        """
        Supply the training data for GB chapters text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_h2_chapters_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                       self.vocab, self.max_chapter_len,
                                                                       self.max_para_len)
        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = train_iterator.get_next()
        return labels, bodies, book_ids, chapter_idx

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        testing_dataset = gb_h2_chapters_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                     self.vocab, self.max_chapter_len,
                                                                     self.max_para_len)
        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = test_iterator.get_next()
        return labels, bodies, book_ids, chapter_idx
