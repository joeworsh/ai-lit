"""
An AI Lit university for training and evaluating the Gutenberg individual chapter text dataset
with the CNN-Kim model.
"""

import numpy as np
import tensorflow as tf

from ai_lit.models import han
from ai_lit.university.gutenberg import gb_h2_chapters_university

# training parameters
tf.flags.DEFINE_float("dropout", 0.5,
                      "The percentage applied for the dropout rate of edges in the graph.")

FLAGS = tf.flags.FLAGS


class GbChaptersHANUniversity(gb_h2_chapters_university.GbH2ChapterUniversity):
    """
    This is an AI Lit university for training the Hiearchical Attention
    Network (HAN) on the Gutenberg h2 chapter text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB HAN university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace, dataset_wkspc)

        # note: future implementations can add weights to the classes to help combat
        # the unbalanced dataset
        self.equal_weights = np.ones(shape=[FLAGS.batch_size], dtype=np.float32)

    def get_model(self, graph):
        """
        Build the HAN model for the GB chapter text exercise
        :param graph: The graph that the model belongs to.
        :return: The HAN model for the training/evaluation session.
        """
        return han.HAN(len(self.vocab), len(self.subjects))

    def perform_training_run(self, session, model, training_batch):
        """
        Perform a training run of the CNN-Kim on the GB chapter text batch.
        :param session: The training session under which the op is performed.
        :param model: The model we are going to train.
        :param training_batch: The batch of labels and bodies to predict.
        :return: summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.sample_weights: self.equal_weights,
            model.inputs: training_batch[1],
            model.word_lengths: np.full([FLAGS.batch_size, self.max_chapter_len], self.max_para_len),
            model.sentence_lengths: np.full([FLAGS.batch_size], self.max_chapter_len),
            model.input_y: training_batch[0],
            model.dropout_keep_prob: FLAGS.dropout}
        _, summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.train_op, model.summary_op, model.global_step, model.loss, model.accuracy, model.targets,
             model.prediction],
            feed_dict)

        return summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions

    def perform_evaluation_run(self, session, model, eval_batch):
        """
        Perform a validation or evaluation run of the CNN-Kim on the GB chapter text batch.
        :param session: The session under which the eval op is performed.
        :param model: The model we are going to evaluate.
        :param eval_batch: The batch of labels, bodies, book_ids and chapter indexes to predict.
        :return: summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.sample_weights: self.equal_weights,
            model.inputs: eval_batch[1],
            model.word_lengths: np.full([FLAGS.batch_size, self.max_chapter_len], self.max_para_len),
            model.sentence_lengths: np.full([FLAGS.batch_size], self.max_chapter_len),
            model.input_y: eval_batch[0],
            model.dropout_keep_prob: 1}
        summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.summary_op, model.loss, model.accuracy, model.targets, model.prediction],
            feed_dict)

        # be sure to record the evaluation run for future book aggregation
        self.record_evaluation_entry(batch_targets, batch_predictions, eval_batch[2], eval_batch[3])

        return summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
