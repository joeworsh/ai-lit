"""
An AI Lit university for training and evaluating the Gutenberg Full text dataset
with the CNN-Kim model.
"""

from ai_lit.input import input_util
from ai_lit.models import cnn_kim
from ai_lit.university.gutenberg import gb_full_university
from ai_lit.university.gutenberg.gb_full_university import TextWindow

import tensorflow as tf

# training parameters
tf.flags.DEFINE_float("dropout", 0.5,
                      "The percentage applied for the dropout rate of edges in the graph.")
tf.flags.DEFINE_bool("use_pretrained_embeddings", False,
                     "Flag to determine whether or not to use pretrained embeddings.")
tf.flags.DEFINE_string("pretrained_embedding_model", "",
                       "The pretrained embedding model to use if configured to use pretrained embeddings.")

FLAGS = tf.flags.FLAGS


class GbFullCnnKimUniversity(gb_full_university.GbFullUniversity):
    """
    This is an AI Lit university for training CNN-Kim on the Gutenberg Full text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc, text_window=TextWindow.beginning, starting_idx=0):
        """
        Initialize the GB Full CNN Kim university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        :param text_window: The window (beginning, middle, end) where the text is pulled from.
        :param starting_idx: When using text_window.end, this parameter will define the starting
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace, dataset_wkspc, text_window=text_window, starting_idx=starting_idx)

    def get_model(self, graph):
        """
        Build the CNN-Kim model for the GB Full text exercise
        :param graph: The graph that the model belongs to.
        :return: The CNN-Kim model for the training/evaluation session.
        """
        pretrained = None
        if FLAGS.use_pretrained_embeddings:
            # Note: this uses the FLAGS.embedding_size imported from cnn_kim
            pretrained = input_util.get_pretrained_vectors(self.vocab, FLAGS.pretrained_embedding_model,
                                                           FLAGS.embedding_size)
        return cnn_kim.CnnKim(len(self.vocab), len(self.subjects), FLAGS.document_length, pretrained)

    def perform_training_run(self, session, model, training_batch):
        """
        Perform a training run of the CNN-Kim on the GB full text batch.
        :param session: The training session under which the op is performed.
        :param model: The model we are going to train.
        :param training_batch: The batch of labels and bodies to predict.
        :return: summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.input_x: training_batch[1],
            model.input_y: training_batch[0],
            model.dropout_keep_prob: FLAGS.dropout}
        _, summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.train_op, model.summary_op, model.global_step, model.loss, model.accuracy, model.targets,
             model.predictions],
            feed_dict)

        return summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions

    def perform_evaluation_run(self, session, model, eval_batch):
        """
        Perform a validation or evaluation run of the CNN-Kim on the GB full text batch.
        :param session: The session under which the eval op is performed.
        :param model: The model we are going to evaluate.
        :param batch_x: The batch of text bodies to classify.
        :return: summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.input_x: eval_batch[1],
            model.input_y: eval_batch[0],
            model.dropout_keep_prob: 1}
        summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.summary_op, model.loss, model.accuracy, model.targets, model.predictions],
            feed_dict)

        return summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
