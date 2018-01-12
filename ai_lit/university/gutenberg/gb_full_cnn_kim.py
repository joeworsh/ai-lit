"""
An AI Lit university for training and evaluating the Gutenberg Full text dataset
with the CNN-Kim model.
"""

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_full_dataset
from ai_lit.models import cnn_kim
from ai_lit.university import ai_lit_university

import tensorflow as tf

# training parameters
tf.flags.DEFINE_integer("document_length", 5000,
                        "The padding and clipped length of each document for CNN processing.")
tf.flags.DEFINE_integer("vocab_count", 5000,
                        "The top number of vocabulary terms to keep for training.")
tf.flags.DEFINE_float("dropout", 0.5,
                      "The percentage applied for the dropout rate of edges in the graph.")
tf.flags.DEFINE_bool("use_pretrained_embeddings", False,
                     "Flag to determine whether or not to use pretrained embeddings.")
tf.flags.DEFINE_string("pretrained_embedding_model", "",
                       "The pretrained embedding model to use if configured to use pretrained embeddings.")

FLAGS = tf.flags.FLAGS


class GbFullCnnKimUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training CNN-Kim on the Gutenberg Full text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB Full CNN Kim university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        """
        super().__init__(model_dir, workspace)
        self.dataset_wkspc = dataset_wkspc
        self.subjects = gb_input.get_subjects(self.dataset_wkspc)
        self.vocab = input_util.get_sorted_vocab(gb_input.get_vocabulary(self.dataset_wkspc))
        self.vocab = self.vocab[:FLAGS.vocab_count]

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

    def get_training_data(self):
        """
        Supply the training data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_full_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                self.vocab, FLAGS.document_length)
        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies = train_iterator.get_next()
        return labels, bodies

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        testing_dataset = gb_full_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                              self.vocab, FLAGS.document_length)
        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies = test_iterator.get_next()
        return labels, bodies

    def perform_training_run(self, session, model, batch_y, batch_x):
        """
        Perform a training run of the CNN-Kim on the GB full text batch.
        :param session: The training session under which the op is performed.
        :param model: The model we are going to train.
        :param batch_y: The batch of labels to predict.
        :param batch_x: The batch of text bodies to classify.
        :return: summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.input_x: batch_x,
            model.input_y: batch_y,
            model.dropout_keep_prob: FLAGS.dropout}
        _, summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.train_op, model.summary_op, model.global_step, model.loss, model.accuracy, model.targets,
             model.predictions],
            feed_dict)

        return summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions

    def perform_evaluation_run(self, session, model, batch_y, batch_x):
        """
        Perform a validation or evaluation run of the CNN-Kim on the GB full text batch.
        :param session: The session under which the eval op is performed.
        :param model: The model we are going to evaluate.
        :param batch_y: The batch of labels to predict.
        :param batch_x: The batch of text bodies to classify.
        :return: summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        feed_dict = {
            model.input_x: batch_x,
            model.input_y: batch_y,
            model.dropout_keep_prob: 1}
        summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = session.run(
            [model.summary_op, model.loss, model.accuracy, model.targets, model.predictions],
            feed_dict)

        return summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
