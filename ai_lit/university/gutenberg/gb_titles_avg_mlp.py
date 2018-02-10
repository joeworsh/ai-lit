"""
An AI Lit university for training and evaluating the Gutenberg titles dataset
where the titles are represented by each word and then averaged together for MLP
classification.
"""

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.input.gutenberg_dataset import gb_titles_dataset
from ai_lit.models import mean_mlp
from ai_lit.university import ai_lit_university

import tensorflow as tf

tf.flags.DEFINE_bool("use_pretrained_embeddings", False,
                     "Flag to determine whether or not to use pretrained embeddings.")
tf.flags.DEFINE_string("pretrained_embedding_model", "",
                       "The pretrained embedding model to use if configured to use pretrained embeddings.")

FLAGS = tf.flags.FLAGS


class GbTitleAvgMlpUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training MeanMLP on the Gutenberg Titles text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB Titles Avg Mean university.
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
        Build the MeanMLP model for the GB title text exercise
        :param graph: The graph that the model belongs to.
        :return: The MeanMLP model for the training/evaluation session.
        """
        pretrained = None
        if FLAGS.use_pretrained_embeddings:
            # Note: this uses the FLAGS.embedding_size imported from mean_mlp
            pretrained = input_util.get_pretrained_vectors(self.vocab, FLAGS.pretrained_embedding_model,
                                                           FLAGS.embedding_size)
        return mean_mlp.MeanMLP(len(self.vocab), len(self.subjects), pretrained)

    def get_training_data(self):
        """
        Supply the training data for GB title text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_titles_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects), self.vocab)
        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies = train_iterator.get_next()
        return labels, bodies

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB title text dataset.
        :return: Labels and bodies tensors for input.
        """
        eval_dataset = gb_titles_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects), self.vocab)
        eval_iterator = eval_dataset.make_one_shot_iterator()
        v_labels, v_bodies = eval_iterator.get_next()
        return v_labels, v_bodies

    def perform_training_run(self, session, model, training_batch):
        """
        Perform a training run of the MeanMLP on the GB title text batch.
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
        Perform a validation or evaluation run of the MeanMLP on the GB title text batch.
        :param session: The session under which the eval op is performed.
        :param model: The model we are going to evaluate.
        :param eval_batch: The batch of labels and bodies to predict.
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
