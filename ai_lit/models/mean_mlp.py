"""
A variant of a standard Multi-Layer Preceptron (MLP) which begins with an
embedding layer which is mean-reduced for a fully connected layer. These embeddings
can be pre-trained or learned on-line.
"""

import tensorflow as tf

tf.flags.DEFINE_integer("embedding_size", 300,
                        "The number of dimensions to use when learning word embeddings from the corpus.")
tf.flags.DEFINE_integer("h1", 200,
                        "The dimensionality of the h1 hidden layer.")
tf.flags.DEFINE_integer("h2", 100,
                        "The dimensionality of the h2 hidden layer.")
tf.flags.DEFINE_integer("h3", 50,
                        "The dimensionality of the h3 hidden layer.")
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "The adam learning rate applied to CNN optimization.")
tf.flags.DEFINE_integer("l2_constraint", 3,
                        "The L2 constraint for training.")

FLAGS = tf.flags.FLAGS


class MeanMLP:
    """
    An AI_Lit model for flattening a sequence of terms and classifying within an MLP.
    """
    def __init__(self, term_count, subject_count, pretrained_embeddings=None):
        # set up tensorflow placeholders to training runtime
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, subject_count], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.targets = tf.argmax(self.input_y, 1, name="targets")

        # begin to assemble the graph
        with tf.name_scope("embedding"):
            if pretrained_embeddings is None:
                embedding_lookup = tf.Variable(tf.random_uniform([term_count, FLAGS.embedding_size], -1.0, 1.0))
            else:
                embedding_lookup = pretrained_embeddings
            embedded_chars = tf.nn.embedding_lookup(embedding_lookup, self.input_x)

        # perform a reduce mean to get the mean vector of each body sequence
        reduced = tf.reduce_mean(embedded_chars, 1)

        # build the fully connected MLP with ReLU and dropout
        h1 = tf.contrib.layers.fully_connected(reduced, FLAGS.h1)
        h1_dropout = tf.contrib.layers.dropout(h1, self.dropout_keep_prob)

        h2 = tf.contrib.layers.fully_connected(h1_dropout, FLAGS.h2)
        h2_dropout = tf.contrib.layers.dropout(h2, self.dropout_keep_prob)

        h3 = tf.contrib.layers.fully_connected(h2_dropout, FLAGS.h3)
        h3_dropout = tf.contrib.layers.dropout(h3, self.dropout_keep_prob)

        self.scores = tf.contrib.layers.fully_connected(h3_dropout, subject_count, activation_fn=None)
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(FLAGS.l2_constraint))

        # additionally set up an accuracy calculation for training and testing
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # define how the the trainer optimizes the network loss function
        # use the builtin Adam optimizer.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # compute gradients seems to fail on the GPU - bind to CPU for this op
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # define the summaries of this model
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.summary_op = tf.summary.merge([loss_summary, acc_summary])
