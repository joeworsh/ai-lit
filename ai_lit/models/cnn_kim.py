"""
Standard CNN implementation for document classification based on
'Convolutional Neural Networks for Sentence Classification' by Yoon Kim, 2014
"""

import tensorflow as tf

tf.flags.DEFINE_string("filters", "3,4,5,6",
                       "String containing a comma-separated list of convolutional filter sizes for the CNN")
tf.flags.DEFINE_integer("num_filters", 128,
                        "The total number of filters for each filter size in the graph.")
tf.flags.DEFINE_integer("embedding_size", 300,
                        "The number of dimensions to use when learning word embeddings from the corpus.")
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "The adam learning rate applied to CNN optimization.")

FLAGS = tf.flags.FLAGS


class CnnKim:
    def __init__(self, term_count, subject_count, doc_length, pretrained_embeddings=None):
        """ Defines the CNN portion of the TensorFlow graph for classification.
        """
        # set up tensorflow placeholders to training runtime
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, subject_count], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.targets = tf.argmax(self.input_y, 1, name="targets")

        # set up a top-level node titled embedding
        with tf.name_scope("embedding"):
            if pretrained_embeddings is None:
                embedding_lookup = tf.Variable(tf.random_uniform([term_count, FLAGS.embedding_size], -1.0, 1.0))
            else:
                embedding_lookup = pretrained_embeddings
            embedded_chars = tf.nn.embedding_lookup(embedding_lookup, self.input_x)
            embedded_chars = tf.expand_dims(embedded_chars, -1)

        # implement the convolutional layers one by one
        # all the layers are then merged into a big feature vector
        pooled_outputs = []
        filters = list(map(int, FLAGS.filters.split(",")))
        for filter_size in filters:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # set up the convolutional layer for this filter size
                filter_shape = tf.TensorShape([filter_size, FLAGS.embedding_size, 1, FLAGS.num_filters])
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, doc_length - int(filter_size) + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = FLAGS.num_filters * len(filters)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # add a dropout layer which is used prevent neurons from "co-adapting"
        # which forces them to learn individually useful features
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # the output layer from max-pooling with dropout can produce a matrix
        # which contains the scores of each class in the CNN
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, subject_count], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[subject_count]), name="b")
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # set up the cross-entropy loss measurements so that network error
        # can be minimized
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # additionally set up an accuracy calculation for training and testing
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.optimize(self.loss)
        self.summaries()

    def optimize(self, loss):
        """ Define the tensors and operations used to optimize the gradients of the graph.
        """
        # define how the the trainer optimizes the network loss function
        # use the builtin Adam optimizer.
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # compute gradients seems to fail on the GPU - bind to CPU for this op
        grads_and_vars = optimizer.compute_gradients(loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def summaries(self):
        """ Add summary operations which are used to write CNN training summaries to
        the TensorBoard.
        """
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.summary_op = tf.summary.merge([loss_summary, acc_summary])
