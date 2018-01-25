'''
A standard flat LSTM to read each word embedding of a piece of text in sequence.
'''

import tensorflow as tf

# flags to configure and define the neural network graph
tf.flags.DEFINE_integer("word_embedding_size", 300,
                        "The number of dimensions to use when learning word embeddings from the corpus.")
tf.flags.DEFINE_integer("doc_embedding_size", 1000,
                        "The number of dimensions to use when learning document embeddings from the corpus.")
tf.flags.DEFINE_float("learning_rate", 1e-3,
                      "The alpha learning rate of the ADAM optimizer.")
tf.flags.DEFINE_integer("l2_constraint", 0.001,
                        "The L2 constraint that is applied to the weights during training.")

FLAGS = tf.flags.FLAGS


class FlatLstm:
    """
    A Neural Network model for classifying document class with a single LSTM layer.
    """

    def __init__(self, term_count, subject_count, doc_length, pretrained_embeddings=None):
        """
        Build the Neural Network Model
        """
        # documents are represented as lists of words pre-encoded into indices
        self.term_count = term_count
        self.subject_count = subject_count
        self.doc_length = doc_length

        # set up tensorflow placeholders to training runtime
        # the batch sizes may be specific to each instance of the run
        # input x is of size [batch_size, batch_doc_size]
        # input_y is of size [batch_size, subject_count]
        with tf.device("/cpu:0"):
            self.input_x = tf.placeholder(tf.int32, [FLAGS.batch_size, doc_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [FLAGS.batch_size, subject_count], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.targets = tf.argmax(self.input_y, 1, name="targets")

            # begin building the term LSTM representation of the documents
            with tf.name_scope("embedding"):
                if pretrained_embeddings is None:
                    embedding_lookup = tf.get_variable("embedding_lookup",
                                                       initializer=tf.random_uniform(
                                                           [term_count + 1, FLAGS.word_embedding_size],
                                                           -1.0, 1.0),
                                                       regularizer=tf.contrib.layers.l2_regularizer(
                                                           FLAGS.l2_constraint))
                else:
                    embedding_lookup = pretrained_embeddings
                embedded_chars = tf.nn.embedding_lookup(embedding_lookup, self.input_x)

            # construct the LSTM
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.doc_embedding_size, reuse=False)
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
            initial_state = rnn_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)
            out, _ = tf.nn.dynamic_rnn(rnn_cell, embedded_chars, initial_state=initial_state, dtype=tf.float32)
            doc_embeddings = out[:, -1, :]

            # add the output layer with softmax
            self.output(doc_embeddings, self.input_y)

            # add training and summary operations to the graph in order for the network to learn
            self.optimize()
            self.summaries()

    def output(self, input_x, input_y):
        """ Build the output of the NN structure with a softmax layer.
        The argument input_x is of shape [batch_size, document_embedding_size].
        The argument input_y is of shape [batch_size, class_num].
        """
        with tf.name_scope("output"):
            w = tf.get_variable("W", [FLAGS.doc_embedding_size, self.subject_count],
                                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_constraint))
            b = tf.get_variable("B", [self.subject_count], initializer=tf.random_uniform_initializer(0, 0.1))
            self.scores = tf.nn.xw_plus_b(input_x, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=input_y)
            self.loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()

        # additionally set up an accuracy calculation for training and testing
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def optimize(self):
        """ Add optimization operations to the graph in order to train the parameters
        of the neural network.
        """
        # define how the the trainer optimizes the network loss function
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step)

    def summaries(self):
        """ Add summary operations which are used to write CNN-RNN training summaries to
        the TensorBoard.
        """
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.summary_op = tf.summary.merge([loss_summary, acc_summary])
