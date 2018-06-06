import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell

from ai_lit.models.model_utils import task_specific_attention, bidirectional_rnn

tf.flags.DEFINE_integer("term_embedding_size", 100,
                        "The number of dimensions to use when learning hidden word embeddings from the corpus.")
tf.flags.DEFINE_integer("sentence_embedding_size", 200,
                        "The number of dimensions to use when learning hidden sentence embeddings from the corpus.")
tf.flags.DEFINE_integer("document_embedding_size", 300,
                        "The number of dimensions to use when learning document embeddings from the corpus.")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      "The maximum gradient normalization which is applied during weight training.")
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "The learning rate applied to HAN optimization.")

FLAGS = tf.flags.FLAGS


class HAN:
    """ Implementation of document classification model described in
      `Hierarchical Attention Networks for Document Classification (Yang et al., 2016)`
      (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)"""

    def __init__(self, term_count, subject_count):
        self.term_count = term_count
        self.subject_count = subject_count

        with tf.variable_scope('tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.input_y = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_y')

            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            (self.document_size,
             self.sentence_size,
             self.word_size) = tf.unstack(tf.shape(self.inputs))
            self.targets = tf.argmax(self.input_y, 1, name="targets")

            self._init_embedding(scope)

            # embeddings cannot be placed on GPU
            with tf.device('/cpu:0'):
                self._init_body(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)

            self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            tf.summary.scalar('loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.targets, 1), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                FLAGS.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.term_count, FLAGS.term_embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                FLAGS.term_embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word') as scope:
                self.word_cell = GRUCell(FLAGS.sentence_embedding_size)
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell, self.word_cell,
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        FLAGS.sentence_embedding_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = tf.nn.dropout(word_level_output, self.dropout_keep_prob)

            # sentence_level

            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, FLAGS.sentence_embedding_size])

            with tf.variable_scope('sentence') as scope:
                self.sentence_cell = GRUCell(FLAGS.document_embedding_size)
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, FLAGS.document_embedding_size, scope=scope)

                with tf.variable_scope('dropout'):
                    sentence_level_output = tf.nn.dropout(sentence_level_output, self.dropout_keep_prob)

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sentence_level_output, self.subject_count, activation_fn=None)

                self.prediction = tf.argmax(self.logits, axis=-1)
