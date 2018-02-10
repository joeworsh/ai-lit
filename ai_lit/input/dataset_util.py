"""
Set of utilities for working with datasets, TensorFlow, TFRecords and configurations.
"""

import tensorflow as tf

tf.flags.DEFINE_integer("epochs", 10,
                        "Defines the number of epochs for a training run.")
tf.flags.DEFINE_integer("batch_size", 20,
                        "The number of records to process per batch during training.")
tf.flags.DEFINE_integer("batch_queue_capacity", 100,
                        "Define the number records held in memory on the batch queue during processing.")

FLAGS = tf.flags.FLAGS
