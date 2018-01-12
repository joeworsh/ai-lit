"""
ai_lit_university is a base class for all training modules within the
ai_lit project. The university is designed to host training and validation
runs and allow specific implementations to provide models, datasets, feed dictionaries
and other runtime specific operations.
"""

from IPython.display import clear_output

import datetime
import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class AILitUniversity:
    """
    Base class for all ai_lit training.
    """

    def __init__(self, model_dir, workspace):
        self.model_dir = model_dir
        self.workspace = workspace

    def get_model(self, graph):
        """
        Return the constructed model to use within this university.
        :param graph: The TensorFlow graph which owns this model.
        :return: The constructed model under the provided graph.
        """
        raise NotImplementedError("All universities must implement this method.")

    def get_training_data(self):
        """
        :return: The label tensor and record tensor for the university training set.
        """
        raise NotImplementedError("All universities must implement this method.")

    def get_validation_data(self):
        """
        :return: The label tensor and record tensor for the university validation set. Optional.
        """
        raise NotImplementedError("Universities may implement this method to provide online validation evaluation.")

    def get_evaluation_data(self):
        """
        :return: The label tensor and record tensor for the university evaluation set..
        """
        raise NotImplementedError("All universities must implement this method.")

    def perform_training_run(self, session, model, batch_y, batch_x):
        """
        Perform a training run of the provided model.
        :param session: The TensorFlow session where this run is held.
        :param model: The university model to run.
        :param batch_y: The batch of labels which we are targeting.
        :param batch_x: The batch of records which we are classifying
        :return: The following output from the model: summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        raise NotImplementedError("All universities must implement this method.")

    def perform_evaluation_run(self, session, model, batch_y, batch_x):
        """
        Perform a validation or evaluation run of the provided model.
        :param session: The TensorFlow session where this run is held.
        :param model: The university model to run.
        :param batch_y: The batch of labels which we are targeting.
        :param batch_x: The batch of records which we are classifying
        :return: The following output from the model: summary, batch_loss, batch_accuracy, batch_targets, batch_predictions
        """
        raise NotImplementedError("All universities must implement this method.")

    def train(self, save_rate=10, eval_rate=10):
        """
        Train the model on the designated dataset.
        :param save_rate: The rate at which the model saves checkpoints to disk.
        :param eval_rate: The rate at which the model is evaluated with the validation set, if a validation set is provided.
        """
        cpt_file, cpt_dir, train_dir, test_dir = self.init_workspace()
        with tf.Graph().as_default() as tf_graph:
            labels, bodies = self.get_training_data()
            try:
                v_labels, v_bodies = self.get_validation_data()
            except NotImplementedError:
                v_labels, v_bodies = None, None

            model = self.get_model(tf_graph)

            # The op for initializing the variables.
            print("Initializing all network variables")
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer(),
                               tf.tables_initializer())

            # start the training session
            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
                train_writer = tf.summary.FileWriter(train_dir, session.graph)
                validation_writer = tf.summary.FileWriter(test_dir, session.graph)

                # We must initialize all variables and threads before we use them.
                print("Starting threads and initializers")
                init_op.run()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session, coord=coord)

                # run until the inputs are exhausted
                print("Begin training")
                try:
                    while True:
                        # get next batch from the list
                        batch_y, batch_x = session.run([labels, bodies])

                        summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_training_run(
                            session, model, batch_y, batch_x)
                        time_str = datetime.datetime.now().isoformat()
                        clear_output()
                        print("{}: Training step {}".format(time_str, step))
                        print('Targets:')
                        print(batch_targets)
                        print('Predictions:')
                        print(batch_predictions)
                        print("Loss {:g}, Acc {:g}".format(batch_loss, batch_accuracy))
                        print()
                        train_writer.add_summary(summary, step)

                        # Save the variables to disk at the configured rate
                        if step % save_rate == 0:
                            saver.save(session, cpt_file, global_step=step)

                        if v_labels is not None and v_bodies is not None and step % eval_rate == 0:
                            batch_y, batch_x = session.run([v_labels, v_bodies])
                            summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_evaluation_run(
                                session, model, batch_y, batch_x)
                            clear_output()
                            print("Validation Step")
                            print('Targets:')
                            print(batch_targets)
                            print('Predictions:')
                            print(batch_predictions)
                            print("Loss {:g}, Acc {:g}".format(batch_loss, batch_accuracy))
                            print()
                            validation_writer.add_summary(summary, step)

                except tf.errors.OutOfRangeError:
                    print("Training examples exhausted")

                # close all resources before the session ends
                print("Shutting down all network threads")
                train_writer.close()
                validation_writer.close()
                coord.request_stop()
                coord.join(threads)

    def evaluate(self, model_checkpoint):
        """
        Evaluate the model in this university using the parameters stored in the provided checkpoint.
        :param model_checkpoint: The directory where the checkpoint to evaluate is stored.
        :return: The set of total targets and the associated predictions. Can be used for further analysis.
        """
        targets = []
        predictions = []

        with tf.Graph().as_default() as tf_graph:
            labels, bodies = self.get_evaluation_data()
            model = self.get_model(tf_graph)

            # The op for initializing the variables.
            print("Initializing all network variables")
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer(),
                               tf.tables_initializer())

            # Load the checkpointed model into the univserity
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_checkpoint)

            # start the evaluation session and load the latest checkpoint
            with tf.Session() as session:
                # We must initialize all variables and threads before we use them.
                init_op.run()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session, coord=coord)

                print("Restoring the graph checkpoint.")
                saver.restore(session, ckpt.model_checkpoint_path)

                # run until the inputs are exhausted
                print("Begin evaluation")
                try:
                    while True:
                        summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_run(
                            model, labels, bodies, False)
                        targets.extend(batch_targets)
                        predictions.extend(batch_predictions)

                except tf.errors.OutOfRangeError:
                    print("Testing examples exhausted")

                # close all resources before the session ends
                print("Shutting down all network threads")
                coord.request_stop()
                coord.join(threads)

        return targets, predictions

    def init_workspace(self):
        """
        Set up the needed workspace to initialize and train a model.
        """
        print("Initializing workspace")
        if not tf.gfile.Exists(self.workspace):
            tf.gfile.MakeDirs(self.workspace)

        out_dir = os.path.join(self.workspace, self.model_dir)
        if not tf.gfile.Exists(out_dir):
            tf.gfile.MakeDirs(out_dir)
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(out_dir, time_str)
        tf.gfile.MakeDirs(run_dir)
        cpt_dir = os.path.join(run_dir, 'checkpoints')
        if not tf.gfile.Exists(cpt_dir):
            tf.gfile.MakeDirs(cpt_dir)
        cpt_file = os.path.join(cpt_dir, 'model')
        train_dir = os.path.join(run_dir, 'train')
        test_dir = os.path.join(run_dir, 'test')

        # write the flags configuration to file
        config_file = os.path.join(run_dir, 'configuration.txt')
        with open(config_file, 'w+') as f:
            f.write(str(FLAGS.__dict__['__flags']))

        return cpt_file, cpt_dir, train_dir, test_dir
