"""
ai_lit_university is a base class for all training modules within the
ai_lit project. The university is designed to host training and validation
runs and allow specific implementations to provide models, datasets, feed dictionaries
and other runtime specific operations.
"""

from IPython.display import clear_output

import datetime
import json
import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS

DATETIME_FORMAT = "%Y%m%d-%H%M%S"
EVAL_TARGETS_FILE = "evaluation_targets.json"
EVAL_PREDICTIONS_FILE = "evaluation_predictions.json"
FLAG_CONFIG_FILE = "tf_configuration.json"


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
        :param save_rate: The rate at which the model saves checkpoints to disk. If None, will only be saved at end.
        :param eval_rate: The rate at which the model is evaluated with the validation set, if a validation set is provided.
        If None, will never be evaluated.
        :return: The run name which was trained. This is timestamp of the run in the model workspace.
        """
        run_name, cpt_file, cpt_dir, train_dir, test_dir = self.init_workspace()
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
                training_step = 0
                try:
                    while True:
                        # get next batch from the list
                        batch_y, batch_x = session.run([labels, bodies])

                        summary, step, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_training_run(
                            session, model, batch_y, batch_x)
                        training_step = step
                        time_str = datetime.datetime.now().isoformat()
                        clear_output(True)
                        print("{}: Training step {}".format(time_str, step))
                        print('Targets:')
                        print(batch_targets)
                        print('Predictions:')
                        print(batch_predictions)
                        print("Loss {:g}, Acc {:g}".format(batch_loss, batch_accuracy))
                        print()
                        train_writer.add_summary(summary, step)

                        # Save the variables to disk at the configured rate
                        if save_rate is not None and step % save_rate == 0:
                            saver.save(session, cpt_file, global_step=step)

                        if eval_rate is not None and v_labels is not None and v_bodies is not None and step % eval_rate == 0:
                            batch_y, batch_x = session.run([v_labels, v_bodies])
                            summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_evaluation_run(
                                session, model, batch_y, batch_x)
                            clear_output(True)
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

                # save a final version of the model to disk
                saver.save(session, cpt_file, global_step=training_step)

                # close all resources before the session ends
                print("Shutting down all network threads")
                train_writer.close()
                validation_writer.close()
                coord.request_stop()
                coord.join(threads)

        return run_name

    def evaluate(self, model_checkpoint, evaluation_name):
        """
        Evaluate the model in this university using the parameters stored in the provided checkpoint.
        :param model_checkpoint: The directory where the checkpoint to evaluate is stored.
        :param evaluation_name: The name of the evaluation to perform. This allows multiple evaluations against the
        same trained model.
        :return: The set of total targets and the associated predictions. Can be used for further analysis.
        """
        FLAGS.epochs = 1
        targets = []
        predictions = []
        ckpt_dir = os.path.join(self.workspace, self.model_dir, model_checkpoint)
        ckpt_restore_dir = os.path.join(ckpt_dir, "checkpoints")
        assert os.path.exists(ckpt_dir)
        assert os.path.exists(ckpt_restore_dir)

        # create the evaluation directory for the evaluation to perform
        eval_dir = os.path.join(ckpt_dir, evaluation_name)
        assert not os.path.exists(eval_dir)
        tf.gfile.MakeDirs(eval_dir)

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
            ckpt = tf.train.get_checkpoint_state(ckpt_restore_dir)

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
                        batch_y, batch_x = session.run([labels, bodies])
                        summary, batch_loss, batch_accuracy, batch_targets, batch_predictions = self.perform_evaluation_run(
                            session, model, batch_y, batch_x)
                        targets.extend(batch_targets)
                        predictions.extend(batch_predictions)

                        clear_output(True)
                        print("Validation Step")
                        print('Targets:')
                        print(batch_targets)
                        print('Predictions:')
                        print(batch_predictions)
                        print("Loss {:g}, Acc {:g}".format(batch_loss, batch_accuracy))
                        print()

                except tf.errors.OutOfRangeError:
                    print("Testing examples exhausted")

                # close all resources before the session ends
                print("Shutting down all network threads")
                coord.request_stop()
                coord.join(threads)

        targets = [int(t) for t in targets]
        eval_targets_file = os.path.join(eval_dir, EVAL_TARGETS_FILE)
        with open(eval_targets_file, 'w') as f:
            json.dump(targets, f, indent=4)  # note must convert from Int32 class to primitive

        predictions = [int(p) for p in predictions]
        eval_preds_file = os.path.join(eval_dir, EVAL_PREDICTIONS_FILE)
        with open(eval_preds_file, 'w') as f:
            json.dump(predictions, f, indent=4)  # note must convert from Int32 class to primitive

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
        time_str = datetime.datetime.now().strftime(DATETIME_FORMAT)
        run_dir = os.path.join(out_dir, time_str)
        tf.gfile.MakeDirs(run_dir)
        cpt_dir = os.path.join(run_dir, 'checkpoints')
        if not tf.gfile.Exists(cpt_dir):
            tf.gfile.MakeDirs(cpt_dir)
        cpt_file = os.path.join(cpt_dir, 'model')
        train_dir = os.path.join(run_dir, 'train')
        test_dir = os.path.join(run_dir, 'test')

        # write the flags configuration to file
        config_file = os.path.join(run_dir, FLAG_CONFIG_FILE)
        with open(config_file, 'w') as f:
            json.dump(FLAGS.__dict__['__flags'], f, indent=4)

        return time_str, cpt_file, cpt_dir, train_dir, test_dir

    def get_latest_run_dir(self):
        """
        Get the latest run of this university model, if it exists. If it does not exist, returns None.
        This will also ensure that the latest run has a matching TensorFlow configuration as the current
        configuration loaded into the university.
        :return: The latest run checkpoint directory, or none.
        """
        latest_run = None
        if tf.gfile.Exists(self.workspace):
            run_dir = os.path.join(self.workspace, self.model_dir)
            if tf.gfile.Exists(run_dir):
                all_runs = sorted(os.listdir(run_dir), key=lambda x: datetime.datetime.strptime(x, DATETIME_FORMAT),
                                  reverse=True)
                if len(all_runs) > 0:
                    print("Found", len(all_runs), "runs. Looking for one with a matching TensorFlow configuration.")
                    current_config = FLAGS.__dict__['__flags']
                    for run in all_runs:
                        current_config_path = os.path.join(run_dir, run, FLAG_CONFIG_FILE)
                        if tf.gfile.Exists(current_config_path):
                            with open(current_config_path, 'r') as f:
                                run_config = json.load(f)
                            if current_config == run_config:
                                print("Run", run,
                                      "is the latest run with a matching configuration. Selecting this one.")
                                latest_run = run
                                break
        return latest_run

    def get_evaluation(self, model_checkpoint, evaluation_name):
        """
        If an evaluation run has already been saved to the checkpoint workspace, load the targets and precisions.
        Returns None, None if an evaluation does not exist.
        :param model_checkpoint: The checkpoint directory to check for an evaluation.
        :param evaluation_name: The name of the evaluation to load. This allows multiple evaluations against the
        same trained model.
        :return: targets, predictions or None, None
        """
        ckpt_dir = os.path.join(self.workspace, self.model_dir, model_checkpoint)
        assert os.path.exists(ckpt_dir)
        targets = None
        predictions = None

        eval_dir = os.path.join(ckpt_dir, evaluation_name)
        if os.path.exists(eval_dir):
            eval_targets_file = os.path.join(eval_dir, EVAL_TARGETS_FILE)
            eval_preds_file = os.path.join(eval_dir, EVAL_PREDICTIONS_FILE)
            if os.path.exists(eval_targets_file) and os.path.exists(eval_preds_file):
                with open(eval_targets_file, 'r') as f:
                    targets = json.load(f)
                with open(eval_preds_file, 'r') as f:
                    predictions = json.load(f)

        return targets, predictions

    def get_or_perform_evaluation(self, model_checkpoint, evaluation_name):
        """
        Helper method to retrieve an evaluation (or perform one if necessary) on the provided checkpoint directory.
        :param model_checkpoint: The checkpoint directory to retrieve an evaluation for.
        :param evaluation_name: The name of the evaluation to load or perform. This allows multiple evaluations
        against the same trained model.
        :return: targets, predictions
        """
        targets, predictions = self.get_evaluation(model_checkpoint, evaluation_name)
        if targets is None or predictions is None:
            print("Could not find a saved run in the checkpoint directory. Will perform evaluation now.")
            targets, predictions = self.evaluate(model_checkpoint, evaluation_name)
        return targets, predictions
