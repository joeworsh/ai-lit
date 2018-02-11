"""
An AI Lit university base class for training and evaluating the Gutenberg individual chapter text dataset.
"""

import json
import os

import numpy as np
import tensorflow as tf

from ai_lit.input import input_util
from ai_lit.input.gutenberg_dataset import gb_chapters_dataset
from ai_lit.input.gutenberg_dataset import gb_input
from ai_lit.university import ai_lit_university

# training parameters
tf.flags.DEFINE_integer("chapter_length", 2500,
                        "The padding and clipped length of each chapter for processing.")
tf.flags.DEFINE_integer("vocab_count", 5000,
                        "The top number of vocabulary terms to keep for training.")

FLAGS = tf.flags.FLAGS


class ChapterRecord:
    """
    Collection of chapter records for aggregation evaluations.
    """

    def __init__(self, target, pred, book_id, chap_idx):
        self.target = target
        self.pred = pred
        self.book_id = book_id
        self.chap_idx = chap_idx


class GbChapterUniversity(ai_lit_university.AILitUniversity):
    """
    This is an AI Lit university for training models on the Gutenberg individual chapters text dataset.
    """

    def __init__(self, model_dir, workspace, dataset_wkspc):
        """
        Initialize the GB chapters university.
        :param model_dir: The directory where this model is stored.
        :param workspace: The workspace directory of this university.
        :param dataset_wkspc: The GB input workspace where all inputs are stored.
        index used for extracting the text window.
        """
        super().__init__(model_dir, workspace)
        self.dataset_wkspc = dataset_wkspc
        self.subjects = gb_input.get_subjects(self.dataset_wkspc)
        self.vocab = input_util.get_sorted_vocab(gb_input.get_vocabulary(self.dataset_wkspc))
        self.vocab = self.vocab[:FLAGS.vocab_count]
        self.evaluation_aggrs = {}

    def get_training_data(self):
        """
        Supply the training data for GB chapters text dataset.
        :return: Labels and bodies tensors for input.
        """
        training_dataset = gb_chapters_dataset.get_training_dataset(self.dataset_wkspc, len(self.subjects),
                                                                    self.vocab, FLAGS.chapter_length)
        train_iterator = training_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = train_iterator.get_next()
        return labels, bodies, book_ids, chapter_idx

    def get_evaluation_data(self):
        """
        Supply the evaluation data for GB Full text dataset.
        :return: Labels and bodies tensors for input.
        """
        testing_dataset = gb_chapters_dataset.get_testing_dataset(self.dataset_wkspc, len(self.subjects),
                                                                  self.vocab, FLAGS.chapter_length)
        test_iterator = testing_dataset.make_one_shot_iterator()
        labels, bodies, book_ids, chapter_idx = test_iterator.get_next()
        return labels, bodies, book_ids, chapter_idx

    def record_evaluation_entry(self, batch_targets, batch_predictions, batch_book_ids, batch_chap_idx):
        """
        Record an aggregated entry for batch record for future aggregation and evaluation across chapters belonging
        to the same book.
        :param batch_targets: The list of true targets within the batch
        :param batch_predictions: The list of true predictions within the batch
        :param batch_book_ids: The list of book IDs that each record belongs to
        :param batch_chap_idx: The list of chapter indexes which specify the order of chapters in the books
        :return: nothing
        """
        for target, pred, book_id, chap_idx in zip(batch_targets, batch_predictions, batch_book_ids, batch_chap_idx):
            record = ChapterRecord(target, pred, book_id[0], chap_idx[0])
            if record.book_id not in self.evaluation_aggrs:
                self.evaluation_aggrs[record.book_id] = []
            self.evaluation_aggrs[record.book_id].append(record)

    def evaluate(self, model_checkpoint, evaluation_name, **kwargs):
        """
        An overridden implementation of evaluate which can be configured through the kwargs to
        evaluate an aggregate of each book, instead of individual chapters.
        :param model_checkpoint: The checkpoint directory of a saved model
        :param evaluation_name: The name of the evaluation to perform
        :param kwargs: The kwargs to specify which form of evaluation to run. aggr_books=True to evaluate all books.
        :return:
        """
        chap_targets, chap_predictions = super().evaluate(model_checkpoint, evaluation_name, **kwargs)

        # if no aggregate evaluation is configured, simply return the basic evaluation
        print("Checking validation mode...", kwargs)
        if kwargs is None or "aggr_books" not in kwargs or kwargs["aggr_books"] is False:
            print("Evaluating targets and predictions based on individual chapters.")
            return chap_targets, chap_predictions

        # have requested an aggregate evaluation of books
        print("Evaluating the full book aggregation.")
        book_targets = []
        book_predictions = []
        for records in self.evaluation_aggrs.values():
            if len(records) > 0:
                book_targets.append(records[0].target)
                book_predictions.append(np.bincount([r.pred for r in records]).argmax())

        # must rewrite the target and predictions files to match the new aggregated evaluation
        ckpt_dir, ckpt_restore_dir, eval_dir = self.get_evavluation_workspace(model_checkpoint, evaluation_name)
        eval_targets_file = os.path.join(eval_dir, ai_lit_university.EVAL_TARGETS_FILE)
        book_targets = [int(t) for t in book_targets]
        if tf.gfile.Exists(eval_targets_file):
            tf.gfile.Remove(eval_targets_file)
        with open(eval_targets_file, 'w') as f:
            json.dump(book_targets, f, indent=4)  # note must convert from Int32 class to primitive

        eval_preds_file = os.path.join(eval_dir, ai_lit_university.EVAL_PREDICTIONS_FILE)
        book_predictions = [int(t) for t in book_predictions]
        if tf.gfile.Exists(eval_preds_file):
            tf.gfile.Remove(eval_preds_file)
        with open(eval_preds_file, 'w') as f:
            json.dump(book_predictions, f, indent=4)  # note must convert from Int32 class to primitive

        return book_targets, book_predictions
