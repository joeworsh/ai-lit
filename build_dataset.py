#!/usr/bin/env python3

import nltk
import sys

from ai_lit.input.gutenberg_dataset import gb_chapters_dataset, gb_titles_dataset, gb_full_dataset, gb_h2_chapters_dataset

# need the punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

subjects = [
    "Science fiction",
    "Adventure stories",
    "Historical fiction",
    "Love stories",
    "Detective and mystery stories",
    "Western stories"
]

if len(sys.argv) != 2:
    print("Expected 1 argument:")
    print("\t1) The path to the Gutenberg dataset directory")
    exit(0)

dataset_dir = sys.argv[1]
workspace = 'workspace'

gb_chapters_dataset.compile_dataset(subjects=subjects, dataset_dir=dataset_dir, workspace=workspace)
gb_titles_dataset.compile_dataset(subjects=subjects, dataset_dir=dataset_dir, workspace=workspace)
gb_full_dataset.compile_dataset(subjects=subjects, dataset_dir=dataset_dir, workspace=workspace)
gb_h2_chapters_dataset.compile_dataset(subjects=subjects, dataset_dir=dataset_dir, workspace=workspace)