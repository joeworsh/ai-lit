"""
A set of utilities specifically for building and managing GB TFRecords.
"""


def get_atomic_subject(gsubs, subjects):
    """
    Return the first subject which is found in both the list
    of subjects in the Gutenberg Index, gsubs, and the list of subjects
    used for the training/inference session.
    :param gsubs: The GB index subjects to extract the atomic subject from.
    :param subjects: The list of subjects which are targeted by the classification problem.
    :return: The first automic subject or null.
    """
    for gsub in gsubs:
        if gsub in subjects:
            return gsub

    return None
