#!/usr/bin/env python3
# encoding: utf-8

"""Module providing utils code for cleaning users text data."""
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard Library
from typing import Tuple, Optional, Union, Callable, Any
import re

# 3rd Party
import numpy as np
import pandas as pd

# Private

# ───────────────────────────────── Code ────────────────────────────────── #


def remove_xml(html_text: str) -> Tuple[Optional[str], Optional[int]]:
    """ 
    Eliminates the HTML tags from the given text and returns a tuple
    that contains the purified text and the number of matches.

    Args:
        html_text (str): a text variable may contain HTML tags

    Returns:
        Tuple[str, int]: (purified text accoding to HTML tags, the number of matches)
    """
    # Input checking
    if pd.isnull(html_text):
        return None, None

    if not isinstance(html_text, str):
        return None, None

    return re.subn(r'<[^>]+?>', '', html_text)


def to_lower(text: str) -> Optional[str]:
    """
    Converts the given text to lower case

    Args:
        text (str): a text to be converted

    Returns:
        Optional[str]: a lowercase string
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return text.lower()


def remove_number(text: str) -> Optional[str]:
    """
    Removes any digits from the given string

    Args:
        text (str): a text may contain digits

    Returns:
        Optional[str]: a purified string that does not have any numbers
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None
    return ''.join(c for c in text if not c.isdigit())


def to_strip(text: str) -> Optional[str]:
    """
    Removes all whitespaces at the beginning and end of the string.
    Also, it eliminates multiple spaces between words.

    Args:
        text (str): a string may contain multiple spaces 

    Returns:
        str: a purified string that does not have useless whitespaces
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return " ".join([c for c in text.split()])


def remove_any_char(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes all characters in the given text

    Args:
        text (str): a text may contain multiple types of characters

    Returns:
        Tuple[Optional[str], Optional[int]]: (purified text according to any characters, the number of matches)
    """
    # Input checking
    if pd.isnull(text):
        return None, None

    if not isinstance(text, str):
        return None, None

    return re.subn(r'[^a-zA-Z\s]', '', text, re.I | re.A)


# Expanding contractions
contractions_dict = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
    "he'd": "he had", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I had", "I'd've": "I would have",
    "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it had",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "iit will have", "it's": "it is", "let's": "let us",
    "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she had", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that had", "that'd've": "that would have",
    "that's": "that is", "there'd": "there had", "there'd've": "there would have", "there's": "there is", "they'd": "they had",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we had", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}


def expand_contraction(text: str, contraction_dict: dict) -> str:
    contraction_pattern = re.compile('({})'.format('|'.join(contraction_dict.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_dict.get(match) \
            if contraction_dict.get(match) \
            else contraction_dict.get(match.lower())
        expanded_contraction = expanded_contraction

        return expanded_contraction

    expanded_text = contraction_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text


def main_contraction(text: str) -> str:

    text = expand_contraction(text, contractions_dict)

    return text
