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

# Using re.compile() and saving the resulting regular expression object for
# reuse is more efficient when the expression will be used several times in a
# single program

EMOJI_PATTERN = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # emoticons
    u'\U0001F300-\U0001F5FF'  # symbols & pictographs
    u'\U0001F680-\U0001F6FF'  # transport & map symbols
    u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
    u'\U00002702-\U000027B0'
    u'\U000024C2-\U0001F251'
    ']+',
    flags=re.UNICODE)

URL_PATTERN = re.compile(r"(ftp://|smtp://|SMTP://|http://|https://|http://www\.|https://www\.|www\.)?"
                         r"(?:[\x21-\x39\x3b-\x3f\x41-\x7e]+(?::[!-9;-?A-~]+)?@)?(?:xn--[0-9a-z]+|[0-9A-Za-z_-]+\.)*"
                         r"(?:xn--[0-9a-z]+|[0-9A-Za-z-]+)\.(?:xn--[0-9a-z]+|[0-9A-Za-z]{2,10})"
                         r"(?::(?:6553[0-5]|655[0-2]\d|65[0-4]\d{2}|6[0-4]\d{3}|[1-5]\d{4}|[1-9]\d{1,3}|\d))?"
                         r"(?:/[\x21\x22\x24\x25\x27-x2e\x30-\x3b\x3e\x40-\x5b\x5d-\x7e]*)*(?:\#[\x21\x22\x24\x25\x27-x2e\x30-\x3b\x3e\x40-\x5b\x5d-\x7e]*)?"
                         r"(?:\?[\x21\x22\x24\x25\x27-\x2e\x30-\x3b\x40-\x5b\x5d-\x7e]+"
                         r"=[\x21\x22\x24\x25\x27-\x2e\x30-\x3b\x40-\x5b\x5d-\x7e]*)?")


XML_PATTERN = re.compile(r"<[^>]+?>")

CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")

SPACE_PATTERN = re.compile(r"\s+")

def remove_xml(html_text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """ 
    Eliminates the HTML tags from the given text and returns a tuple
    that contains the purified text and the number of matches.

    Args:
        html_text (Optional[str]): a text variable may contain HTML tags

    Returns:
        Tuple[str, int]: (purified text accoding to HTML tags, the number of matches)
    """
    # Input checking
    if pd.isnull(html_text):
        return None, None

    if not isinstance(html_text, str):
        return None, None

    return re.subn(XML_PATTERN, r"", html_text)


def to_lower(text: Optional[str]) -> Optional[str]:
    """
    Converts the given text to lower case

    Args:
        text (Optional[str]): a text to be converted

    Returns:
        Optional[str]: a lowercase string
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return text.lower()


def remove_number(text: Optional[str]) -> Optional[str]:
    """
    Removes any digits from the given string

    Args:
        text (Optional[str]): a text may contain digits

    Returns:
        Optional[str]: a purified string that does not have any numbers
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None
    return ''.join(c for c in text if not c.isdigit())


def to_strip(text: Optional[str]) -> Optional[str]:
    """
    Removes all whitespaces at the beginning and end of the string.
    Also, it eliminates multiple spaces between words.

    Args:
        text (Optional[str]): a string may contain multiple spaces 

    Returns:
        str: a purified string that does not have useless whitespaces
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return " ".join([c for c in text.split()])


def remove_any_char(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes all characters in the given text

    Args:
        text (Optional[str]): a text may contain multiple types of characters

    Returns:
        Tuple[Optional[str], Optional[int]]: (purified text according to any characters, the number of matches)
    """
    # Input checking
    if pd.isnull(text):
        return None, None

    if not isinstance(text, str):
        return None, None

    return re.subn(CHAR_PATTERN, r"", text, re.I | re.A)


def remove_duplication(text: Optional[str]) -> Optional[str]:
    """
    Removes duplicate words from a string

    Args:
        text (Optional[str]): a text may contain multiple words that are the same

    Returns:
        str: purified text that does not contain duplicate words
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    # lower_case_text = to_lower(text)
    tokenize_text = text.split()
    return " ".join(sorted(set(tokenize_text), key=tokenize_text.index))


def remove_many_spaces(text: Optional[str]) -> Optional[str]:
    """
    Removes continuous whitespaces

    Args:
        text (Optional[str]): a text that may contain multiple spaces sequentially

    Returns:
        Optional[str]: a text string without any continuous whitespaces
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return re.sub(SPACE_PATTERN, r" ", text)


def remove_emoji(text: Optional[str]) -> Optional[str]:
    """
    Removes any emojis in the given text

    Args:
        text (Optional[str]): a text that may contain multiple emojis

    Returns:
        Optional[str]: a purified string that does not have any emojis
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return re.sub(EMOJI_PATTERN, r'', text)


def remove_url(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes any url in the given text (Example: https://regex101.com/r/RvtAey/1)

    Args:
        text (Optional[str]): a text that may contain multiple URLS

    Returns:
        Tuple[Optional[str], Optional[int]]: a purified string that does not have any URLs
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(URL_PATTERN, r'', text)


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
