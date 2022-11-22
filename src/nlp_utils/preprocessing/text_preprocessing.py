"""Module providing utils code for cleaning users text data."""
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard Library
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import string
import re

# 3rd Party
from langdetect import detect, detect_langs
from nltk.corpus import stopwords as nltk_sw
import nltk
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import contractions
import spacy
from spacy.language import Language

# Private

# ───────────────────────────────── Code ────────────────────────────────── #

# Using re.compile() and saving the resulting regular expression object for
# reuse is more efficient when the expression will be used several times in a
# single program

EMOJI_PATTERN = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)

# Thanks : https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
EMOTICON_PATTERN = re.compile(u'(' + u'|'.join(k for k in EMOTICONS_EMO) + u')')


URL_PATTERN = re.compile(r"(ftp://|smtp://|SMTP://|http://|https://|http://www\.|https://www\.|www\.)?"
                         r"(?:[\x21-\x39\x3b-\x3f\x41-\x7e]+(?::[!-9;-?A-~]+)?@)?(?:xn--[0-9a-z]+|[0-9A-Za-z_-]+\.)*"
                         r"(?:xn--[0-9a-z]+|[0-9A-Za-z-]+)\.(?:xn--[0-9a-z]+|[0-9A-Za-z]{2,10})"
                         r"(?::(?:6553[0-5]|655[0-2]\d|65[0-4]\d{2}|6[0-4]\d{3}|[1-5]\d{4}|[1-9]\d{1,3}|\d))?"
                         r"(?:/[\x21\x22\x24\x25\x27-x2e\x30-\x3b\x3e\x40-\x5b\x5d-\x7e]*)*(?:\#[\x21\x22\x24\x25\x27-x2e\x30-\x3b\x3e\x40-\x5b\x5d-\x7e]*)?"
                         r"(?:\?[\x21\x22\x24\x25\x27-\x2e\x30-\x3b\x40-\x5b\x5d-\x7e]+"
                         r"=[\x21\x22\x24\x25\x27-\x2e\x30-\x3b\x40-\x5b\x5d-\x7e]*)?")

# XML_PATTERN = re.compile('<.*?>')
XML_PATTERN = re.compile(r"<[^>]+?>")

CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")

SPACE_PATTERN = re.compile(r"\s+")

TWITTER_USERNAME_PATTERN = re.compile(r"(?<!\w)@[\w+]{1,15}\b")

ANY_USERNAME_PATTERN = re.compile(r"(?<!\w)@[\w+]+\b")

HASHTAG_PATTERN = re.compile(r"(?:#|＃)([\wÀ-ÖØ-öø-ÿ]+)")

EMAIL_PATTERN = re.compile(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)")

BLANK_PATTERN = re.compile(r"^\s*$")

CONS_DUPLICATION_PATTERN = re.compile(r"\b(\w+)( \1\b)+", flags=re.IGNORECASE)

# Typos, slang and other
SAMPLE_TYPOS_SLANG = {
    "w/e": "whatever",
    "usagov": "usa government",
    "recentlu": "recently",
    "ph0tos": "photos",
    "amirite": "am i right",
    "exp0sed": "exposed",
    "<3": "love",
    "luv": "love",
    "amageddon": "armageddon",
    "trfc": "traffic",
    "16yr": "16 year"
}

# Acronyms
SAMPLE_ACRONYMS = {
    "mh370": "malaysia airlines flight 370",
    "okwx": "oklahoma city weather",
    "arwx": "arkansas weather",
    "gawx": "georgia weather",
    "scwx": "south carolina weather",
    "cawx": "california weather",
    "tnwx": "tennessee weather",
    "azwx": "arizona weather",
    "alwx": "alabama weather",
    "usnwsgov": "united states national weather service",
    "2mw": "tomorrow"
}

# Some common abbreviations
SAMPLE_ABBR = {
    "$": " dollar ",
    "€": " euro ",
    "4ao": "for adults only",
    "a.m": "before midday",
    "a3": "anytime anywhere anyplace",
    "aamof": "as a matter of fact",
    "acct": "account",
    "adih": "another day in hell",
    "afaic": "as far as i am concerned",
    "afaict": "as far as i can tell",
    "afaik": "as far as i know",
    "afair": "as far as i remember",
    "afk": "away from keyboard",
    "app": "application",
    "approx": "approximately",
    "apps": "applications",
    "asap": "as soon as possible",
    "asl": "age, sex, location",
    "atk": "at the keyboard",
    "ave.": "avenue",
    "aymm": "are you my mother",
    "ayor": "at your own risk",
    "b&b": "bed and breakfast",
    "b+b": "bed and breakfast",
    "b.c": "before christ",
    "b2b": "business to business",
    "b2c": "business to customer",
    "b4": "before",
    "b4n": "bye for now",
    "b@u": "back at you",
    "bae": "before anyone else",
    "bak": "back at keyboard",
    "bbbg": "bye bye be good",
    "bbc": "british broadcasting corporation",
    "bbias": "be back in a second",
    "bbl": "be back later",
    "bbs": "be back soon",
    "be4": "before",
    "bfn": "bye for now",
    "blvd": "boulevard",
    "bout": "about",
    "brb": "be right back",
    "bros": "brothers",
    "brt": "be right there",
    "bsaaw": "big smile and a wink",
    "btw": "by the way",
    "bwl": "bursting with laughter",
    "c/o": "care of",
    "cet": "central european time",
    "cf": "compare",
    "cia": "central intelligence agency",
    "csl": "can not stop laughing",
    "cu": "see you",
    "cul8r": "see you later",
    "cv": "curriculum vitae",
    "cwot": "complete waste of time",
    "cya": "see you",
    "cyt": "see you tomorrow",
    "dae": "does anyone else",
    "dbmib": "do not bother me i am busy",
    "diy": "do it yourself",
    "dm": "direct message",
    "dwh": "during work hours",
    "e123": "easy as one two three",
    "eet": "eastern european time",
    "eg": "example",
    "embm": "early morning business meeting",
    "encl": "enclosed",
    "encl.": "enclosed",
    "etc": "and so on",
    "faq": "frequently asked questions",
    "fawc": "for anyone who cares",
    "fb": "facebook",
    "fc": "fingers crossed",
    "fig": "figure",
    "fimh": "forever in my heart",
    "ft.": "feet",
    "ft": "featuring",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "g9": "genius",
    "gahoy": "get a hold of yourself",
    "gal": "get a life",
    "gcse": "general certificate of secondary education",
    "gfn": "gone for now",
    "gg": "good game",
    "gl": "good luck",
    "glhf": "good luck have fun",
    "gmt": "greenwich mean time",
    "gmta": "great minds think alike",
    "gn": "good night",
    "g.o.a.t": "greatest of all time",
    "goat": "greatest of all time",
    "goi": "get over it",
    "gps": "global positioning system",
    "gr8": "great",
    "gratz": "congratulations",
    "gyal": "girl",
    "h&c": "hot and cold",
    "hp": "horsepower",
    "hr": "hour",
    "hrh": "his royal highness",
    "ht": "height",
    "ibrb": "i will be right back",
    "ic": "i see",
    "icq": "i seek you",
    "icymi": "in case you missed it",
    "idc": "i do not care",
    "idgadf": "i do not give a damn fuck",
    "idgaf": "i do not give a fuck",
    "idk": "i do not know",
    "ie": "that is",
    "i.e": "that is",
    "ifyp": "i feel your pain",
    "IG": "instagram",
    "iirc": "if i remember correctly",
    "ilu": "i love you",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "imu": "i miss you",
    "iow": "in other words",
    "irl": "in real life",
    "j4f": "just for fun",
    "jic": "just in case",
    "jk": "just kidding",
    "jsyk": "just so you know",
    "l8r": "later",
    "lb": "pound",
    "lbs": "pounds",
    "ldr": "long distance relationship",
    "lmao": "laugh my ass off",
    "lmfao": "laugh my fucking ass off",
    "lol": "laughing out loud",
    "ltd": "limited",
    "ltns": "long time no see",
    "m8": "mate",
    "mf": "motherfucker",
    "mfs": "motherfuckers",
    "mfw": "my face when",
    "mofo": "motherfucker",
    "mph": "miles per hour",
    "mr": "mister",
    "mrw": "my reaction when",
    "ms": "miss",
    "mte": "my thoughts exactly",
    "nagi": "not a good idea",
    "nbc": "national broadcasting company",
    "nbd": "not big deal",
    "nfs": "not for sale",
    "ngl": "not going to lie",
    "nhs": "national health service",
    "nrn": "no reply necessary",
    "nsfl": "not safe for life",
    "nsfw": "not safe for work",
    "nth": "nice to have",
    "nvr": "never",
    "nyc": "new york city",
    "oc": "original content",
    "og": "original",
    "ohp": "overhead projector",
    "oic": "oh i see",
    "omdb": "over my dead body",
    "omg": "oh my god",
    "omw": "on my way",
    "p.a": "per annum",
    "p.m": "after midday",
    "pm": "prime minister",
    "poc": "people of color",
    "pov": "point of view",
    "pp": "pages",
    "ppl": "people",
    "prw": "parents are watching",
    "ps": "postscript",
    "pt": "point",
    "ptb": "please text back",
    "pto": "please turn over",
    "qpsa": "what happens",  # "que pasa",
    "ratchet": "rude",
    "rbtl": "read between the lines",
    "rlrt": "real life retweet",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off",
    "rt": "retweet",
    "ruok": "are you ok",
    "sfw": "safe for work",
    "sk8": "skate",
    "smh": "shake my head",
    "sq": "square",
    "srsly": "seriously",
    "ssdd": "same stuff different day",
    "tbh": "to be honest",
    "tbs": "tablespooful",
    "tbsp": "tablespooful",
    "tfw": "that feeling when",
    "thks": "thank you",
    "tho": "though",
    "thx": "thank you",
    "tia": "thanks in advance",
    "til": "today i learned",
    "tl;dr": "too long i did not read",
    "tldr": "too long i did not read",
    "tmb": "tweet me back",
    "tntl": "trying not to laugh",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "u4e": "yours for ever",
    "utc": "coordinated universal time",
    "w/": "with",
    "w/o": "without",
    "w8": "wait",
    "wassup": "what is up",
    "wb": "welcome back",
    "wtf": "what the fuck",
    "wtg": "way to go",
    "wtpa": "where the party at",
    "wuf": "where are you from",
    "wuzup": "what is up",
    "wywh": "wish you were here",
    "yd": "yard",
    "ygtr": "you got that right",
    "ynk": "you never know",
    "zzz": "sleeping bored and tired"
}

SAMPLE_TYPOS_SLANG_PATTERN = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in SAMPLE_TYPOS_SLANG.keys()) + r')(?!\w)')
SAMPLE_ACRONYMS_PATTERN = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in SAMPLE_ACRONYMS.keys()) + r')(?!\w)')
SAMPLE_ABBR_PATTERN = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in SAMPLE_ABBR.keys()) + r')(?!\w)')

# Expanding contractions
CONTRACTIONS_DIC = {
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
CONTRACTIONS_PATTERN = re.compile('({})'.format('|'.join(CONTRACTIONS_DIC.keys())), flags=re.IGNORECASE | re.DOTALL)


def remove_xml(html_text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """ 
    Eliminates the HTML tags from the given text and returns a tuple
    that contains the purified text and the number of matches.

    Args:
        html_text (Optional[str]): a text that may contain HTML tags

    Returns:
        Tuple[str, int]: (the purified text accoding to HTML tags, the number of matches)
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
        Tuple[Optional[str], Optional[int]]: (the purified text according to any characters, the number of matches)
    """
    # Input checking
    if pd.isnull(text):
        return None, None

    if not isinstance(text, str):
        return None, None

    return re.subn(CHAR_PATTERN, r"", text, re.I | re.A)


def remove_all_duplication(text: Optional[str]) -> Optional[str]:
    """
    Removes all duplicate words from the given string and just maintains the first one

    Args:
        text (Optional[str]): a text may contain multiple words that are the same

    Returns:
        str: the purified text that does not contain duplicated words
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    # lower_case_text = to_lower(text)
    tokenize_text = text.split()
    return " ".join(sorted(set(tokenize_text), key=tokenize_text.index))


def remove_consecutive_duplication(text: Optional[str]) -> Optional[str]:
    """
    Removes consecutive duplicate words from the given text
    Args:
        text (Optional[str]): a text may contain the same words 

    Returns:
        Optional[str]: the purified text that does not contain consecutive duplicated words
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return re.sub(CONS_DUPLICATION_PATTERN, r'\1', text)


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
    Removes any emojis in the given text. 
    http://www.unicode.org/Public/emoji/1.0//emoji-data.txt

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
        Tuple[Optional[str], Optional[int]]: (the purified string that does not have any URLs, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(URL_PATTERN, r'', text)


def remove_twitter_username(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes twitter username from the given text w.r.t Twitter username policies.
    A maximum of 15 characters (words) are allowed.

    Args:
        text (Optional[str]): a text that may contain multiple usernames

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text doesn't have any twitter username, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(TWITTER_USERNAME_PATTERN, r'', text)


def remove_username(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes username parts that start with the "@" sign. 
    In this function, there is no limitaion for the length of the username.

    Args:
        text (Optional[str]): a text that may contain multiple usernames

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text doesn't have any username, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(ANY_USERNAME_PATTERN, r'', text)


def remove_hashtag(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes hashtagh from the given text. This function supports multilanguage hashtags. 
    Example: https://regex101.com/r/SxRara/1

    Args:
        text (Optional[str]): a text that may contain multiple hashtags

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text doesn't have any hashtag, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(HASHTAG_PATTERN, r'', text)


def remove_email_address(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes email address/s from the given text

    Args:
        text (Optional[str]): a text that may contain multiple email addresses

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text  does not have any email addresses, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(EMAIL_PATTERN, r'', text)


def blank_checker(text: Optional[str]) -> Optional[bool]:
    """
    Checkes the given text is composed space, \t, \r, and \n.

    Args:
        text (Optional[str]): a text can be empty

    Returns:
        Optional[bool]: a boolean value that shows whether the given text is empty or not.
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    if re.search(BLANK_PATTERN, text):
        return True
    else:
        return False


def remove_special_char(text: Optional[str], special_char: Optional[List[str]]) -> Tuple[Optional[str], Optional[int]]:
    """
    Removes special characters through the input list from the given text

    Args:
        text (Optional[str]): a gien text
        special_char (Optional[List[str]]): a list contains special characters

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text does not have the given characters, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    if not isinstance(special_char, List):
        return text, None

    if len(special_char) == 0:
        return text, None

    return re.subn("|".join(special_char), r"", text)


def expand_contractions(text: Optional[str]) -> Optional[str]:
    """
    Contractions are words or combinations of words that are shortened by dropping letters and replacing them with an apostrophe. 
    With this function, we are going to convert the text into the standard form.

    Args:
        text (Optional[str]): a text that may contain shortened forms of words

    Returns:
        Optional[str]: A converted text in which shortened words are transferred to a standard shape. 
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTIONS_DIC.get(match) \
            if CONTRACTIONS_DIC.get(match) \
            else CONTRACTIONS_DIC.get(match.lower())
        expanded_contraction = expanded_contraction

        return expanded_contraction

    expanded_text = CONTRACTIONS_PATTERN.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    # We used the contractions linbrary to expand our coverage expantion (https://github.com/kootenpv/contractions)
    # TODO: We can use contractions.add('mychange', 'my change') to add our contractions to the contractions linbrary
    contractions.fix(expanded_text)
    return expanded_text


def convert_to_unicode(text: Optional[Any], encoding: str = "utf-8") -> Optional[str]:
    """
    Converts the given text to Unicode. Input must be utf-8.

    Args:
        text (Optional[Any]): a given text can be in any format
        encoding (str, optional): The type of encoding. Defaults to "utf-8".

    Returns:
         Optional[str]: a text in the string format
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode(encoding, "ignore")
    else:
        return None


def stop_words_spacy():
    # TODO: NOT finished!
    # Download multi language files
    # python - m spacy download xx_ent_wiki_sm
    # python -m spacy download xx_sent_ud_sm

    # loading the english language small model of spacy
    en = spacy.load('en_core_web_sm')
    stopwords = en.Defaults.stop_words
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(u"Tommorow will be too late, its now or never")

    for token in doc:
        print(token.text, token.is_stop)
    #  Remove a stop word
    # .add()
    nlp.Defaults.stop_words.remove("what")
    nlp.Defaults.stop_words -= {"who", "when"}

    lst = []
    for token in text.split():
        if token.lower() not in en_stopwords:
            lst.append(token)
    print(' '.join(lst))

    # Create list of word tokens after removing stopwords
    filtered_sentence = []

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)
    print(token_list)
    print(filtered_sentence)

    text_without_stopword = [word for word in text.split() if word not in spacy_stopwords]

    def remove_mystopwords(sentence):
        tokens = sentence.split(" ")
        tokens_filtered = [word for word in text_tokens if not word in my_stopwords]
        return (" ").join(tokens_filtered)

    # from nltk.tokenize import word_tokenize
    # text_tokens = word_tokenize(text)
    # tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]

    lexeme = nlp.vocab["is"].is_stop
    print('Number of stop words: %d' % len(spacy_stopwords))


def stopwords_nltk(pref_lang_lst: Optional[List[str]]) -> Optional[Set[str]]:
    """
    Returns a set that contains all stop words in NLTK based on the given language list.

    Args:
        pref_lang_lst (Optional[List[str]]): a list of languages

    Returns:
        Optional[Set[str]]: a set of all stop words w.r.t the given languages
    """
    # Input checking
    if isinstance(pref_lang_lst, str):
        # The input is in string format, instead of list or set
        pref_lang_lst = [pref_lang_lst]
    elif pd.isnull(pref_lang_lst) or not isinstance(pref_lang_lst, (Set, List)):
        return None
    if len(pref_lang_lst) == 0:
        return None

    NLTK_sup_lang = set(nltk_sw.fileids())
    pref_lang_lst_lower_case = [lang.lower().strip() for lang in pref_lang_lst]
    pref_lang_set = set(pref_lang_lst_lower_case)

    # Check the intersection between two sets: NLTK_sup_lang and pref_lang_set
    if pref_lang_set.issubset(NLTK_sup_lang):
        total_lang = pref_lang_set
    elif pref_lang_set - NLTK_sup_lang:
        total_lang = pref_lang_set.intersection(NLTK_sup_lang)
    else:
        # There is no intersection between sets
        return None

    stop_words = set()

    try:
        stop_words = nltk_sw.words("english")
    except LookupError:
        nltk.download('stopwords')
    finally:
        for language in total_lang:
            stop_words = stop_words.union(set(nltk_sw.words(language)))

    return stop_words


def remove_stopwords(text: Optional[str], stopwords: Set) -> Optional[str]:
    """
    Removes all stopwords from the given text

    Args:
        text (Optional[str]): a text may contain stopwords
        stopwords (Set): the desired stopwords set based on a specific language/s
        
    Returns:
        Optional[str]: a purified text w/o any stopwords
    """    
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None
    if pd.isnull(stopwords) or not isinstance(stopwords, Set):
        return None
    
    return " ".join([word for word in str(text).split() if word not in stopwords])

def language_detection(text: Optional[str]) -> Optional[str]:
    """
    To detect the language of the text. The method returns a single language 
    name which has the highest probability.

    Note: 
    Language detection algorithm is non-deterministic, which means that if 
    we try to run it on a text which is either too short or too ambiguous, we 
    might get different results everytime you run it.

    see more:
    https://code.google.com/archive/p/language-detection/wikis/Tools.wiki
    https://github.com/Mimino666/langdetect

    Args:
        text (Optional[str]): a given that that can be in any language

    Returns:
        Optional[str]: a single language abbreviation which has the highest probability.
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return detect(text)


def language_prob_detection(text: Optional[str]) -> Optional[List[Tuple[str, float]]]:
    """
    To find out the probabilities for the top languages.

    Note: 
    Language detection algorithm is non-deterministic, which means that if 
    we try to run it on a text which is either too short or too ambiguous, we 
    might get different results everytime you run it.

    see more:
    https://code.google.com/archive/p/language-detection/wikis/Tools.wiki
    https://github.com/Mimino666/langdetect

    Args:
        text (Optional[str]): a given that that can be in any language

    Returns:
        Optional[List[Tuple[str, float]]]: a list of multiple languages and their probabilities.
    """

    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return detect_langs(text)


def lang_conv_spacy(lang: Optional[str]) -> Optional[str]:
    """
    This function maps the name of the language into its corresponding abbreviation form. 
    I have to mention that Spacy doesn't support ISO-639-1 (https://github.com/noumar/iso639).

    Args:
        lang (Optional[str]): the name of the language

    Returns:
        Optional[str]: the abbreviation form of a language
    """
    # Input checking
    if pd.isnull(lang) or not isinstance(lang, str):
        return None

    dic_lang = {"Catalan": "ca", "Chinese": "zh", "Croatian": "hr", "Danish": "da", "Dutch": "nl", "English": "en", "Finnish": "fi", "French": "fr", "German": "de", "Greek": "el", "Italian": "it", "Japanese": "ja", "Korean": "ko",
                "Lithuanian": "lt", "Macedonian": "mk", "Norwegian Bokmål": "nb", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", "Russian": "ru", "Spanish": "es", "Swedish": "sv", "Ukrainian": "uk", "Multi-language": "xx"}

    if dic_lang.__contains__(lang):
        return dic_lang[lang]
    else:
        return None


def pipeline_source_selector_spacy(lang_abb: Optional[str]) -> Optional[str]:
    """
    Returns a proper source for the specified language.

    Args:
        lang_abb (Optional[str]): the language abbreviation that Spacy supports

    Returns:
        Optional[str]: a proper source based on the given language
    """
    # Input checking
    if pd.isnull(lang_abb) or not isinstance(lang_abb, str):
        return None

    web_source = ["zh", "en"]
    news_source = ["ca", "hr", "da", "nl", "fi", "fr", "de", "el", "it", "ja", "ko", "lt", "mk", "nb", "pl", "pt", "ro", "ru", "es", "sv", "uk"]
    wiki_source = ["xx"]

    selected_source = None

    if lang_abb in web_source:
        selected_source = "web"
    elif lang_abb in news_source:
        selected_source = "news"
    elif lang_abb in wiki_source:
        selected_source = "wiki"

    return selected_source


def pipeline_size_str_conv_spacy(pipeline_size: Literal["small", "medium", "large", "transformer"]) -> Optional[str]:
    """
    Converts the pipeline size w.r.t the Spacy specifications

    Args:
        pipeline_size (Literal["small", "medium", "large", "transformer"]): acceptable pipeline size

    Returns:
        Optional[str]: the abbreviation form of the pipeline size
    """
    # Input checking
    if pd.isnull(pipeline_size) or not isinstance(pipeline_size, str):
        return None

    if pipeline_size == "small":
        return "sm"
    elif pipeline_size == "medium":
        return "md"
    elif pipeline_size == "large":
        return "lg"
    elif pipeline_size == "transformer":
        return "trf"
    else:
        return None


def pipeline_selector_spacy(pref_lang: Optional[str], pref_pipeline_size: Literal["small", "medium", "large", "transformer"]) -> Optional[Language]:
    """
    Returns a Spacy language pipeline based on the input specifications.
    https://spacy.io/usage/models#languages

    Args:
        pref_lang (Optional[str]): a preferable language
        pref_pipeline_size (Literal["small", "medium", "large", "transformer"]): a preferable size of the pipeline

    Returns:
        Optional[Language]: a Spacy language pipeline
    """
    pipeline_lang = lang_conv_spacy(pref_lang)
    if pipeline_lang is None:
        # Spacy doen't support this language
        return None

    pipeline_source = pipeline_source_selector_spacy(pipeline_lang)
    if pipeline_source is None:
        # Spacy doen't have this pretrained model
        return None

    pipeline_size = pipeline_size_str_conv_spacy(pref_pipeline_size)
    if pipeline_size is None:
        # Spacy doen't support the given size
        return None

    pipeline_name = str(pipeline_lang + "_core_" + pipeline_source + "_" + pipeline_size)

    try:
        nlp_pipeline = spacy.load(pipeline_name)
    except:
        spacy.cli.download(pipeline_name)
    finally:
        nlp_pipeline = spacy.load(pipeline_name)

    return nlp_pipeline


def remove_emoticons(text: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
        Removes all emoticons from the given text

    Args:
        text (Optional[str]): a text which contains multiple emoticons

    Returns:
        Tuple[Optional[str], Optional[int]]: (the purified text does not have any emoticons, the number of matches)
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None, None

    return re.subn(EMOTICON_PATTERN, r'', text)


def abbreviation_converter(text: Optional[str]) -> Optional[str]:
    """
    Converts abbreviation forms of the input text into the normal shape.

    Args:
        text (Optional[str]): a text with abbreviation forms of words

    Returns:
        Optional[str]: a converted text without any abbreviation
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    text = re.sub(SAMPLE_TYPOS_SLANG_PATTERN, lambda x: SAMPLE_TYPOS_SLANG[x.group()], text)
    text = re.sub(SAMPLE_ACRONYMS_PATTERN, lambda x: SAMPLE_ACRONYMS[x.group()], text)
    text = re.sub(SAMPLE_ABBR_PATTERN, lambda x: SAMPLE_ABBR[x.group()], text)

    return text


def remove_punctuation(text: Optional[str]) -> Optional[str]:
    """
    Remove the punctuations from the given text

    Args:
        text (Optional[str]): a text that may contain punctuation

    Returns:
        Optional[str]: a purified string that does not have any punctuation
    """
    # Input checking
    if pd.isnull(text) or not isinstance(text, str):
        return None

    return text.translate(str.maketrans('', '', string.punctuation))

# TODO: add specific character remove
# TODO: give re and apply that
# TODO: some function should apply to the whole data set such as remove frequent words, rare words, and distribution of language if doesn't have language label
# TODO: stemming and lemmatization
# TODO: Conversion of Emoticon to Words
# TODO: Conversion of Emoji to Words
# TODO: remove xml precisely BeautifulSoup
# TODO: Chat Words Conversion
# TODO: Spelling Correction
# TODO: camelcase
# TODO: Convert the abbreviation of countries to the standard shape
# TODO: User add RE and replace text

# full_stopwords_set = set.union(set(custom_extended_stopwords), set(stopwords.words("english")))
# from cleaner_helper import custom_extended_stopwords, custom_shortforms, custom_direct_replacement_dict
