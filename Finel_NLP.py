#!/usr/bin/env python3
# encoding: utf-8
import re

# Expanding contractions

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()


nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)
text = 'This is an english text.'
doc = nlp(text)
print(doc._.language)
