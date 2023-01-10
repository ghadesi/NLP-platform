# Single function
#     from nlp_utils.preprocessing.text_preprocessing import [name of the function]
# Multiple functions
#     from nlp_utils.preprocessing.text_preprocessing import (f1, f2, f3)"

# By the below lines we fixed the issue of import testing for the cleaner_helper module
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent))

from nlp_utils.preprocessing.text_preprocessing import to_strip
from nlp_utils.preprocessing.text_preprocessing import to_lower
from nlp_utils.preprocessing.text_preprocessing import remove_number
from nlp_utils.preprocessing.text_preprocessing import remove_any_char
from nlp_utils.preprocessing.text_preprocessing import remove_all_duplication
from nlp_utils.preprocessing.text_preprocessing import remove_consecutive_duplication
from nlp_utils.preprocessing.text_preprocessing import remove_many_spaces
from nlp_utils.preprocessing.text_preprocessing import remove_emoji
from nlp_utils.preprocessing.text_preprocessing import remove_url
from nlp_utils.preprocessing.text_preprocessing import remove_twitter_username
from nlp_utils.preprocessing.text_preprocessing import remove_username
from nlp_utils.preprocessing.text_preprocessing import remove_hashtag
from nlp_utils.preprocessing.text_preprocessing import remove_email_address
from nlp_utils.preprocessing.text_preprocessing import blank_checker
from nlp_utils.preprocessing.text_preprocessing import remove_special_char
from nlp_utils.preprocessing.text_preprocessing import remove_punctuation
from nlp_utils.preprocessing.text_preprocessing import remove_emoticon
from nlp_utils.preprocessing.text_preprocessing import abbreviation_converter
from nlp_utils.preprocessing.text_preprocessing import convert_to_unicode
from nlp_utils.preprocessing.text_preprocessing import expand_contractions
from nlp_utils.preprocessing.text_preprocessing import spell_correction_v1
from nlp_utils.preprocessing.text_preprocessing import add_word_to_stopwords_set
from nlp_utils.preprocessing.text_preprocessing import stopwords_nltk
from nlp_utils.preprocessing.cleaner_helper import custom_extended_stopwords, custom_shortforms, custom_direct_replacement_dict