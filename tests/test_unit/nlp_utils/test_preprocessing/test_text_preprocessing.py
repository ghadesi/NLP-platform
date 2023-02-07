"""Module providing utils testing for users to convert data."""
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard library
from typing import List, Union, Any, Optional, Set, Iterable
import pandas as pd
import numpy as np
import sys
import re

# 3rd Party
import pytest

# Private
import test_unit.nlp_utils.test_preprocessing._synthetic_dbs as dbs
from nlp_utils.preprocessing.text_preprocessing import remove_xml
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
from nlp_utils.preprocessing.text_preprocessing import remove_special_char
from nlp_utils.preprocessing.text_preprocessing import blank_checker
from nlp_utils.preprocessing.text_preprocessing import remove_punctuation
from nlp_utils.preprocessing.text_preprocessing import remove_emoticon
from nlp_utils.preprocessing.text_preprocessing import abbreviation_converter
from nlp_utils.preprocessing.text_preprocessing import convert_to_unicode
from nlp_utils.preprocessing.text_preprocessing import expand_contractions
from nlp_utils.preprocessing.text_preprocessing import spell_correction_v1
from nlp_utils.preprocessing.text_preprocessing import add_word_to_stopwords_set
from nlp_utils.preprocessing.text_preprocessing import stopwords_nltk
from nlp_utils.preprocessing.text_preprocessing import convert_emoji_to_words
from nlp_utils.preprocessing.text_preprocessing import convert_emoticon_to_words
from nlp_utils.preprocessing.text_preprocessing import remove_regex_match
from nlp_utils.preprocessing.text_preprocessing import substitue_regex_match
from nlp_utils.preprocessing.text_preprocessing import to_lemmatize
from nlp_utils.preprocessing.text_preprocessing import to_tokenize
from nlp_utils.preprocessing.cleaner_helper import custom_extended_stopwords, custom_shortforms, custom_direct_replacement_dict

# ───────────────────────────────── Tests ────────────────────────────────── #


class TestHTML:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("""<html lang="en" class="notranslate" translate="no" data-theme="light"><HEAD> This is HEAD <INSIDE> The is inside tag </INSIDE></HEAD> <BODY> This is BODY </BODY></HTML>""",
             " This is HEAD  The is inside tag   This is BODY ", 8),
            ("Hello world!", "Hello world!", 0),
            (None, None, 0),
        ],
    )
    def test_remove_xml_HTML_tags(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_xml(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_xml_HTML_tags(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_xml(text)


class TestToStrip:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (" Hello world   ! ", "Hello world !"),
            (" Hello  world   ! ", "Hello world !"),
            (None, None)
        ],
    )
    def test_to_strip(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = to_strip(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_to_strip(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = to_strip(text)


class TestToLower:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("A beAuTifUl WORD", "a beautiful word"),
            (None, None)
        ],
    )
    def test_to_lower(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = to_lower(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_to_lower(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = to_lower(text)


class TestNumbers:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello123 678 1000 44word 5", "Hello   word "),
            (" 5", " "),
            ("H$Charly is now available to claim at https://t.co/LPOl9Kt08V üö∞ @Charlytoken7 üëë ¬† #Cardano $ADA #TapTools https://t.co/6hJBnoqwrs", "H$Charly is now available to claim at https://t.co/LPOlKtV üö∞ @Charlytoken üëë ¬† #Cardano $ADA #TapTools https://t.co/hJBnoqwrs"),
            ("$XTZ. Push! Keep on rising! ‚Ä¢ Price (USD): $ 1.98900000 ‚Ä¢ Sharing = Pushing!!",
             "$XTZ. Push! Keep on rising! ‚Ä¢ Price (USD): $ . ‚Ä¢ Sharing = Pushing!!"),
            (None, None)
        ],
    )
    def test_remove_number(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_number(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_number(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_number(text)


class TestCharacter:
    # @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("!&  Hel_lo *@ $world!(", "    Hel lo     world  ", 8),
            ("$XTZ. Push! Keep on rising! ‚Ä¢ Price (USD): $ 1.98900000 ‚Ä¢ Sharing = Pushing!!",
             " XTZ  Push  Keep on rising      Price  USD                    Sharing   Pushing  ", 27),
            (None, None, 0),
        ],
    )
    def test_remove_any_char(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_any_char(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_any_char(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_any_char(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, input_spec_char, ex_output_text, ex_num_matches",
        [
            ("Hello, ¿the first char is ٪ and these onces《 》!٪", ["٪", "《", "》", "¿"], "Hello,  the first char is   and these onces   ! ", 5),
            ("Hello, ", None, "Hello, ", 0),
            ("H$Charly is now available to claim at https://t.co/LPOl9Kt08V üö∞ @Charlytoken7 üëë ¬† #Cardano $ADA #TapTools https://t.co/6hJBnoqwrs", ["$", "", "ü", "ö", "∞", "@", "ë", "¬", "†", "#"], "H Charly is now available to claim at https://t.co/LPOl9Kt08V       Charlytoken7          Cardano  ADA  TapTools https://t.co/6hJBnoqwrs", 15),
            ("$XTZ. Push! Keep on rising! ‚Ä¢ Price (USD): $ 1.98900000 ‚Ä¢ Sharing = Pushing!!", [
             "$", ".", "!", ",", ":", "Ä", "¢", "=", "!", "‚"], " XTZ  Push  Keep on rising      Price (USD)    1 98900000     Sharing   Pushing  ", 16),
            (None, None, None, 0),
        ],
    )
    def test_remove_special_char(self, input_text: Optional[str], input_spec_char: Optional[List[str]], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_special_char(input_text, input_spec_char)
        
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_special_char(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_special_char(text, ["٪", "《", "》", "¿"])

    # @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text",
        [
            ("James: \"Hi Thomas, I haven't seen you for ages! How have you been?\"", "James   Hi Thomas  I haven t seen you for ages  How have you been  "),
            ("!hi. wh?at is the weat[h]er lik?e.", " hi  wh at is the weat h er lik e "),
            ("H$Charly is now available to claim at https://t.co/LPOl9Kt08V üö∞ @Charlytoken7 üëë ¬† #Cardano $ADA #TapTools https://t.co/6hJBnoqwrs", 
             "H Charly is now available to claim at https   t co LPOl9Kt08V üö∞  Charlytoken7 üëë ¬†  Cardano  ADA  TapTools https   t co 6hJBnoqwrs"),
            (None, None),
        ],
    )
    def test_remove_punctuation(self, input_text: Optional[str], ex_output_text: Optional[str]):

        result_text = remove_punctuation(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output_text, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_punctuation(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_punctuation(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text",
        [
            (None, None),
            ("", ""),
        ],
    )
    def test_spell_correction_v1(self, input_text: Optional[str], ex_output_text: Optional[str]):

        result_text = spell_correction_v1(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output_text, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_spell_correction_v1(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            # result_text = spell_correction_v1(text)
            ...


class TestDuplication:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello hello", "Hello hello"),
            ("Hello world world! world Hello world", "Hello world world!"),
            ("My name is Amin Amin. Amin comes from Iran!!", "My name is Amin Amin. comes from Iran!!"),
            ("My name is Amin Amin , Amin comes from Iran", "My name is Amin , comes from Iran"),
            ("what type of people were most likely to be able to be able to be able to be able to be 1.35 able to be ?",
             "what type of people were most likely to be able 1.35 ?"),

            (None, None)
        ],
    )
    def test_remove_all_duplication(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = remove_all_duplication(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_all_duplication(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_all_duplication(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello hello", "Hello"),
            ("this is just is is", "this is just is"),
            ("this just so So so nice", "this just so nice"),
            (None, None)
        ],
    )
    def test_remove_consecutive_duplication(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = remove_consecutive_duplication(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_consecutive_duplication(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_consecutive_duplication(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello          world", "Hello world"),
            (None, None)
        ],
    )
    def test_remove_many_spaces(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = remove_many_spaces(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_many_spaces(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_many_spaces(text)


class TestEmojiEmoticons:
    # @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output, ex_num_matches",
        [
            ("RT @ kaastore: 😁 un sourire=un cadeau 🎁", "RT @ kaastore:   un sourire=un cadeau  ", 2),
            (None, None, 0),
            ("⚪ An Anon Swapped $799K in $WBTC for $sBTC on #1inch 🐉 🐋 ($7.46M)", "  An Anon Swapped $799K in $WBTC for $sBTC on #1inch     ($7.46M)", 3),
            ("[SOLD after 21 hour(s)] 🦀 GEM(✨ Earl Cray 5/18📈📉)", "[SOLD after 21 hour(s)]   GEM(  Earl Cray 5/18 )", 4), 
            ("picture says more than words 😎💥🚀 🦍🤝🦍", "picture says more than words    ", 6),
        ],
    )
    def test_remove_emoji(self, input_text: Optional[str], ex_output: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_emoji(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_emoji(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_emoji(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output, ex_num_matches",
        [
            ("Hello :)", "Hello ", 1),
            ("Hello :-)", "Hello ", 1),
            (None, None, 0),
        ],
    )
    def test_remove_emoticon(self, input_text: Optional[str], ex_output: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_emoticon(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_emoticon(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_emoticon(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (None, None),
            ("", ""),
            ("Hello", "Hello"),
            ("Hello 😂", "Hello face_with_tears_of_joy"),  # Ask to Amir: Should we remove the underscore?
            ("Hello 🍕", "Hello pizza"),
            ("Hello ✅ ✍🏻 🧚🏼‍♀️", "Hello check_mark_button writing_handlight_skin_tone fairymedium-light_skin_tone‍female_sign️"),
        ],
    )
    def test_convert_emoji_to_words(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = convert_emoji_to_words(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_convert_emoji_to_words(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = convert_emoji_to_words(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (None, None),
            ("", ""),
            ("Hello", "Hello"),
            ("Hello :)", "Hello Happy_face_or_smiley"),  # Ask to Amir: Should we remove the underscore?
            ("Hello :-)", "Hello Happy_face_smiley"),
            ("Hello :) :(", "Hello Happy_face_or_smiley Frown_sad_andry_or_pouting"),
        ],
    )
    def test_convert_emoticon_to_words(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = convert_emoticon_to_words(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_convert_emoticon_to_words(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = convert_emoticon_to_words(text)


class TestURL:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("My website is https://www.twanda.com/apps/details?id=com.skgames.trafficracer%22", "My website is ", 1),
            ("My websites are https://www.google.com and www.turintech.ai", "My websites are  and ", 2),
            ("Look at these links: www.my.com:8069/tf/details?id=com.j.o%22 and ftp://amazon.com/g/G/e/2011/u-3.jpg", "Look at these links:  and ", 2),
            (None, None, 0),
        ],
    )
    def test_remove_url(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_url(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_url(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_url(text)


class TestUsername:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("RT @Stephan007: @Devoxx @collignont @idriss_neumann @John_Doe2000 @gunnarmorling @DevoxxFR @lescastcodeurs If interested, the Devoxx Belgium CFP opens en…",
             "RT :        If interested, the Devoxx Belgium CFP opens en…", 8),
            ("@probablyfaketwitterusername @RayFranco is answering to @AnPel, this is a real '@username83' but this is an@email.com, and this is a ",
             "@probablyfaketwitterusername  is answering to , this is a real '' but this is an@email.com, and this is a ", 3),
            (None, None, 0),
        ],
    )
    def test_remove_twitter_username(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_twitter_username(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_twitter_username(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_twitter_username(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("RT @Stephan007: @Devoxx @collignont @idriss_neumann @John_Doe2000 @gunnarmorling @DevoxxFR @lescastcodeurs If interested, the Devoxx Belgium CFP opens en…",
             "RT :        If interested, the Devoxx Belgium CFP opens en…", 8),
            ("@probablyfaketwitterusername @RayFranco is answering to @AnPel, this is a real '@username83' but this is an@email.com, and this is a ",
             "  is answering to , this is a real '' but this is an@email.com, and this is a ", 4),
            (None, None, 0),
        ],
    )
    def test_remove_username_any(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_username(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_username(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_username(text)


class TestHashtag:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("RT @iamffi23: #lahore #LahoreBlasts : #punjabgovt 9 #dead.", "RT @iamffi23:   :  9 .", 4),
            ("20 #injured. https://t.co/UU7doEtTmZ #pray_ee4 #unite #notgivingup", "20 . https://t.co/UU7doEtTmZ   ", 4),
            ("text #hashtag! text  #hashtag1 #hash_tagüäö text #hash0ta #hash_tag", "text ! text    text  ", 5),
            ("#хэш_тег #中英字典 #مهسا_امینی Not hashtags text #1234", "   Not hashtags text ", 4),
            ("", "", 0),
            (None, None, 0),
        ],
    )
    def test_remove_hashtag(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_hashtag(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_hashtag(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_hashtag(text)


class TestEmailAddress:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("type1: crisca@gmail.com.es", "type1: ", 1),
            ("type2: login@dom1.dom2.dom-3.dom-4.com", "type2: ", 1),
            ("type3: amin.cs@gmal.com", "type3: ", 1),
            ("type4: am_ghad@gmail.com", "type4: ", 1),
            ("", "", 0),
            (None, None, 0),
        ],
    )
    def test_remove_email_address(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_email_address(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_email_address(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text, _ = remove_email_address(text)


class TestChecker:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("\t Hello", False),
            ("      ", True),
            ("\t", True),
            ("\n \n", True),
            (None, None)
        ],
    )
    def test_blank_checker(self, input_text: Optional[str], ex_output: Optional[bool]):

        result = blank_checker(input_text)

        assert isinstance(result, (bool, type(None))), "The output text is not string."
        assert result == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_blank_checker(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = blank_checker(text)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (u"Klüft inför på fédéral électoral große", "Klüft inför på fédéral électoral große"),
            # ("This is a \x00hell\x08o wor\x9Fld sentence", "This is a hello world sentence"),
            # (None, None)
        ],
    )
    def test_convert_to_unicode(self, input_text: Optional[str], ex_output: Optional[bool]):
        # TODO: Fix this test
        result = convert_to_unicode(input_text)

        # assert isinstance(result, (bool, type(None))), "The output text is not string."
        # assert result == ex_output, "Expectation mismatch."

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, input_regex, ex_output",
        [
            (None, None, None),
            ("", None, None),
            (None, "", None),
            ("", "", ""),
            ("Hello, my id are 123 and 000.", r"\d+", "Hello, my id are  and .")
        ],
    )
    def test_remove_regex_match(self, input_text: Optional[str], input_regex: Optional[str], ex_output: Optional[bool]):

        result = remove_regex_match(input_text, input_regex)

        assert isinstance(result, (str, type(None))), "The output text is not string."
        assert result == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_remove_regex_match(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = remove_regex_match(text, r"\d+")

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, input_regex, input_sub_text, ex_output",
        [
            (None, None, None, None),
            ("", None, None, None),
            (None, "", None, None),
            (None, None, "", None),
            ("", "", "", ""),
            ("Hello, my id are 123 and 000.", r"\d+", "[NUM]", "Hello, my id are [NUM] and [NUM].")
        ],
    )
    def test_substitue_regex_match(self, input_text: Optional[str], input_regex: Optional[str], input_sub_text: Optional[str], ex_output: Optional[bool]):

        result = substitue_regex_match(input_text, input_regex, input_sub_text)

        assert isinstance(result, (str, type(None))), "The output text is not string."
        assert result == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_substitue_regex_match(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = substitue_regex_match(text, r"\d+", "hello")


class TestConversion:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            # https://www.slicktext.com/blog/2019/02/text-abbreviations-guide/
            ("the top FAQ are", "the top Frequently Asked Questions are"),
            ("the top faq are", "the top frequently asked questions are"),
            (None, None)
        ],
    )
    def test_abbreviation_converter(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = abbreviation_converter(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_abbreviation_converter(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = abbreviation_converter(text)


class TestAdder:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_word, ex_output_bool",
        [
            (None, False),
            ("", False),
            ([], False),
            ({}, False),
            ("book", True),
            (["book"], True),
            (["book", "star"], True),
            ({"book", "star"}, True),
        ],
    )
    def test_add_word_to_stopwords_set(self, input_word: Union[List, Set, str, None], ex_output_bool: Optional[bool]):

        english_stop_words = stopwords_nltk(["English"])
        result_stop_words = add_word_to_stopwords_set(english_stop_words, input_word)

        if isinstance(result_stop_words, type(None)):
            assert False == ex_output_bool, "Expectation mismatch."

        elif isinstance(input_word, type(None)):
            len_original_stop_words_set = len(english_stop_words)
            len_eddited_stop_words_set = len(result_stop_words)

            assert (len_original_stop_words_set == len_eddited_stop_words_set) and False == ex_output_bool, "Expectation mismatch."

        else:
            len_original_stop_words_set = len(english_stop_words)
            len_eddited_stop_words_set = len(result_stop_words)

            # Subset checker
            added_flag = False

            if isinstance(input_word, str):
                input_word_set = set([input_word])

            else:
                input_word_set = set(input_word)

            if input_word_set.issubset(result_stop_words):
                added_flag = True

            if len_original_stop_words_set == len_eddited_stop_words_set:
                added_flag = False

            assert added_flag == ex_output_bool, "Expectation mismatch."


class TestExpantion:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("I'm Amin.", "I am Amin."),
            ("It's a book.", "it is a book."),
            (None, None)
        ],
    )
    def test_expand_contractions(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = expand_contractions(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_expand_contractions(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = expand_contractions(text)


class TestLemmatization:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (None, None),
            ("", ""),
            ("The striped bats are hanging on their feet for best", "The striped bat be hang on their foot for best"),
        ],
    )
    def test_to_lemmatize(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = to_lemmatize(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_to_lemmatize(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = to_lemmatize(text)


class TestTokenization:
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (None, None),
            ("", []),
            ("Hello my name is Amin", ["Hello", "my", "name", "is", "Amin"]),
        ],
    )
    def test_to_tokenize(self, input_text: Optional[str], ex_output: Optional[List]):

        result = to_tokenize(input_text)

        assert isinstance(result, (List, type(None))), "The output text is not string."
        assert result == ex_output, "Expectation mismatch."

    @pytest.mark.parametrize(
        "n_samples",
        [
            (1000),
            (2000),
        ],
    )
    def test_perf_to_tokenize(self, n_samples: int):

        data = dbs.Synthetic_tweet_emotion_en(n_samples=n_samples)

        for text in data.get_text_list():
            result_text = to_tokenize(text)
