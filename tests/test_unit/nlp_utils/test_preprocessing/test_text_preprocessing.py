"""Module providing utils testing for users to convert data."""
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard library
from typing import List, Union, Any, Optional
import pandas as pd
import numpy as np
import sys

# 3rd Party
import pytest

# Private
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
from nlp_utils.preprocessing.cleaner_helper import custom_extended_stopwords, custom_shortforms, custom_direct_replacement_dict

# ───────────────────────────────── Tests ────────────────────────────────── #
# from tests.evoml_utils.ml_model.explain.explain_test_utils import (_synthetic_classification)

# if self.is_binary:
#     pytest.skip("Test only applies for functions with multiclass enabled")

#  data = _synthetic_classification(n_rows=100, n_unique_y=n_unique_y)
#  inputs = {"y_test": data.y, "y_scores": data.y_prob, "unique_y": data.unique_y, "unique_labels": data.unique_labels}
 
# @staticmethod
# def output_check(output: Any, **kwargs):
#     # No checks being done on parent class, must be implemented in child class
#     warnings.warn("No output checks have been implemented")
# @staticmethod
# def output_check(output: Any, output_expected: bool = True, positive_class: Optional[Any] = None, **kwargs):
#     if not output_expected:
#         assert output is None
#         return
#     assert isinstance(output, dict)
#     assert output["type"] == "rocCurve", "output['type'] value is not 'rocCurve'."
#     if positive_class is not None:
#         assert len(output["graphJson"]["data"]) == 1
#         assert output["graphJson"]["data"][0]["name"] == positive_class
            
# self.output_check(output, output_expected=output_expected)

# import sys
# import pytest


# @pytest.mark.parametrize(
#     ("n", "expected"),
#     [
#         (1, 2),
#         pytest.param(1, 0, marks=pytest.mark.xfail),
#         pytest.param(1, 3, marks=pytest.mark.xfail(reason="some bug")),
#         (2, 3),
#         (3, 4),
#         (4, 5),
#         pytest.param(
#             10, 11, marks=pytest.mark.skipif(sys.version_info >= (3, 0), reason="py2k")
#         ),
#     ],
# )
# def test_increment(n, expected):
#     assert n + 1 == expected


# For the test Repetition 
# https://pypi.org/project/pytest-repeat/
# $ pip install pytest-aggreport
# pip install pytest-repeat 
# pytest --count=10 -x test_file.py
    # x: to force the test runner to stop at the first failure
# Usage: @pytest.mark.repeat(3)

# HTML report: https://pytest-html.readthedocs.io/en/latest/user_guide.html
#  pip install pytest-html
# pytest --html=report.html --self-contained-html

class TestHTML:
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


class TestToStrip:
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


class TestToLower:
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


class TestNumbers:
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello123 678 1000 44word 5", "Hello   word "),
            (" 5", " "),
            (None, None)
        ],
    )
    def test_remove_number(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_number(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."


class TestCharacter:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("!&  Hel_lo *@ $world!(", "  Hello  world", 8),
            (None, None, 0),
        ],
    )
    def test_remove_any_char(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_any_char(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "input_text, input_spec_char, ex_output_text, ex_num_matches",
        [
            ("Hello, ¿the first char is ٪ and these onces《 》!٪", ["٪", "《", "》", "¿"], "Hello, the first char is  and these onces !", 5),
            ("Hello, ", None, "Hello, ", 0),
            (None, None, None, 0),
        ],
    )
    def test_remove_special_char(self, input_text: Optional[str], input_spec_char: Optional[List[str]], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_special_char(input_text, input_spec_char)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."

    @pytest.mark.parametrize(
        "input_text, ex_output_text",
        [
            ("James: \"Hi Thomas, I haven't seen you for ages! How have you been?\"", "James Hi Thomas I havent seen you for ages How have you been"),
            ("!hi. wh?at is the weat[h]er lik?e.", "hi what is the weather like"),
            (None, None),
        ],
    )
    def test_remove_punctuation(self, input_text: Optional[str], ex_output_text: Optional[str]):

        result_text = remove_punctuation(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output_text, "Expectation mismatch."

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



class TestDuplication:
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


class TestEmpjiEmoticons:
    @pytest.mark.parametrize(
        "input_text, ex_output, ex_num_matches",
        [
            ("RT @ kaastore: 😁 un sourire=un cadeau 🎁", "RT @ kaastore:  un sourire=un cadeau ", 2),
            (None, None, 0),
        ],
    )
    def test_remove_emoji(self, input_text: Optional[str], ex_output: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_emoji(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output and result_matches == ex_num_matches, "Expectation mismatch."

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


class TestURL:
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


class TestUsername:
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


class TestHashtag:
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


class TestEmailAddress:
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


class TestChecker:
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


class TestConversion:
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


class TestExpantion:
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

    # @pytest.mark.skip
    # @pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
    # @pytest.mark.repeat(5)
    @pytest.fixture
    def test_speed(self):
        px = np.sort(np.random.default_rng().normal(0, 1, 1000000))
        py = np.sort(np.random.default_rng().normal(0, 1, 1000000))

