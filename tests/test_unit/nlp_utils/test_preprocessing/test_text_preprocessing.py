
"""Module providing utils testing for users to convert data."""
# ───────────────────────────────── Imports ────────────────────────────────── #

# Standard library
from typing import List, Union, Any, Optional
import pandas as pd

# 3rd Party
import pytest

# Private
from nlp_utils.preprocessing.text_preprocessing import remove_xml
from nlp_utils.preprocessing.text_preprocessing import to_strip
from nlp_utils.preprocessing.text_preprocessing import to_lower
from nlp_utils.preprocessing.text_preprocessing import remove_number
from nlp_utils.preprocessing.text_preprocessing import remove_any_char
from nlp_utils.preprocessing.text_preprocessing import remove_duplication
from nlp_utils.preprocessing.text_preprocessing import remove_many_spaces
from nlp_utils.preprocessing.text_preprocessing import remove_emoji
from nlp_utils.preprocessing.text_preprocessing import remove_url
from nlp_utils.preprocessing.text_preprocessing import remove_twitter_username
from nlp_utils.preprocessing.text_preprocessing import remove_username
# ───────────────────────────────── Tests ────────────────────────────────── #


class TestHTML:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_maches",
        [
            ("""<html lang="en" class="notranslate" translate="no" data-theme="light"><HEAD> This is HEAD <INSIDE> The is inside tag </INSIDE></HEAD> <BODY> This is BODY </BODY></HTML>""",
             " This is HEAD  The is inside tag   This is BODY ", 8),
            ("Hello world!", "Hello world!", 0),
            (None, None, None),
        ],
    )
    def test_remove_HTML_tags(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_maches: Optional[int]):

        result_text, result_maches = remove_xml(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_maches, (int, type(None))), "The number of maches shoulb be integer."
        assert result_text == ex_output_text and result_maches == ex_num_maches, "Expection mismatch."


class TestToStrip:
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            (" Hello world   ! ", "Hello world !"),
            (None, None)
        ],
    )
    def test_to_strip(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = to_strip(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expection mismatch."


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
        assert result_text == ex_output, "Expection mismatch."


class TestNumbers:
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello123 678 1000 44word 5", "Hello   word "),
            (None, None)
        ],
    )
    def test_remove_numbers(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_number(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expection mismatch."


class TestCharacter:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_maches",
        [
            ("!&  Hel_lo *@ $world!(", "  Hello  world", 8),
            (None, None, None),
        ],
    )
    def test_remove_chars(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_maches: Optional[int]):

        result_text, result_maches = remove_any_char(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_maches, (int, type(None))), "The number of maches shoulb be integer."
        assert result_text == ex_output_text and result_maches == ex_num_maches, "Expection mismatch."


class TestDuplication:
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello hello", "Hello hello"),
            ("Hello world world! world Hello world", "Hello world world!"),
            (None, None)
        ],
    )
    def test_remove_duplication(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_duplication(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expection mismatch."

    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("Hello          world", "Hello world"),
            (None, None)
        ],
    )
    def test_many_spaces(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_many_spaces(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expection mismatch."


class TestEmpji:
    @pytest.mark.parametrize(
        "input_text, ex_output",
        [
            ("RT @ kaastore: 😁 un sourire=un cadeau 🎁", "RT @ kaastore:  un sourire=un cadeau "),
            (None, None),
        ],
    )
    def test_remove_emojis(self, input_text: Optional[str], ex_output: Optional[str]):

        result_text = remove_emoji(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expection mismatch."


class TestURL:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_maches",
        [
            ("My website is https://www.twanda.com/apps/details?id=com.skgames.trafficracer%22", "My website is ", 1),
            ("Look at these links: www.my.com:8069/tf/details?id=com.j.o%22 and ftp://amazon.com/g/G/e/2011/u-3.jpg", "Look at these links:  and ", 2),
            (None, None, None),
        ],
    )
    def test_remove_urls(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_maches: Optional[int]):

        result_text, result_maches = remove_url(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_maches, (int, type(None))), "The number of maches shoulb be integer."
        assert result_text == ex_output_text and result_maches == ex_num_maches, "Expection mismatch."


class TestUsername:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_maches",
        [
            ("RT @Stephan007: @Devoxx @collignont @idriss_neumann @John_Doe2000 @gunnarmorling @DevoxxFR @lescastcodeurs If interested, the Devoxx Belgium CFP opens en…", 
             "RT :        If interested, the Devoxx Belgium CFP opens en…", 8),
            ("@probablyfaketwitterusername @RayFranco is answering to @AnPel, this is a real '@username83' but this is an@email.com, and this is a ",
             "@probablyfaketwitterusername  is answering to , this is a real '' but this is an@email.com, and this is a ", 3),
            (None, None, None),
        ],
    )
    def test_remove_twitter_username(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_maches: Optional[int]):

        result_text, result_maches = remove_twitter_username(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_maches, (int, type(None))), "The number of maches shoulb be integer."
        assert result_text == ex_output_text and result_maches == ex_num_maches, "Expection mismatch."

    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_maches",
        [
            ("RT @Stephan007: @Devoxx @collignont @idriss_neumann @John_Doe2000 @gunnarmorling @DevoxxFR @lescastcodeurs If interested, the Devoxx Belgium CFP opens en…",
             "RT :        If interested, the Devoxx Belgium CFP opens en…", 8),
            ("@probablyfaketwitterusername @RayFranco is answering to @AnPel, this is a real '@username83' but this is an@email.com, and this is a ",
             "  is answering to , this is a real '' but this is an@email.com, and this is a ", 4),
            (None, None, None),
        ],
    )
    def test_remove_any_username(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_maches: Optional[int]):

        result_text, result_maches = remove_username(input_text)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_maches, (int, type(None))), "The number of maches shoulb be integer."
        assert result_text == ex_output_text and result_maches == ex_num_maches, "Expection mismatch."
