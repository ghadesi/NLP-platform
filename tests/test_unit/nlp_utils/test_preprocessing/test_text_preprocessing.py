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
# ───────────────────────────────── Tests ────────────────────────────────── #


class TestHTML:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("""<html lang="en" class="notranslate" translate="no" data-theme="light"><HEAD> This is HEAD <INSIDE> The is inside tag </INSIDE></HEAD> <BODY> This is BODY </BODY></HTML>""",
             " This is HEAD  The is inside tag   This is BODY ", 8),
            ("Hello world!", "Hello world!", 0),
            (None, None, None),
        ],
    )
    def test_remove_HTML_tags(self, input_text: Optional[str], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

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
    def test_remove_numbers(self, input_text: Optional[str], ex_output: Optional[str]):
        result_text = remove_number(input_text)
        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert result_text == ex_output, "Expectation mismatch."


class TestCharacter:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("!&  Hel_lo *@ $world!(", "  Hello  world", 8),
            (None, None, None),
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
            ("Hello, ", None, "Hello, ", None),
            (None, None, None, None),
        ],
    )
    def test_remove_special_char(self, input_text: Optional[str], input_spec_char: Optional[List[str]], ex_output_text: Optional[str], ex_num_matches: Optional[int]):

        result_text, result_matches = remove_special_char(input_text, input_spec_char)

        assert isinstance(result_text, (str, type(None))), "The output text is not string."
        assert isinstance(result_matches, (int, type(None))), "The number of matches shoulb be integer."
        assert result_text == ex_output_text and result_matches == ex_num_matches, "Expectation mismatch."


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
        assert result_text == ex_output, "Expectation mismatch."


class TestURL:
    @pytest.mark.parametrize(
        "input_text, ex_output_text, ex_num_matches",
        [
            ("My website is https://www.twanda.com/apps/details?id=com.skgames.trafficracer%22", "My website is ", 1),
            ("My websites are https://www.google.com and www.turintech.ai", "My websites are  and ", 2),
            ("Look at these links: www.my.com:8069/tf/details?id=com.j.o%22 and ftp://amazon.com/g/G/e/2011/u-3.jpg", "Look at these links:  and ", 2),
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
            (None, None, None),
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
