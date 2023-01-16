"""Module providing utility dbs for developers to test the performance of the preprocessing moduls. """
# ───────────────────────────────── Imports ────────────────────────────────── #
# Standard library
from typing import List, Union
import pandas as pd
import warnings

# 3rd Party

# Private

# ───────────────────────────────── DBs Class ────────────────────────────────── #


class Synthetic_tweet_emotion_en:
    """
        Synthetic tweet emotion database for testing the preprocessing module.
    """

    def __init__(self, n_samples: int = 100, seed: int = 100):
        self.n_samples = n_samples
        self.seed = seed

        tweet_df = pd.read_csv("./Datasets/Twitter_DS_emotion.txt", sep="\t", header=None, names=["Index", "Text", "Feeling"], index_col=None)
        self.selected_tweet_df = tweet_df.sample(n=n_samples, random_state=seed)

    def __repr__(self) -> str:
        return f"Synthetic_tweet_emotion_en_db(n_samples={self.n_samples}, seed={self.seed})"

    def get_text_list(self) -> List:
        return self.selected_tweet_df.Text.values.tolist()

    def get_df(self) -> pd.DataFrame:
        return self.selected_tweet_df

    def get_label(self) -> List:
        return self.selected_tweet_df.Feeling.values.tolist()

    def get_row_info(self, text: str) -> pd.DataFrame:
        return self.selected_tweet_df.loc[self.selected_tweet_df['Texr'] == text]


class Synthetic_tweet_multi_language:
    """
        Synthetic tweet multi language database for testing the preprocessing module.
    """

    def __init__(self, n_samples: int = 100, seed: int = 100, language: Union[None, str, List] = "en"):

        tweet_df = pd.read_csv("./Datasets/Twitter_DS_multi_language.csv", header=0, lineterminator='\n')
        self.language = self.__language_support(tweet_df, language)

        # Shuffle the dataset
        tweet_df = tweet_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        tweet_df_pref_lan = tweet_df[tweet_df.tweet_language.isin(self.language)].reset_index()
        self.selected_tweet_df = tweet_df.sample(n=n_samples, random_state=seed)

        self.n_samples = n_samples
        self.seed = seed

    def __language_support(self, tweet_df: pd.DataFrame, language: Union[None, str, List]) -> List:
        """
        [Private method] Check if the database supports the preferred language.

        Args:
            tweet_df (pd.dataFrame): input database
            language (Union[None, str, List]): preferred language

        Raises:
            TypeError: if language is not None, str or list

        Warnings:
            RuntimeWarning: if the database does not support the preferred language

        Returns:
            List: returns a list of languages supported by the database based on the preferred language.
        """

        if language is None:
            language = ["en"]
        elif isinstance(language, str):
            anguage = [language]
        elif isinstance(language, list):
            language = language
        else:
            raise TypeError(f"language must be None, str or list. Got {type(language)}")

        db_sup_lang = set(tweet_df.tweet_language.value_counts().index)
        pref_lang = set(language)

        if not pref_lang.issubset(db_sup_lang):
            warnings.warn(f"Database does not support the following languages: {pref_lang - db_sup_lang}", RuntimeWarning, stacklevel=1)
            return list(pref_lang & db_sup_lang)

        return language

    def __repr__(self) -> str:
        return f"Synthetic_tweet_multi_language_db(n_samples={self.n_samples}, seed={self.seed}, language={self.language})"

    def get_text_list(self) -> List:
        return self.selected_tweet_df.tweet_full_text.values.tolist()

    def get_df(self) -> pd.DataFrame:
        return self.selected_tweet_df

    def get_row_info(self, text: str) -> pd.DataFrame:
        return self.selected_tweet_df.loc[self.selected_tweet_df['tweet_full_text'] == text]
