"""Module providing utility dbs for developers to test the performance of the preprocessing moduls. """
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
import pandas as pd
from typing import List

# ───────────────────────────────── DBs Class ────────────────────────────────── #


class Synthetic_tweet_emotion_en_db:
    def __init__(self, n_samples: int = 100, seed: int = 100):
        self.n_samples = n_samples
        self.seed = seed
        
        tweet_df = pd.read_csv("./Datasets/Twitter_DS.txt", sep="\t", header=None, names=["Index", "Text", "Feeling"])
        self.selected_tweet_df = tweet_df.sample(n=n_samples, random_state=seed)

    def __repr__(self) -> str:
        return f"Synthetic_tweet_emotion_en_db(n_samples={self.n_samples}, seed={self.seed})"

    def get_list(self) -> List:
        return self.selected_tweet_df.Text.values.tolist()

    def get_df(self) -> pd.DataFrame:
        return self.selected_tweet_df

    def get_label(self) -> List:
        return self.selected_tweet_df.Feeling.values.tolist()

    def get_row_info(self, text: str) -> pd.DataFrame:
        return self.selected_tweet_df.loc[self.selected_tweet_df['Texr'] == text]


class Synthetic_tweet_emotion_en_db:
    def __init__(self, n_samples: int = 100, seed: int = 100):
        self.n_samples = n_samples
        self.seed = seed

        tweet_df = pd.read_csv("./Datasets/Twitter_DS_emotion.txt", sep="\t", header=None, names=["Index", "Text", "Feeling"])
        self.selected_tweet_df = tweet_df.sample(n=n_samples, random_state=seed)

    def __repr__(self) -> str:
        return f"Synthetic_tweet_emotion_en_db(n_samples={self.n_samples}, seed={self.seed})"

    def get_list(self) -> List:
        return self.selected_tweet_df.Text.values.tolist()

    def get_df(self) -> pd.DataFrame:
        return self.selected_tweet_df

    def get_label(self) -> List:
        return self.selected_tweet_df.Feeling.values.tolist()

    def get_row_info(self, text: str) -> pd.DataFrame:
        return self.selected_tweet_df.loc[self.selected_tweet_df['Texr'] == text]
