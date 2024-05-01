from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.base import BaseEstimator
from typing import Self
import numpy as np
from scipy.sparse._csr import csr_matrix
import math


class PositionalVectorizer(_VectorizerMixin, BaseEstimator):

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def fit(self, raw_documents, y=None) -> Self:
        self.vocabulary_ = self._build_vocabulary(raw_documents)
        return self

    def transform(self, raw_documents):
        self._check_vocabulary()

        analyze = self.build_analyzer()

        j_indices = []
        values = []
        indptr = [0]

        for doc in raw_documents:
            indexes = []
            rankings = []
            avoid_duplicateds = set()

            for position, feature in enumerate(analyze(doc)):
                if feature in self.vocabulary_ and feature not in avoid_duplicateds:
                    feature_idx = self.vocabulary_[feature]
                    indexes.append(feature_idx)
                    rankings.append(1 / (math.log(position + 1) + 1))
                    avoid_duplicateds.add(feature)

            j_indices.extend(indexes)
            values.extend(rankings)
            indptr.append(len(j_indices))

        X = csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.vocabulary_)),
        )

        return X

    def _build_vocabulary(self, raw_documents) -> dict:
        self._validate_ngram_range()
        self._warn_for_unused_params()
        self._validate_vocabulary()

        vocabulary = {}
        idx = 0

        analyze = self.build_analyzer()
        for doc in raw_documents:
            for feature in analyze(doc):
                if feature not in vocabulary:
                    vocabulary[feature] = idx
                    idx += 1

        return vocabulary
