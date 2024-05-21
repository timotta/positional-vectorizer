from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.base import BaseEstimator
from typing import List
from scipy.sparse._csr import csr_matrix
import math

# Python 3.10 has no Self on typing module
try:
    from typing import Self  # type: ignore[attr-defined]
except Exception:
    from typing_extensions import Self


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
        vocabulary=None,
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
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary

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

            for feature_list_for_gram in self._split_features_grams(analyze(doc)):
                for rank, feature in enumerate(feature_list_for_gram):
                    if feature in self.vocabulary_ and feature not in avoid_duplicateds:
                        feature_idx = self.vocabulary_[feature]
                        indexes.append(feature_idx)
                        rankings.append(1 / (math.log(rank + 1) + 1))
                        avoid_duplicateds.add(feature)

            j_indices.extend(indexes)
            values.extend(rankings)
            indptr.append(len(j_indices))

        X = csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.vocabulary_)),
        )

        return X

    def _split_features_grams(self, feature_list: List[str]) -> List[List[str]]:
        """Split a list of features into list of a list of features. Each list of features is a representation
        of one of the n-grams range. We assume here that the features are ordered by the ngram representation
        If scikit-learn change this behavior, the tests are going to fail and this method will need to be updated.

        Parameters
        ----------
        feature_list: List[str]
            List of features

        Returns
        -------
        ngrams: List[List[str]]
            A list of lists of features where each list of features is a representation of one of the n-grams kind

        Examples
        --------
        >>> _split_features_grams(['xpto', 'blenga', 'xpto blenga'])
        [['xpto', 'blenga'], ['xpto blenga']]
        """

        wich_gram = len if self.analyzer == "char" else self._wich_nram_by_space

        last_gram = 0
        result = []
        actual_gram_result: List[str] = []

        for feature in feature_list:
            actual_gram = wich_gram(feature)
            if actual_gram > last_gram:
                last_gram = actual_gram
                if actual_gram_result:
                    result.append(actual_gram_result)
                    actual_gram_result = []
            actual_gram_result.append(feature)
        if actual_gram_result:
            result.append(actual_gram_result)
        return result

    def _wich_nram_by_space(self, feature: str) -> int:
        return feature.count(" ") + 1

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
