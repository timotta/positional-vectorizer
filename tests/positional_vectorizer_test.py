from unittest import TestCase

from scipy.sparse._csr import csr_matrix
from positional_vectorizer import PositionalVectorizer


class PositionalVectorizerTest(TestCase):

    def test_simple_basic(self):
        input_texts = ["my text here", "other text here"]

        vectorizer = PositionalVectorizer()
        vectorizer.fit(input_texts)

        assert vectorizer.vocabulary_ == {"my": 0, "text": 1, "here": 2, "other": 3}

        output_matrix = vectorizer.transform(input_texts)

        assert isinstance(output_matrix, csr_matrix)

        assert output_matrix.toarray().tolist() == [
            [1.0, 0.5906161091496412, 0.4765053580405043, 0.0],
            [0.0, 0.5906161091496412, 0.4765053580405043, 1.0],
        ]

    def test_ignore_duplicated(self):
        input_texts = ["my text here text", "other text here other"]

        vectorizer = PositionalVectorizer()
        vectorizer.fit(input_texts)

        assert vectorizer.vocabulary_ == {"my": 0, "text": 1, "here": 2, "other": 3}

        output_matrix = vectorizer.transform(input_texts)

        assert isinstance(output_matrix, csr_matrix)

        assert output_matrix.toarray().tolist() == [
            [1.0, 0.5906161091496412, 0.4765053580405043, 0.0],
            [0.0, 0.5906161091496412, 0.4765053580405043, 1.0],
        ]

    def test_ngram_range_word(self):
        input_texts = ["my text here", "other text here"]

        vectorizer = PositionalVectorizer(ngram_range=(1, 2))
        vectorizer.fit(input_texts)

        assert vectorizer.vocabulary_ == {
            "my": 0,
            "text": 1,
            "here": 2,
            "my text": 3,
            "text here": 4,
            "other": 5,
            "other text": 6,
        }

        output_matrix = vectorizer.transform(input_texts)

        assert isinstance(output_matrix, csr_matrix)

        assert output_matrix.toarray().tolist() == [
            [
                1.0,
                0.5906161091496412,
                0.4765053580405043,
                1.0,
                0.5906161091496412,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.5906161091496412,
                0.4765053580405043,
                0.0,
                0.5906161091496412,
                1.0,
                1.0,
            ],
        ]

    def test_ngram_range_char(self):
        input_texts = ["abc", "bcd"]

        vectorizer = PositionalVectorizer(ngram_range=(1, 2), analyzer="char")
        vectorizer.fit(input_texts)

        assert vectorizer.vocabulary_ == {
            "a": 0,
            "b": 1,
            "c": 2,
            "ab": 3,
            "bc": 4,
            "d": 5,
            "cd": 6,
        }
        output_matrix = vectorizer.transform(input_texts)

        assert isinstance(output_matrix, csr_matrix)

        assert output_matrix.toarray().tolist() == [
            [
                1.0,
                0.5906161091496412,
                0.4765053580405043,
                1.0,
                0.5906161091496412,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.5906161091496412,
                0.0,
                1.0,
                0.4765053580405043,
                0.5906161091496412,
            ],
        ]
