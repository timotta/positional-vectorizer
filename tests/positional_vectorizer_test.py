from unittest import TestCase

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse._csr import csr_matrix


class PositionalVectorizerTest(TestCase):

    def test_counter(self):
        input_texts = ["my text here", "other text here"]

        vectorizer = CountVectorizer()
        vectorizer.fit(input_texts)

        assert vectorizer.vocabulary_ == {"my": 1, "text": 3, "here": 0, "other": 2}

        print(vectorizer.transform(input_texts).toarray().tolist())

        output_matrix = vectorizer.transform(input_texts)

        assert isinstance(output_matrix, csr_matrix)

        assert output_matrix.toarray().tolist() == [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
        ]
