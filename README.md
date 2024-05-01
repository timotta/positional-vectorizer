# Positional Vectorizer

The Positional Vectorizer is a transformer in scikit-learn designed to transform text into a bag of words vector using a positional ranking algorithm to assign scores. Similar to scikit-learn's CountVectorizer and TFIDFVectorizer, it assigns a value to each dimension based on the term's position in the original text.

## How to use

```bash
pip install positional-vectorizer
```

Using to generate de text vectors

```python
from positional_vectorizer import PositionalVectorizer

input_texts = ["my text here", "other text here"]

vectorizer = PositionalVectorizer()
vectorizer.fit(input_texts)

encoded_texts = vectorizer.transform(input_texts)
```

Using with scikit-learn pipeline

```python
from positional_vectorizer import PositionalVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

pipeline = Pipeline([
    ('vect', PositionalVectorizer(ngram_range=(1, 2))),
    ('clf', SGDClassifier(random_state=42, loss='modified_huber'))
])

pipeline.fit(X_train, y_train)
```

## Why this new vectorizer?

Text embeddings based on bag-of-words using count, binary, or TF-IDF normalization are highly effective in most scenarios. However, in certain cases, such as those involving languages like Latin, the position of terms becomes crucial, which these techniques fail to capture.

For instance, consider the importance of word position in a Portuguese classification task distinguishing between a smartphone device and a smartphone accessory. In traditional bag-of-words approaches with stop words removed, the following titles yield identical representations:

* "xiaomi com fone de ouvido" => {"xiaomi", "fone", "ouvido"}
* "fone de ouvido do xiaomi" => {"xiaomi", "fone", "ouvido"}

As demonstrated, the order of words significantly alters the meaning, but this meaning is not reflected in the vectorization.

One common workaround is to employ n-grams instead of single words, but this can inflate the feature dimensionality, potentially increasing the risk of overfitting.

## How it works

The value in each dimension is calculated as `1 / math.log(rank + 1)` (similar to the Discounted Cumulative Gain formula), where the rank denotes the position of the corresponding term, starting from 1.

If a term appears multiple times in the text, only its lowest rank is taken into account.

## TODO

* Test the common parameters of _VectorizerMixin to identify potential issues when upgrading scikit-learn. Currently, only the `ngrams_range` and `analyzer` parameters are automatically tested.
* Implement the max_features parameter.

