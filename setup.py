from os import environ
from typing import List
import setuptools


def read_multiline_as_list(file_path: str) -> List[str]:
    with open(file_path) as fh:
        contents = fh.read().split("\n")
        if contents[-1] == "":
            contents.pop()
        return [c for c in contents if not c.startswith("--")]


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = read_multiline_as_list("requirements.txt")


version = environ["POSITIONAL_VECTORIZER_VERSION"]

setuptools.setup(
    name="positional-vectorizer",
    version=version,
    author="Tiago Albineli Motta",
    author_email="timotta@gmail.com",
    description="Positional Vectorizer is a scikit-learn transformer that converts text to bag of words vector using a positional ranking algorithm as score",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timotta/positional-vectorizer",
    packages=setuptools.find_packages(
        include=["positional_vectorizer", "positional_vectorizer.*"]
    ),
    include_package_data=True,
    keywords="machine learning, embedding, vectorizer, scikit-learn, text, NLP",
    entry_points={
        "console_scripts": [
            # '',
        ],
    },
    python_requires=">=3.11, <3.12",
    install_requires=requirements,
)
