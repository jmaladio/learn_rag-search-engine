import string
from nltk.stem import PorterStemmer


def normalize_text(text: str) -> str:
    """
    Normalizes the input text by converting it to lowercase and removing punctuation.

    Args:
        text (str): The input string to normalize.
    Returns:
        str: The normalized string.
    """
    exclusion_table = str.maketrans('', '', string.punctuation)
    return text.casefold().translate(exclusion_table)

def tokenize_text(value: str) -> list[str]:
    """
    Tokenizes a string into a list of words.

    Args:
        value (str): The input string to tokenize.
    Returns:
        list[str]: A list of words.
    """

    return value.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes stopwords from a list of tokens.

    Args:
        tokens (list[str]): The list of tokens to filter.
    Returns:
        list[str]: A list of tokens with stopwords removed.
    """
    with open("data/stopwords.txt", "r") as f:
        stopwords = set(f.read().splitlines())

    return [token for token in tokens if token not in stopwords]

def stem_words(tokens: list[str]) -> list[str]:
    """
    Stems a list of tokens using the Porter stemming algorithm.

    Args:
        tokens (list[str]): The list of tokens to stem.
    Returns:
        list[str]: A list of stemmed tokens.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]