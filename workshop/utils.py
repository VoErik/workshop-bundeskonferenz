
import re
import random
from collections import Counter
from typing import Literal, get_args

Level = Literal["char", "word"]
BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,!?;:\-']|\n") # get individual words
CLEAN_RE = re.compile(r"[^A-Za-z\s.,!?;:\-]") # gets letters + .,!?;:- (removes odd special chars and numbers)
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "-", "'"}
NO_SPACE_AFTER = {"\n", "-", "'", EOS}


def load_data(
        source: str = "assets/tiny_shakespeare.txt",
        split: bool = False,
) -> str | tuple[str, str]:
    """
    Loads dataset from a .txt file.
    TODO: make more general for transformer part? @Johannes

    :param source: location of dataset
    :param split: whether to split dataset into train and validation
    :return: train and validation dataset
    """
    with open(source, "r", encoding="utf-8") as f:
        text = f.read()

    if split:
        i = int(len(text) * 0.9)
        train_text, validation_text = text[:i], text[i:]
        text = (train_text, validation_text)

    return text

def to_chunks(text, separator="\n\n"):
    return text.split(separator) if separator else [text]

def tokenize(
        text: str,
        level: Level = "char",
        lowercase: bool = True
) -> list[str]:
    """Tokenizes text into individual tokens.

    :param text: text to tokenize
    :param level: level of tokens to tokenize
    :param lowercase: lowercase or not
    :return: list of individual tokens
    """
    assert level in get_args(Level), f"level must be one of {get_args(Level)}"
    text = CLEAN_RE.sub("", text)
    if lowercase:
        text = text.lower()
    if level == "char":
        return list(text)
    if level == "word":
        return WORD_RE.findall(text)

def _join_piece(
        prev: str | None,
        tok: str,
        level: Level
) -> str:
    """Text to emit for `tok` given the previous token needed for streaming."""
    if level == "char":
        return tok
    if (prev is None
            or tok == "\n"
            or prev in NO_SPACE_AFTER
            or tok in NO_SPACE_BEFORE
    ):
        return tok
    return " " + tok

def detokenize(
        tokens: list[str],
        level: Level
) -> str:
    """
    Creates string of text from tokens, removing white spaces where appropriate.

    :param tokens: list of tokens
    :param level: char or word

    :return: joined string of tokens
    """
    out, prev = [], None
    for tok in tokens:
        out.append(_join_piece(prev, tok, level))
        prev = tok
    return "".join(out)

def build_vocabulary(
        tokens: list[str]
) -> list[str]:
    """Create sorted vocabulary from tokens and prepends special tokens."""
    counts = Counter(tokens)
    vocab = sorted(t for t, c in counts.items())
    return [BOS, EOS, UNK] + vocab

def prepare_data(
        dataset: str,
        level: str,
        separator: str = "\n\n",
):
    data = load_data(source=dataset)
    chunks = [tokenize(c, level) for c in to_chunks(text=data, separator=separator) if c.strip()]
    vocabulary = build_vocabulary([t for doc in chunks for t in doc])
    return chunks, vocabulary

def make_toy_dataset(
        path: str = "assets/toy.txt",
        vocabulary: str | list[str] = "abcde ",
        length: int = 20_000,
        alpha: float = 0.3,
        seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Creates a toy dataset with limited vocabulary.

    :param path: where to save the dataset
    :param vocabulary: vocabulary to use
    :param length: length of dataset
    :param alpha: strength of transition probability. The larger, the more random it is.
    :param seed: random seed

    :returns: Dictionary with transition statistics
    """
    rng = random.Random(seed)
    vocab = list(vocabulary)
    transitions = {}
    for tok in vocab:
        weights = [rng.gammavariate(alpha, 1) for _ in vocab]
        total = sum(weights)
        transitions[tok] = {t: w / total for t, w in zip(vocab, weights)}

    tokens = [rng.choice(vocab)]
    for _ in range(length - 1):
        options = transitions[tokens[-1]]
        tokens.append(rng.choices(list(options), weights=list(options.values()))[0])

    sep = "" if isinstance(vocabulary, str) else " "
    with open(path, "w", encoding="utf-8") as f:
        f.write(sep.join(tokens))
    return transitions