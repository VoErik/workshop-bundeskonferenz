
import re
import random
from collections import Counter
from typing import Literal, get_args

Level = Literal["char", "word"]
BOS, EOS, UNK = "<bos>", "<eos>", "<unk>"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[.,!?;:\-']|\n") # individuelle Worte
CLEAN_RE = re.compile(r"[^A-Za-z\s.,!?;:\-]") # Buchstaben + .,!?;:- (Entfernt Sonderzeichen und Zahlen)
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "-", "'"}
NO_SPACE_AFTER = {"\n", "-", "'", EOS}


def lade_datensatz(
        quelle: str = "assets/tiny_shakespeare.txt",
        split: bool = False,
) -> str | tuple[str, str]:
    """
    Laedt Datensatz aus .txt Datei.

    :param source: Speicherort des Datensatzes
    :param split: Soll der Datensatz in 'Training' und 'Validation' gesplittet werden? Default = False.
    :return: Datensatz splits
    """
    with open(quelle, "r", encoding="utf-8") as f:
        text = f.read()

    if split:
        i = int(len(text) * 0.9)
        train_text, validation_text = text[:i], text[i:]
        text = (train_text, validation_text)

    return text

def als_chunks(text, separator="\n\n"):
    return text.split(separator) if separator else [text]

def tokenisiere(
        text: str,
        level: Level = "char",
        kleinschreibung: bool = True
) -> list[str]:
    """Tokenisiert Text.

    :param text: Zu tokenisierender Text
    :param level: Level der Tokenisierung (Wort, Zeichen)
    :param lowercase: Nur Kleinschreibung? Default = True
    :return: Liste der individuellen Token
    """
    assert level in get_args(Level), f"Level muss eins der folgenden sein: {get_args(Level)}"
    text = CLEAN_RE.sub("", text)
    if kleinschreibung:
        text = text.lower()
    if level == "char":
        return list(text)
    if level == "word":
        return WORD_RE.findall(text)

# nur intern
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

def detokeninisiere(
        tokens: list[str],
        level: Level
) -> str:
    """
    Creates string of text from tokens, removing white spaces where appropriate.

    :param tokens: Liste der Tokens
    :param level: char oder word

    :return: Zusammengefuegte Liste der Tokens
    """
    out, prev = [], None
    for tok in tokens:
        out.append(_join_piece(prev, tok, level))
        prev = tok
    return "".join(out)

def baue_vokabular(
        tokens: list[str]
) -> list[str]:
    """Erstellt sortiertes Vokabular aus den Tokens mit Sondertokens."""
    counts = Counter(tokens)
    vocab = sorted(t for t, c in counts.items())
    return [BOS, EOS, UNK] + vocab

def bereite_daten_vor(
        datensatz: str,
        level: str,
        separator: str = "\n\n",
):
    data = lade_datensatz(quelle=datensatz)
    chunks = [tokenisiere(c, level) for c in als_chunks(text=data, separator=separator) if c.strip()]
    vocabulary = baue_vokabular([t for doc in chunks for t in doc])
    return chunks, vocabulary

# nur intern
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