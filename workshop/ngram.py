import random
import time
from html import escape
from collections import (
    Counter,
    defaultdict
)
from workshop.utils import (
    BOS,
    EOS,
    UNK,
    Level,
    tokenize,
    detokenize,
    _join_piece
)

def _char_label(ch: str) -> str:
    return {"\n": "⏎", " ": "␣", "\t": "⇥"}.get(ch, ch)

class NGram:
    """
    Simple N-Gram model that works for characters or words.

    :param n: number of n-gram to use
    :param level: level of n-gram to use (word or char)
    """
    def __init__(
            self,
            n: int = 2,
            level: Level = "char"
    ):
        if n < 1: raise ValueError("n must be > 0")
        self.n = n
        self.level = level
        self.vocab = None
        self.vocab_set = None
        self.counts = None

    def _map(self, tok: str) -> str:
        return tok if tok in self.vocab_set else UNK

    def _context(
            self,
            history: list[str],
            order: int
    ) -> tuple[str, ...]:
        """Last (order-1) tokens of history, left-padded with <bos>.
        Anything up to and including an <eos> is dropped, so a new document starts fresh."""
        size = order - 1
        recent = history[max(0, len(history) - size):] if size else []
        if EOS in recent:
            recent = recent[len(recent) - recent[::-1].index(EOS):]
        padded = [BOS] * size + [self._map(t) for t in recent]
        return tuple(padded[len(padded) - size:])

    def fit(
            self,
            docs: list[list[str]],
            vocabulary: list[str]
    ) -> "NGram":
        """
        Fits the model to the data.

        :param docs: list of lists of tokens
        :param vocabulary: the allowed vocabulary
        """
        self.vocab = vocabulary
        self.vocab_set = set(vocabulary)
        # counts[order-1][context][next_token]; lower orders are kept for backoff and the matrix
        self.counts = [defaultdict(Counter) for _ in range(self.n)]

        for doc in docs:
            seq = [BOS] * (self.n - 1) + [self._map(t) for t in doc] + [EOS]
            for i in range(self.n - 1, len(seq)):
                tok = seq[i]
                for order in range(1, self.n + 1):
                    ctx = tuple(seq[i - order + 1:i])
                    self.counts[order - 1][ctx][tok] += 1
        return self

    def _sample(
            self,
            history: list[str],
            rng: random.Random,
            temperature: float
    ) -> str:
        """
        Samples  the  next token.

        :param history: list of tokens
        :param rng: random seed
        :param temperature: temperature to smooth out the sample distribution
        :returns: next token
        """
        # back off to shorter contexts if the full one was never seen
        for order in range(self.n, 0, -1):
            dist = self.counts[order - 1].get(self._context(history, order))
            if dist:
                break
        choices = [t for t in dist if t != UNK] or list(dist)
        weights = [dist[t] ** (1.0 / temperature) for t in choices]
        return rng.choices(choices, weights=weights)[0]

    def generate(
            self,
            length: int=200,
            prompt: str="",
            *,
            temperature: float=1.0,
            seed: int | None =None,
            stream: bool=True,
            delay: float = 0.05
    ) -> str:
        """
        Repeatedly samples the distribution to generate new tokens.

        :param length: how many new tokens are generated
        :param prompt: starting sequence
        :param temperature: factor to smooth out the sampling distribution
        :param seed: random seed
        :param stream: whether to stream the result (defaults to True)
        :param delay: delay between repetitions when streaming

        :return: the newly generated tokens
        """
        rng = random.Random(seed)
        history = tokenize(prompt, self.level) if isinstance(prompt, str) else list(prompt)
        text = detokenize(history, self.level)
        if stream:
            print(text, end="", flush=True)
        for _ in range(length):
            tok = self._sample(history, rng, temperature)
            if tok == EOS:
                piece = "\n\n"
            else:
                piece = _join_piece(history[-1] if history else None, tok, self.level)
            history.append(tok)
            text += piece
            if stream:
                print(piece, end="", flush=True)
                time.sleep(delay)
        return text

    # From here on out its just pretty visualisations and convenience.
    # Nothing here is "essential" for the N-Gram to work.

    def association_matrix(
            self,
            chars: str | list[str] | None = None,
            normalize: bool = True,
            order: int | None = None,
            top: int | None = 40,
    ) -> tuple[list[str], list[str], list[list[float]]]:
        """Rows = contexts (order-1 chars), columns = next char.

        order: 2 = bigram, 3 = trigram, ... (defaults to the model's n)
        top: only show the `top` most frequent contexts (None = all observed ones)
        """
        order = order or self.n
        if self.level != "char":
            raise ValueError("association matrix is only available for char-level models")
        if not 2 <= order <= self.n:
            raise ValueError(f"order must be between 2 and n={self.n}")

        cols = list(chars) if chars is not None else [t for t in self.vocab if t not in (BOS, EOS, UNK)]
        allowed = set(cols)
        table = self.counts[order - 1]

        contexts = [ctx for ctx in table if all(c in allowed for c in ctx)]
        contexts.sort(key=lambda ctx: sum(table[ctx].values()), reverse=True)
        if top is not None:
            contexts = contexts[:top]

        rows = []
        for ctx in contexts:
            counter = table[ctx]
            row = [counter[b] for b in cols]
            if normalize:
                total = sum(counter.values())
                row = [x / total for x in row]
            rows.append(row)
        row_labels = ["".join(_char_label(c) for c in ctx) for ctx in contexts]
        return row_labels, cols, rows

    def show_association_matrix(
            self,
            chars: str | list[str] | None = None,
            normalize: bool = True,
            order: int | None = None,
            top: int | None = 40,
            show_values: bool = False,
            gamma: float = 0.5,
    ) -> None:
        row_labels, cols, rows = self.association_matrix(chars, normalize, order, top)
        col_labels = [_char_label(c) for c in cols]
        peak = max((v for row in rows for v in row), default=0) or 1
        try:
            from IPython.display import HTML, display
        except ImportError:
            print(self._matrix_text(row_labels, col_labels, rows, peak, gamma))
            return
        display(HTML(self._matrix_html(row_labels, col_labels, rows, peak, gamma, normalize, show_values)))

    # thank you AI overlord :)
    @staticmethod
    def _matrix_html(row_labels, col_labels, rows, peak, gamma, normalize, show_values) -> str:
        size = 26 if show_values else 18
        head_style = ("font-weight:600;text-align:center;padding:0;"
                      f"width:{size}px;height:{size}px;color:inherit;background:none;")
        row_head_style = ("font-weight:600;text-align:right;padding:0 6px;white-space:pre;"
                          f"height:{size}px;color:inherit;background:none;")
        cell_style = (f"width:{size}px;height:{size}px;padding:0;text-align:center;"
                      "font-size:9px;border:1px solid rgba(128,128,128,0.15);")

        def fmt(v):
            if not v:
                return ""
            return f"{v:.2f}".lstrip("0") if normalize else str(v)

        out = [
            '<div style="overflow:auto;max-height:80vh;">',
            '<table style="border-collapse:collapse;font-family:monospace;font-size:12px;">',
            f'<tr><th style="{row_head_style}">↓→</th>',
        ]
        out += [f'<th style="{head_style}">{escape(b)}</th>' for b in col_labels]
        out.append("</tr>")
        for a, row in zip(row_labels, rows):
            out.append(f'<tr><th style="{row_head_style}">{escape(a)}</th>')
            for b, v in zip(col_labels, row):
                alpha = (v / peak) ** gamma if v else 0.0
                text_color = "white" if alpha > 0.6 else "inherit"
                tip = f"{a} → {b}: {v:.4f}" if normalize else f"{a} → {b}: {v}"
                out.append(
                    f'<td title="{escape(tip)}" style="{cell_style}'
                    f'background:rgba(37,99,235,{alpha:.3f});color:{text_color};">'
                    f'{fmt(v) if show_values else ""}</td>'
                )
            out.append("</tr>")
        out.append("</table></div>")
        caption = "P(next | context)" if normalize else "counts"
        out.append(f'<div style="font-size:11px;opacity:0.7;margin-top:4px;">'
                   f'rows: context · columns: next char · {caption} · hover for values</div>')
        return "".join(out)

    @staticmethod
    def _matrix_text(row_labels, col_labels, rows, peak, gamma) -> str:
        shades = " ░▒▓█"
        w = max((len(r) for r in row_labels), default=1)
        lines = [" " * (w + 1) + "".join(col_labels)]
        for a, row in zip(row_labels, rows):
            cells = "".join(
                shades[max(1, min(len(shades) - 1, round(((v / peak) ** gamma) * (len(shades) - 1))))] if v else "·"
                for v in row
            )
            lines.append(f"{a:>{w}} {cells}")
        return "\n".join(lines)