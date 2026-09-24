import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator


def plot_ngram_matrix_size(
        vocab_sizes=(5, 26),
        n_grams=(2, 3, 4, 5, 6, 7, 8, 9, 10),
        training_size: int | None = None,
        path: str | None = None,
        style: str = "assets/workshop.mplstyle",
        scale: str = "log"
):
    with plt.style.context(style):
        fig, ax = plt.subplots()

        for v in vocab_sizes:
            cells = [v ** n for n in n_grams]
            line, = ax.plot(n_grams, cells, marker="o", markeredgecolor="white",
                            markeredgewidth=0.6, zorder=3)
            ax.annotate(f"$V = {v}$", xy=(n_grams[-1], cells[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=line.get_color())

        all_cells = [v ** n for v in vocab_sizes for n in n_grams]
        if training_size:
            all_cells.append(training_size)

        if scale == "log":
            y_low, y_high = min(all_cells) / 5, max(all_cells) * 50
        else:
            y_low, y_high = 0, max(all_cells) * 1.05
        if training_size:
            ax.set_yscale(scale)
            if scale == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.axhline(training_size, color="0.3", lw=1, ls=(0, (4, 3)), zorder=2)
            ax.axhspan(training_size, y_high, color="0.5", alpha=0.08, lw=0, zorder=1)
            mid = n_grams[len(n_grams) // 2]
            ax.annotate(f"Training-Tokens ({training_size:,})", xy=(mid, training_size),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                        color="0.3", fontsize=9)
            ax.annotate("Mehr Einträge als Training-Tokens:\nDie Matrix ist sparse.",
                        xy=(0.02, 0.97), xycoords="axes fraction", ha="left", va="top",
                        color="0.45", fontsize=8, style="italic")
        ax.set_ylim(y_low, y_high)

        ax.set_yscale(scale)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(n_grams[0] - 0.3, n_grams[-1] + 0.3)
        ax.set_xlabel("$n$ (N-Gram Kontextgröße)")
        ax.set_ylabel("Num. Einträge in der Assoziationsmatrix ($V^{\\,n}$)")

        fig.tight_layout()
        if path:
            fig.savefig(path)
        plt.show()
    return fig, ax