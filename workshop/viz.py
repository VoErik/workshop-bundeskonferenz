import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
import torch
import numpy as np
import ipywidgets as widgets
from IPython.display import display


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



def visualisiere_sampling(modell, standard_prompt: str = "The capital of France is", top_n: int = 15):
    """
    Interaktive Visualisierung von Temperature und Top-p (Nucleus Sampling)
    anhand der echten Logits eines geladenen Sprachmodells.
    """
    cache = {"prompt": None, "logits": None}

    def hole_logits(prompt_text: str):
        if cache["prompt"] != prompt_text:
            inputs = modell.tokenizer(prompt_text, return_tensors="pt").to(modell.model.device)
            with torch.no_grad():
                outputs = modell.model(**inputs)
                # Logits des allerletzten Tokens der Eingabe extrahieren: Form (vocab_size,)
                cache["logits"] = outputs.logits[0, -1, :].detach().float().cpu()
                cache["prompt"] = prompt_text
        return cache["logits"]

    def plot_update(prompt_text: str, temperature: float, top_p: float):
        if not prompt_text.strip():
            return
        
        raw_logits = hole_logits(prompt_text)

        # 1. Temperatur-Skalierung auf dem gesamten Vokabular
        if temperature == 0.0:
            probs = torch.zeros_like(raw_logits)
            probs[torch.argmax(raw_logits)] = 1.0
        else:
            scaled_logits = raw_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

        # 2. Absteigend sortieren über das gesamte Vokabular
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 3. Top-p (Nucleus) Cutoff bestimmen:
        # Ein Token gehört zum Nucleus, wenn die kumulative Summe VOR diesem Token < top_p war
        # (Das erste Token ist immer enthalten)
        in_nucleus = (cumulative_probs - sorted_probs) < top_p
        in_nucleus[0] = True

        # 4. Renormalisierte Wahrscheinlichkeiten berechnen
        filtered_probs = torch.where(in_nucleus, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum()

        # Auf die Top-N Kandidaten für die Darstellung beschränken
        top_probs = sorted_probs[:top_n].numpy()
        top_renorm = filtered_probs[:top_n].numpy()
        top_cum = cumulative_probs[:top_n].numpy()
        top_mask = in_nucleus[:top_n].numpy()
        top_tokens = [
            repr(modell.tokenizer.decode([idx])) for idx in sorted_indices[:top_n]
        ]

        nucleus_total_size = int(in_nucleus.sum().item())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
        x = np.arange(top_n)

        # Links: Temperatur + Top-p Schwelle
        colors_left = ["#2b7bba" if keep else "#d95f02" for keep in top_mask]
        ax1.bar(x, top_probs * 100, color=colors_left, alpha=0.85, label="Token-Wahrscheinlichkeit $P_T(w)$")
        ax1.step(x, top_cum * 100, where="mid", color="#333333", linestyle="--", linewidth=1.8, label="Kumulativ (CDF)")
        ax1.axhline(top_p * 100, color="#d95f02", linestyle=":", linewidth=2, label=f"Top-p Schwelle ({top_p:.0%})")
        
        ax1.set_title(f"1. Nach Temperatur (T={temperature:.2f}) & Top-p Cutoff\nAktiver Nucleus im Gesamtvokabular: {nucleus_total_size} Tokens")
        ax1.set_ylabel("Wahrscheinlichkeit (%)")
        ax1.set_ylim(0, 105)
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_tokens, rotation=45, ha="right", fontfamily="monospace")
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(axis="y", alpha=0.3)

        # Rechts: Finale Sampling-Verteilung nach Renormalisierung
        ax2.bar(x, top_renorm * 100, color="#2ca25f", alpha=0.85)
        ax2.set_title(f"2. Finale Sampling-Verteilung $P'(w)$ (Renormalisiert)\nAus diesen grünen Balken wird das nächste Wort gezogen")
        ax2.set_ylabel("Sampling-Wahrscheinlichkeit (%)")
        ax2.set_ylim(0, 105)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_tokens, rotation=45, ha="right", fontfamily="monospace")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Widgets mit voreingestellten Beispiel-Prompts für den Unterricht
    prompt_dropdown = widgets.Dropdown(
        options=[
            ("Faktisch / Eindeutig ('The capital of France is')", "The capital of France is"),
            ("Mathematik ('The derivative of x^2 is')", "The derivative of x^2 is"),
            ("Offen / Kreativ ('Once upon a time, inside the old')", "Once upon a time, inside the old"),
            ("Deutsch ('Die Hauptstadt von Deutschland heißt')", "Die Hauptstadt von Deutschland heißt"),
        ],
        value=standard_prompt if standard_prompt in ["The capital of France is", "The derivative of x^2 is", "Once upon a time, inside the old", "Die Hauptstadt von Deutschland heißt"] else "The capital of France is",
        description="Beispiel:",
        layout=widgets.Layout(width="600px"),
    )

    prompt_input = widgets.Text(
        value=standard_prompt,
        description="Prompt:",
        layout=widgets.Layout(width="600px"),
    )

    def on_dropdown_change(change):
        prompt_input.value = change["new"]
    prompt_dropdown.observe(on_dropdown_change, names="value")

    temp_slider = widgets.FloatSlider(
        value=1.0, min=0.0, max=2.5, step=0.05,
        description="Temp (T):", continuous_update=False,
        layout=widgets.Layout(width="450px")
    )
    top_p_slider = widgets.FloatSlider(
        value=0.85, min=0.05, max=1.0, step=0.05,
        description="Top-p (p):", continuous_update=False,
        layout=widgets.Layout(width="450px")
    )

    ui = widgets.VBox([
        prompt_dropdown,
        prompt_input,
        widgets.HBox([temp_slider, top_p_slider])
    ])
    out = widgets.interactive_output(
        plot_update,
        {"prompt_text": prompt_input, "temperature": temp_slider, "top_p": top_p_slider}
    )
    display(ui, out)