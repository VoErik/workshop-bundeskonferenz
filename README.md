# Natürliche Sprachverarbeitung mit KI

Der Workshop gibt einen mathematischen Überblick über Sprachmodelle: von klassischen N-Gram-Modellen bis zu den Transformern, die in modernen KI-Sprachmodellen stecken. Er richtet sich an Lehrkräfte der Mittel- und Oberstufe für Mathematik, Informatik und Physik. Durchführung auf der [Schule MIT Wissenschaft Bundeskonferenz 2026](https://schule-mit-wissenschaft.de/veranstaltungen/bundeskonferenz-2026-karlsruhe-hybrid/erik-voigt-johannes-langer/).
 
## Materialien
 
| Notebook | |
|---|---|
| N-Gram Sprachmodelle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoErik/workshop-bundeskonferenz/blob/main/n_gram.ipynb) |
| Large Language Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoErik/workshop-bundeskonferenz/blob/main/llm.ipynb) |
 
Folien, Handout und weitere Materialien: [Google Drive](LINK_ZUM_DRIVE_ORDNER)

## Setup
 
**Google Colab (empfohlen):** Notebook über den Button öffnen, mit *Datei → Kopie in Drive speichern* eine eigene Kopie anlegen und die erste Code-Zelle ausführen. Sie installiert alles Nötige. Sie benötigen nur einen Google-Account.
 
**Lokal** mit [uv](https://docs.astral.sh/uv/):
 
```bash
git clone https://github.com/VoErik/workshop-bundeskonferenz.git
cd workshop-bundeskonferenz
uv sync
uv run --with jupyterlab jupyter lab
```

## Zitieren
 
Alle Materialien dürfen frei verwendet und verändert werden. Wenn Sie sie nutzen, freuen wir uns über eine Nennung:
 
> Voigt, E. & Langer, J. (2026). *Natürliche Sprachverarbeitung mit KI*. Workshop, SMW Bundeskonferenz 2026, Universität Bamberg.
 
```bibtex
@misc{voigt2026sprachverarbeitung,
  author       = {Voigt, Erik and Langer, Johannes},
  title        = {Natürliche Sprachverarbeitung mit KI},
  howpublished = {Workshop, SMW Bundeskonferenz 2026, Universität Bamberg},
  year         = {2026},
  url          = {https://github.com/VoErik/workshop-bundeskonferenz.git}
}
```

