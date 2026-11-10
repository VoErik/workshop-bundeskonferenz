"""
Ersetzt die farbigen HTML-Boxen in Notebooks durch Formatierungen, die auch
Google Colab darstellt (Colab entfernt style-Attribute und kennt keine
Bootstrap-Klassen):

- Aufgaben und Fragen (blau)      -> Zitatblock mit Emoji (📝, ❓, 🧪)
- Antworten / Lösungen (orange)   -> <details> mit ✅
- Tipps (grau, "Tipp")            -> <details> mit 💡
- Zusatzinfos (grau)              -> <details> mit ℹ️
- Lehrplanbezug (grün)            -> <details> mit 🎓

Aufruf: python boxen_colab.py llm.ipynb n_gram.ipynb
"""
import json
import re
import sys
from pathlib import Path

BLAU = re.compile(
    r'<div style="background-color:#e3f2fd[^"]*">[ \t]*\n(.*?)\n[ \t]*</div>', re.DOTALL
)
DETAILS_BOX = re.compile(
    r'<div (?:style="background-color:#(?P<farbe>\w+)[^"]*"|class="alert alert-(?P<klasse>\w+)")>'
    r"\s*<details>\s*<summary>\s*(?P<titel>.*?)\s*</summary>(?P<inhalt>.*?)</details>\s*</div>",
    re.DOTALL,
)
EMOJI_ANFANG = {"**Aufgabe:**": "📝", "**Frage:**": "❓", "**Zum Ausprobieren:**": "🧪"}


def zitatblock(m: re.Match) -> str:
    inhalt = m.group(1).strip("\n")
    for marker, emoji in EMOJI_ANFANG.items():
        if inhalt.startswith(marker):
            inhalt = f"{emoji} {inhalt}"
            break
    return "\n".join(f"> {z}" if z.strip() else ">" for z in inhalt.split("\n"))


def details(m: re.Match) -> str:
    farbe, klasse, titel = m["farbe"], m["klasse"], m["titel"]
    if farbe == "fff3e0" or klasse == "warning":
        emoji = "✅"
    elif klasse == "success":
        emoji = "🎓"
    elif titel.lower().startswith("tipp"):
        emoji = "💡"
    else:
        emoji = "ℹ️"
    return f"<details>\n<summary>{emoji} <b>{titel}</b></summary>{m['inhalt']}</details>"


for pfad in map(Path, sys.argv[1:]):
    nb = json.loads(pfad.read_text(encoding="utf-8"))
    for zelle in nb["cells"]:
        if zelle["cell_type"] != "markdown":
            continue
        text = "".join(zelle["source"])
        text = DETAILS_BOX.sub(details, text)
        text = BLAU.sub(zitatblock, text)
        zelle["source"] = text.splitlines(keepends=True)
        if "<div" in text:
            print(f"{pfad}: nicht umgewandelte Box übrig:\n{text[:200]}\n")
    pfad.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{pfad}: fertig")
