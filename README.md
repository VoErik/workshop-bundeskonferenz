## Natürliche Sprachverarbeitung mit KI

Dieser Workshop gibt einen mathematischen Überblick über verschiedene Aspekte der Sprachverarbeitung, insbesondere -generierung. Ausgehend von klassischer N-Gram Modellierung lösen wir instruktive Probleme des Ansatzes um abschließend zur Methodik von Transformern, die in modernen Sprachmodellen stecken, zu gelangen.

### Anforderungen

TN benötigen ihr eigenes Laptop/Tablet (bevorzugt mit Tastatur) und einen Google Account, da Codebeispiele auf Google Colab ausgeführt werden sollen.

### Lernziele

Die Zielgruppe sind primär Lehrkräfte der späten Mittelstufe bis Oberstufe für Mathematik, Informatik, oder Physik.

- Die TN verstehen die grundlegende Idee der wortbasierten natürlichen Sprachverarbeitung.
- Die TN können die Entwicklung sprachbasierter Technologien von einfachen N-Grams bis hin zu modernen Transformern technisch nachvollziehen.
- Die TN entwickeln ein grundlegendes Verständnis für die mathematischen Grundlagen hinter Sprachmodellen.
- Die TN können Bezug zwischen Themen der natürlichen Sprachverarbeitung und den Lehrplaninhalten herstellen.

Notizen:

- Datensatz (TinyShakespeare)

### Ngram

Einfaches Beispiel Berechnung per Hand (ggf so aufbereiten dass es in Unterrichtseinheit übernommen werden kann)

- co-occurence matrix (geht auf Wort/Tokenebene)
- Sind N-gramme skalierbar
Token vs Wort
Vokabular als "Fachbegriff" einführen
Anwendungsbeispiel: Autocomplete / Autocorrect
Bezug zu LP: Wahrscheinlichkeiten / Stochastik

### RNN / Embeddings

NNs brauchen Vektoren :(
Darum Wort zu Zahl :) -> Wie?

### Transformer / Attention

- HuggingFace (WS Johannes / Franzi)
- Attention Mechanismus erklären
- Autoregression
- wie erinnert sich ein LM an etwas?

