import io
import os
from contextlib import redirect_stdout
from logging import getLogger
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown as RichMarkdown

getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel("ERROR")

# Gewichte nacheinander statt mit mehreren Threads laden. Sonst stürzt der Kernel auf
# Apple-GPUs (MPS) beim Laden von Qwen3.5 ab (Segfault in PyTorch, wenn mehrere Threads
# gleichzeitig float32 -> bfloat16 konvertieren).
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
import circuitsvis as cv
import torch
from IPython.display import Markdown, display
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    TextStreamer,
)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()  # versteckt transformers-Warnungen, Errors bleiben
# sichtbar. Relevant für Qwen3.5

def _convert_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        try:
            prompt = str(prompt, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as e:
            print("Invalide Zeichen - Prompt kann nicht als String interpretiert werden.")
            return None
    if not prompt.strip():  # auf leere Nachrichten ist kein Modell trainiert
        print("Bitte gib einen Prompt ein.")
        return None
    return prompt

class Gedaechtnis(list):
    """
    Wrapper-Klasse für list[dict[str, str]] Datenstruktur, die mit einer einfachen
    Methode neue Einträge vornehmen kann.
    """

    def neuer_eintrag(self, rolle: str, inhalt: str) -> None:
        """
        Fügt einen neuen Eintrag im Gedächtnis hinzu. 
        Platziert System-Prompt an Stelle 0 und löscht alten System Prompt.
        """
        if rolle == "system":
            self[:] = [d for d in self if d.get("role") != "system"]
            self.insert(0, {"role": rolle, "content": inhalt})
        else:
            self.append({"role": rolle, "content": inhalt})

    def loesche(self) -> None:
        self.clear()

    def zeige(self) -> None:
        console = Console()
        stile = {
            "system": ("bold yellow", "System Prompt"),
            "user": ("bold cyan", "User"),
            "assistant": ("bold green", "Assistant"),
        }

        for eintrag in self:
            rolle = eintrag.get("role", "unknown")
            inhalt = eintrag.get("content", "")
            farbe, titel = stile.get(rolle, ("white", rolle.capitalize()))

            console.print(
                Panel(
                    RichMarkdown(inhalt),
                    title=f"[{farbe}]{titel}[/]",
                    title_align="left",
                    border_style=farbe.split()[-1],
                    expand=False,
                )
            )


def gedaechtnis_mit_system_prompt(prompt: str) -> Gedaechtnis:
    gedaechtnis = Gedaechtnis()
    gedaechtnis.neuer_eintrag("system", prompt)
    return gedaechtnis


class MarkdownStreamer(TextStreamer):
    """
    Zeigt die Antwort während der Generierung als Markdown im Notebook an.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.text = ""
        self.anzeige = display(Markdown(""), display_id=True)

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        self.text += text
        self.anzeige.update(Markdown(self.text))  # pyright: ignore[reportOptionalMemberAccess]


class Sprachmodell:
    """
    Wrapper-Klasse für Huggingface Sprachmodelle. Implementiert nötige Methoden für den
    Workshop.
    """

    use_llammlein_specific_settings: bool

    def __init__(
        self,
        model_name: str = "LSX-UniWue/LLaMmlein_1B_chat_selected",
        *,
        thinking: bool = False,
        device: str = "auto",
    ):
        """ """
        self.use_llammlein_specific_settings = model_name in (
            "LSX-UniWue/LLaMmlein_1B_chat_selected",
            "LSX-UniWue/LLaMmlein_1B",
        )

        if model_name == "LSX-UniWue/LLaMmlein_1B_chat_selected":
            conf = PeftConfig.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(
                conf.base_model_name_or_path,  # pyright: ignore[reportArgumentType]
                torch_dtype="auto",
                device_map=device,
            )

            _model.resize_token_embeddings(32064, mean_resizing=False)

            self.model = PeftModel.from_pretrained(_model, model_name)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # transformers gibt beim Import mancher Modelle Docstring-Checks per print() aus
            with redirect_stdout(io.StringIO()):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map=device
                )

            if model_name == "LSX-UniWue/LLaMmlein_1B":
                # Eigener Tokenizer des Basismodells: EOS ist </s>. Der Chat-Tokenizer
                # hätte <|im_end|> als EOS, dessen ID im Basismodell aber "$" ist.
                # Das Basismodell kennt kein Chat-Format und setzt nur Text fort, daher
                # werden die Nachrichten einfach aneinandergehängt.
                self.tokenizer.chat_template = (
                    "{{ bos_token }}"
                    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
                )
            elif not getattr(self.tokenizer, "chat_template", None):
                # Fallback für Basismodelle ohne Chat-Template:
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['content'] }}"
                    "{% endfor %}"
                )

        self.gedaechtnis: Optional[Gedaechtnis] = None

    def prompt(
        self,
        prompt: str,
        *,
        gedaechtnis: Optional[Gedaechtnis] = None,
        max_tokens: int = 512,
        stream: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> Optional[str]:
        """ """
        prompt = _convert_prompt(prompt)
        if not prompt: return None

        gedaechtnis = gedaechtnis if isinstance(gedaechtnis, Gedaechtnis) else Gedaechtnis()

        gedaechtnis.neuer_eintrag("user", prompt)

        return self._generiere(gedaechtnis, max_tokens, stream, temperature=temperature, top_p=top_p)

    def prompt_auto(
        self,
        prompt: str,
        *,
        gedaechtnis: Optional[Gedaechtnis] = None,
        max_tokens: int = 512,
        stream: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> Optional[str]:
        """ """
        prompt = _convert_prompt(prompt)
        if not prompt: return None

        self.gedaechtnis = (
            gedaechtnis
            if isinstance(gedaechtnis, Gedaechtnis)
            else Gedaechtnis()
            if self.gedaechtnis is None
            else self.gedaechtnis
        )

        self.gedaechtnis.neuer_eintrag("user", prompt)

        return self._generiere(self.gedaechtnis, max_tokens, stream, temperature=temperature, top_p=top_p)

    def _generiere(
        self, 
        gedaechtnis: Gedaechtnis, 
        max_tokens: int, 
        stream: bool,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> Optional[str]:
        """
        Generiert die Antwort auf den Gesprächsverlauf und trägt sie ins Gedächtnis ein.
        Mit `stream=True` wird die Antwort live angezeigt statt zurückgegeben.
        """
        # Direkt tokenisieren: ein separater tokenizer()-Aufruf würde ein zusätzliches
        # <s> voranstellen, mit dem LLäMmlein schlechter zum Ende der Antwort findet
        model_inputs = self.tokenizer.apply_chat_template(
            gedaechtnis,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        do_sample = temperature > 0.0
        output_ids = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
            **model_inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            # verhindert Endlosschleifen wie "Die Hauptstadt ist Berlin.\n\nDie Hauptstadt..."
            repetition_penalty=1.1 if self.use_llammlein_specific_settings else None,
            streamer=MarkdownStreamer(self.tokenizer) if stream else None,
        )[0][len(model_inputs["input_ids"][0]) :].tolist()

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        gedaechtnis.neuer_eintrag("assistant", response)

        return None if stream else response


    def zeige_prompt(self, prompt_or_conversation: str | list[dict[str,str]]) -> str:
        if isinstance(prompt_or_conversation, str):
            conversation = [{"role": "user", "content": prompt_or_conversation}]
        else:
            conversation = prompt_or_conversation

        return str(
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def zeige_attention(
        self,
        eingabe: str | list[dict[str, str]] | None = None,
        *,
        schicht: int | list[int] | str | None = None,
        chat_format: bool = False,
    ) -> None:
        """
        Zeigt die Attention-Muster aller Heads einer (oder mehrerer) Schichten mit
        CircuitsVis an, im Stil der Transformer-Circuits-Artikel von Anthropic.
 
        :param: `eingabe`: ein Text, ein Gesprächsverlauf (z.B. ein `Gedaechtnis`) oder
          `None` für den Verlauf aus `prompt_auto`.
        :param: `schicht`: Index der Schicht (negativ zählt von hinten, -1 = letzte), eine
          Liste von Indizes oder "alle". Standard: die erste Schicht mit Attention
          (bei Qwen3.5 haben nur manche Schichten echte Attention).
        :param: `chat_format`: Text wie bei `prompt` ins Chat-Template einbetten, d.h. genau
          das anzeigen, was das Modell sieht, bevor es antwortet.
 
        Bedienung: Über die kleinen Kacheln fahren bzw. klicken, um einen Head groß
        anzuzeigen. Jede Zeile ist ein Token, das "hinschaut" (Query), jede Spalte ein
        Token, auf das geschaut wird (Key). Fährt man über ein Token unten im Text,
        wird eingefärbt, worauf es schaut.
        """
        if eingabe is None:
            if not self.gedaechtnis:
                print("Kein Gesprächsverlauf vorhanden - bitte einen Text übergeben.")
                return None
            eingabe = self.gedaechtnis
 
        # 1. Eingabe tokenisieren (genauso wie bei der Generierung)
        if isinstance(eingabe, str):
            if not eingabe.strip():
                print("Bitte gib einen Text ein.")
                return None
            if chat_format:
                model_inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": eingabe}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                model_inputs = self.tokenizer(eingabe, return_tensors="pt")
        else:
            # Gesprächsverlauf: ohne Generation-Prompt, damit die letzte Antwort des
            # Modells mit drin ist ("Worauf hat es beim Antworten geschaut?")
            model_inputs = self.tokenizer.apply_chat_template(
                list(eingabe),
                add_generation_prompt=False,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        model_inputs = model_inputs.to(self.model.device)
        input_ids = model_inputs["input_ids"][0].tolist()
 
        if len(input_ids) > 300:
            print(
                f"Hinweis: {len(input_ids)} Tokens - die Darstellung kann langsam werden."
            )
 
        # Forward-Pass mit Attention-Ausgabe. Die schnellen Attention-Kernel (sdpa,
        # flash) berechnen die Attention-Matrix nie explizit, daher kurz auf die
        # "eager"-Implementierung umschalten und danach zurück.
        basis = (
            self.model.get_base_model()
            if isinstance(self.model, PeftModel)
            else self.model
        )
        alte_impl = basis.config._attn_implementation
        if alte_impl != "eager":
            basis.set_attn_implementation("eager")
        try:
            with torch.no_grad():
                ausgabe = self.model(**model_inputs, output_attentions=True)
        finally:
            if alte_impl != "eager":
                basis.set_attn_implementation(alte_impl)
 
        attentions, schicht_ids = self._attention_schichten(ausgabe.attentions)
        if not attentions:
            print("Dieses Modell liefert keine Attention-Muster.")
            return None
 
        # auswahl der anzuzeigenden Schichten
        if schicht is None:
            auswahl = [0]
        elif schicht == "alle":
            auswahl = list(range(len(attentions)))
        else:
            gewuenscht = [schicht] if isinstance(schicht, int) else list(schicht)
            n_gesamt = self._anzahl_schichten(basis)
            auswahl = []
            for s in gewuenscht:
                s_abs = s + n_gesamt if s < 0 else s
                if s_abs in schicht_ids:
                    auswahl.append(schicht_ids.index(s_abs))
                else:
                    print(
                        f"Schicht {s} hat keine Attention-Muster. Verfügbar: "
                        f"{schicht_ids[0]} bis {schicht_ids[-1]}"
                        + (
                            f" (nur {schicht_ids})"
                            if len(schicht_ids) != n_gesamt
                            else ""
                        )
                    )
 
        # zeige
        tokens = self._token_texte(input_ids)
        for i in auswahl:
            l = schicht_ids[i]
            muster = attentions[i][0].float().cpu().numpy().round(4)  # [heads, T, T]
            display(Markdown(f"**Schicht {l}** ({muster.shape[0]} Heads)"))
            display(
                cv.attention.attention_heads(
                    attention=muster,
                    tokens=tokens,
                    attention_head_names=[f"L{l}H{h}" for h in range(muster.shape[0])],
                )
            )
        return None
 
    def _anzahl_schichten(self, basis) -> int:
        config = basis.config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        return config.num_hidden_layers
 
    def _attention_schichten(self, attentions) -> tuple[list, list[int]]:
        """
        Gibt die Attention-Tensoren und die zugehörigen Schicht-Indizes zurück.
        Hybride Modelle wie Qwen3.5 haben neben normalen Attention-Schichten auch
        "lineare" Schichten (Gated DeltaNet), die keine Attention-Matrix haben.
        Diese fehlen in der Ausgabe oder sind None.
        """
        attentions = list(attentions or [])
        basis = (
            self.model.get_base_model()
            if isinstance(self.model, PeftModel)
            else self.model
        )
        n_gesamt = self._anzahl_schichten(basis)
 
        if len(attentions) == n_gesamt:
            paare = [(a, i) for i, a in enumerate(attentions) if a is not None]
        else:
            config = basis.config
            if hasattr(config, "get_text_config"):
                config = config.get_text_config()
            typen = getattr(config, "layer_types", None) or []
            mit_attention = [i for i, t in enumerate(typen) if "linear" not in t]
            vorhanden = [a for a in attentions if a is not None]
            if len(mit_attention) != len(vorhanden):
                mit_attention = list(range(len(vorhanden)))
            paare = list(zip(vorhanden, mit_attention))
 
        return [a for a, _ in paare], [i for _, i in paare]
 
    def _token_texte(self, input_ids: list[int]) -> list[str]:
        """
        Wandelt Token-IDs in lesbare Texte um. Schrittweises Dekodieren erhält
        Leerzeichen und Umlaute, die bei einzeln dekodierten Tokens verloren gehen.
        """
        texte, bisher = [], ""
        for i in range(len(input_ids)):
            neu = self.tokenizer.decode(input_ids[: i + 1])
            # Neuer Text = alles nach dem gemeinsamen Anfang. Ist ein Umlaut auf zwei
            # Byte-Tokens verteilt, zeigt das erste "\ufffd" und das zweite den Umlaut.
            stueck = neu[len(os.path.commonprefix([bisher, neu])) :]
            bisher = neu
            stueck = stueck.replace("\n", "\\n").replace("\t", "\\t")
            texte.append(stueck if stueck else "\ufffd")
        return texte

