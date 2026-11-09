import builtins
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

