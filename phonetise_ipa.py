"""Diacritized Arabic word to space-separated IPA, via Arabic-Phonetiser.

The phonetiser converts Arabic script to Buckwalter and returns phonetic
tokens. Those tokens are mapped with arabic_buckwalter_to_ipa.csv. A token
that is not in the table is split by longest-match, and characters that are
still unknown are kept (emphatic I0 / UU0 become ``I 0`` / ``U U 0``).
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import os
import re
import subprocess
import sys
from pathlib import Path

PHONETISER_REPO = "https://github.com/nawarhalabi/Arabic-Phonetiser.git"

# Arabic letters (excluding tatweel) plus harakat. Everything else becomes a space.
_DROP_RE = re.compile(
    r"[^\u0621-\u063A\u0641-\u064A\u0671"
    r"\u064B-\u0652\u0653-\u0655\u0670\s]+"
)


def normalize_transcript(text: str) -> str:
    """Remove tatweel and punctuation. Keep Arabic letters and diacritics."""
    text = text.replace("\u0640", "")
    text = _DROP_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_phonetiser_patch(phonetiser_dir: Path) -> None:
    """Make phonetise-Arabic.py importable on Python 3.

    The published file has a Python 2 print statement, true division where the
    pronunciation index must be an int, and a stress check that compares an
    empty string with an int when a pronunciation is empty.
    """
    path = phonetiser_dir / "phonetise-Arabic.py"
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace("print 'skipped'", "print('skipped')")
    text = text.replace("\tprint pronunciation\n", "\tprint(pronunciation)\n")
    text = text.replace(
        "curIndex = (i / iterations) % len(letter)",
        "curIndex = (i // iterations) % len(letter)",
    )
    text = text.replace(
        "if(stressIndex < len(pronunciation) and stressIndex != -1):",
        "if(isinstance(stressIndex, int) and 0 <= stressIndex < len(pronunciation)):",
    )
    if text != original:
        path.write_text(text, encoding="utf-8")


def ensure_phonetiser(phonetiser_dir: Path) -> None:
    script = phonetiser_dir / "phonetise-Arabic.py"
    if not script.is_file():
        phonetiser_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", PHONETISER_REPO, str(phonetiser_dir)],
            check=True,
        )
    apply_phonetiser_patch(phonetiser_dir)


def load_ipa_table(csv_path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["Buckwalter"]
            ipa = row["IPA"]
            if key and ipa:
                table[key] = ipa
    if not table:
        raise SystemExit(f"No mappings found in {csv_path}")
    return table


def map_phones(tokens: list[str], table: dict[str, str]) -> list[str]:
    """Map phonetiser tokens to IPA phones.

    Longer table keys are tried first so ``ii0`` and ``bb`` stay one phone.
    ``sil`` is dropped. Unknown symbols are emitted one character at a time.
    """
    keys = sorted(table, key=len, reverse=True)
    phones: list[str] = []
    for raw in tokens:
        token = raw.rstrip("'")
        if not token or token == "sil":
            continue
        index = 0
        while index < len(token):
            matched = False
            for key in keys:
                if token.startswith(key, index):
                    mapped = table[key]
                    if mapped != "(sil)":
                        phones.append(mapped)
                    index += len(key)
                    matched = True
                    break
            if not matched:
                phones.append(token[index])
                index += 1
    return phones


class Phonetiser:
    def __init__(self, phonetiser_dir: Path, csv_path: Path) -> None:
        ensure_phonetiser(phonetiser_dir)
        self.phonetiser_dir = phonetiser_dir
        self.table = load_ipa_table(csv_path)
        self._phonetise = self._load_phonetise()

    def _load_phonetise(self):
        sys.path.insert(0, str(self.phonetiser_dir))
        script = self.phonetiser_dir / "phonetise-Arabic.py"
        spec = importlib.util.spec_from_file_location("phonetise_arabic", script)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not load {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.phonetise

    def buckwalter_phones(self, word: str) -> list[str] | None:
        """First pronunciation for one diacritized word, without utterance sil."""
        previous = os.getcwd()
        try:
            # findstress.py appends unknown phones to a relative "errors" file.
            os.chdir(self.phonetiser_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                _bounded, utterances, _dictionary = self._phonetise(word)
        except Exception:
            return None
        finally:
            os.chdir(previous)
        if not utterances or not utterances[0].strip():
            return None
        phones = utterances[0].split()
        while phones and phones[0] == "sil":
            phones.pop(0)
        while phones and phones[-1] == "sil":
            phones.pop()
        return phones or None

    def ipa_for_word(self, word: str) -> list[str] | None:
        phones = self.buckwalter_phones(word)
        if not phones:
            return None
        mapped = map_phones(phones, self.table)
        return mapped or None
