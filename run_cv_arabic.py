#!/usr/bin/env python3
"""Diacritize Common Voice Arabic and forced-align it with MFA.

Stages, in order: download, select, diacritize, lexicon, prepare, validate,
train, align. Each stage can be run on its own. Outputs land in work/ar/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from phonetise_ipa import Phonetiser, normalize_transcript

STAGES = [
    "download",
    "select",
    "diacritize",
    "lexicon",
    "prepare",
    "validate",
    "train",
    "align",
]

MIN_HOURS = 2.0
MIN_SECONDS = MIN_HOURS * 3600.0
MDC_DATASET = "cmu5wom3z00ceo107guxtg4xa"
MDC_DOWNLOAD = (
    f"https://mozilladatacollective.com/api/datasets/{MDC_DATASET}/download"
)
MODEL_ID = "rufaelfekadu/diactc-combined"
URL_KEYS = ("downloadUrl", "url", "download_url")

ROOT = Path(__file__).resolve().parent


class Paths:
    def __init__(self) -> None:
        self.root = ROOT
        self.cv_dir = Path(os.environ.get("CV_DIR", ROOT / "data" / "cv-corpus-27.0-2026-09-11" / "ar"))
        self.work_dir = Path(os.environ.get("WORK_DIR", ROOT / "work" / "ar"))
        self.diac_root = Path(os.environ.get("DIACTC_ROOT", ROOT / "third_party" / "DiaCTC"))
        self.ipa_csv = ROOT / "arabic_buckwalter_to_ipa.csv"
        self.phonetiser_dir = ROOT / "third_party" / "Arabic-Phonetiser"

    @property
    def selected(self) -> Path:
        return self.work_dir / "selected.tsv"

    @property
    def diacritized(self) -> Path:
        return self.work_dir / "diacritized.jsonl"

    @property
    def empty_log(self) -> Path:
        return self.work_dir / "diacritize_empty.log"

    @property
    def lexicon(self) -> Path:
        return self.work_dir / "lexicon.txt"

    @property
    def oov(self) -> Path:
        return self.work_dir / "oov_words.txt"

    @property
    def kept(self) -> Path:
        return self.work_dir / "kept.tsv"

    @property
    def corpus(self) -> Path:
        return self.work_dir / "mfa_corpus"

    @property
    def acoustic(self) -> Path:
        return self.work_dir / "acoustic" / "ar_diactc_acoustic.zip"

    @property
    def alignments(self) -> Path:
        return self.work_dir / "alignments"

    @property
    def download_dir(self) -> Path:
        return self.work_dir / "_cv_extract"


def require_file(path: Path, stage: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run the {stage} stage first.")


def read_tsv(path: Path, quote_none: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        kwargs = {"delimiter": "\t"}
        if quote_none:
            kwargs["quoting"] = csv.QUOTE_NONE
        reader = csv.DictReader(handle, **kwargs)
        return list(reader)


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def clips_ready(cv_dir: Path) -> bool:
    clips = cv_dir / "clips"
    if not (cv_dir / "validated.tsv").is_file() or not clips.is_dir():
        return False
    return any(clips.glob("*.mp3"))


def _find_url(obj) -> str | None:
    if isinstance(obj, dict):
        for key in URL_KEYS:
            value = obj.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in obj.values():
            found = _find_url(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_url(item)
            if found:
                return found
    return None


def _archive_name(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    for ext in (".tar.gz", ".tar.bz2", ".tgz", ".zip", ".tar"):
        if path.endswith(ext):
            return "cv_corpus_download" + ext
    return "cv_corpus_download"


def _download(url: str, dest: Path, headers: dict[str, str] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Download failed ({exc.code}): {detail}") from exc
    partial.replace(dest)


def _request_archive_url(api_key: str) -> str:
    request = urllib.request.Request(
        MDC_DOWNLOAD,
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Mozilla Data Collective download request failed ({exc.code}): {detail}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Download response was not JSON: {raw[:200]!r}") from exc
    url = _find_url(payload)
    if not url:
        raise SystemExit(
            "Download response had no downloadUrl, url, or download_url. "
            f"Keys: {list(payload) if isinstance(payload, dict) else type(payload).__name__}"
        )
    return url


def _archive_kind(path: Path) -> str:
    with path.open("rb") as handle:
        magic = handle.read(8)
        handle.seek(257)
        ustar = handle.read(5)
    if magic.startswith(b"PK"):
        return "zip"
    if magic.startswith(b"\x1f\x8b") or ustar == b"ustar":
        return "tar"
    raise SystemExit(f"{path} is not a zip or tar archive (magic {magic[:4]!r}).")


def _safe_members_ok(dest: Path, names: list[str]) -> None:
    root = dest.resolve()
    for name in names:
        target = (root / name).resolve()
        if os.path.commonpath([root, target]) != str(root):
            raise SystemExit(f"Refusing to extract unsafe archive path {name}")


def _extract(archive: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    kind = _archive_kind(archive)
    if kind == "zip":
        with zipfile.ZipFile(archive) as zipped:
            _safe_members_ok(dest, zipped.namelist())
            zipped.extractall(dest)
        return
    with tarfile.open(archive) as tar:
        _safe_members_ok(dest, [member.name for member in tar.getmembers()])
        tar.extractall(dest)


def _find_clips(root: Path) -> Path | None:
    for dirpath, dirnames, _filenames in os.walk(root):
        if "clips" in dirnames:
            return Path(dirpath) / "clips"
        dirnames[:] = [name for name in dirnames if name != "clips"]
    return None


def _install_corpus(extracted: Path, cv_dir: Path) -> None:
    clips_src = _find_clips(extracted)
    if clips_src is None:
        raise SystemExit(f"No clips directory in {extracted}")
    lang_src = clips_src.parent
    cv_dir.mkdir(parents=True, exist_ok=True)
    if not (cv_dir / "validated.tsv").is_file():
        for tsv in lang_src.glob("*.tsv"):
            shutil.copy2(tsv, cv_dir / tsv.name)
    dest_clips = cv_dir / "clips"
    if clips_src.resolve() == dest_clips.resolve():
        return
    if any(dest_clips.glob("*.mp3")) if dest_clips.is_dir() else False:
        return
    if dest_clips.exists():
        shutil.rmtree(dest_clips)
    shutil.move(str(clips_src), str(dest_clips))


def stage_download(paths: Paths) -> None:
    if clips_ready(paths.cv_dir):
        print(f"Clips already present in {paths.cv_dir / 'clips'}; skipping download.")
        return
    api_key = os.environ.get("MDC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set MDC_API_KEY to download Common Voice audio.")
    print("Requesting Common Voice Arabic download URL.")
    url = _request_archive_url(api_key)
    archive = paths.work_dir / _archive_name(url)
    if not archive.is_file():
        print(f"Downloading archive to {archive}")
        _download(url, archive)
    else:
        print(f"Reusing existing archive {archive}")
    print(f"Extracting into {paths.download_dir}")
    _extract(archive, paths.download_dir)
    _install_corpus(paths.download_dir, paths.cv_dir)
    if not clips_ready(paths.cv_dir):
        raise SystemExit(f"Download finished but no mp3 clips were found in {paths.cv_dir / 'clips'}")
    print(f"Common Voice audio is in {paths.cv_dir / 'clips'}")


def stage_select(paths: Paths, max_hours: float | None) -> None:
    validated_path = paths.cv_dir / "validated.tsv"
    duration_path = paths.cv_dir / "clip_durations.tsv"
    require_file(validated_path, "download")
    require_file(duration_path, "download")

    durations: dict[str, float] = {}
    for row in read_tsv(duration_path, quote_none=True):
        durations[row["clip"]] = float(row["duration[ms]"]) / 1000.0

    speakers: dict[str, str] = {}
    next_speaker = 1
    kept: list[dict[str, str]] = []
    n_validated = 0
    for row in read_tsv(validated_path, quote_none=True):
        n_validated += 1
        sentence = (row.get("sentence") or "").strip()
        filename = row.get("path") or ""
        if not sentence or "common_voice" in sentence or not filename:
            continue
        duration = durations.get(filename)
        if duration is None or duration <= 1.0:
            continue
        client_id = row.get("client_id") or ""
        speaker_id = speakers.get(client_id)
        if speaker_id is None:
            speaker_id = str(next_speaker)
            speakers[client_id] = speaker_id
            next_speaker += 1
        kept.append(
            {
                "path": filename,
                "client_id": client_id,
                "speaker_id": speaker_id,
                "duration": f"{duration:.3f}",
                "sentence": sentence,
            }
        )

    total_seconds = sum(float(row["duration"]) for row in kept)
    if total_seconds < MIN_SECONDS:
        raise SystemExit(
            f"Selected audio is {total_seconds / 3600.0:.2f} hours, under the {MIN_HOURS:g} hour minimum."
        )

    if max_hours is not None:
        cap = max_hours * 3600.0
        chosen: list[dict[str, str]] = []
        running = 0.0
        for row in sorted(kept, key=lambda item: item["path"]):
            chosen.append(row)
            running += float(row["duration"])
            if running >= cap:
                break
        kept = chosen
        total_seconds = running

    fieldnames = ["path", "client_id", "speaker_id", "duration", "sentence"]
    write_tsv(paths.selected, kept, fieldnames)
    n_speakers = len({row["speaker_id"] for row in kept})
    print(
        f"Wrote {len(kept)} clips ({total_seconds / 3600.0:.2f} hours, {n_speakers} speakers) "
        f"from {n_validated} validated rows to {paths.selected}"
    )


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = record.get("path")
            if filename:
                done.add(filename)
    return done


def _import_wav2vec(diac_root: Path):
    src = diac_root / "src"
    if not (src / "diactc").is_dir():
        raise SystemExit(f"DiaCTC package not found at {src}. Set DIACTC_ROOT.")
    sys.path.insert(0, str(src))
    try:
        from diactc.models import Wav2Vec2DiacritizationModel
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("k2") or exc.name == "k2":
            raise SystemExit(
                "k2 is required for WFST decoding and is not installed. "
                "Install a build that matches this CUDA and PyTorch version."
            ) from exc
        raise
    return Wav2Vec2DiacritizationModel


def stage_diacritize(paths: Paths, device: str) -> None:
    require_file(paths.selected, "select")
    clips = paths.cv_dir / "clips"

    if not clips.is_dir():
        raise SystemExit(f"Missing audio directory {clips}. Run the download stage.")

    rows = read_tsv(paths.selected)
    done = _load_done(paths.diacritized)
    pending = [row for row in rows if row["path"] not in done]
    print(f"{len(done)} clips already diacritized, {len(pending)} remaining.")
    if not pending:
        return

    try:
        model_cls = _import_wav2vec(paths.diac_root)
    except Exception as exc:
        print(f"Error importing wav2vec2: {exc}")
        raise SystemExit(f"Error importing wav2vec2: {exc}") from exc

    print(f"Loading {MODEL_ID} on {device}.")
    model = model_cls(MODEL_ID, device=device, use_blank_token=True)

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_empty = 0
    with paths.diacritized.open("a", encoding="utf-8") as out, paths.empty_log.open("a", encoding="utf-8") as empty_log:
        for index, row in enumerate(pending, start=1):
            audio = clips / row["path"]
            record = {
                "audio_filepath": str(audio),
                "path": row["path"],
                "text": row["sentence"],
                "speaker_id": row["speaker_id"],
                "client_id": row["client_id"],
                "duration": float(row["duration"]),
                "hypothesis": "",
                "rtf": None,
            }
            if not audio.is_file():
                record["error"] = "missing audio"
                empty_log.write(f"{row['path']}\tmissing audio\n")
                n_empty += 1
            else:
                try:
                    hypothesis, rtf = model.diacritize(
                        row["sentence"],
                        str(audio),
                        constrained=True,
                        method="wfst",
                    )
                    record["hypothesis"] = (hypothesis or "").strip()
                    record["rtf"] = rtf
                except Exception as exc:
                    record["error"] = str(exc)
                    record["hypothesis"] = ""
                if not record["hypothesis"]:
                    reason = record.get("error") or "empty hypothesis"
                    empty_log.write(f"{row['path']}\t{reason}\n")
                    n_empty += 1
                else:
                    n_ok += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            empty_log.flush()
            if index % 100 == 0 or index == len(pending):
                print(f"diacritized {index}/{len(pending)} (ok {n_ok}, empty {n_empty})", flush=True)
    print(f"Diacritization records are in {paths.diacritized}")


def _load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = record.get("path")
            if filename:
                records[filename] = record
    return records


def stage_lexicon(paths: Paths) -> None:
    require_file(paths.selected, "select")
    require_file(paths.diacritized, "diacritize")
    require_file(paths.ipa_csv, "lexicon")

    selected = read_tsv(paths.selected)
    diacritized = _load_jsonl(paths.diacritized)
    missing = [row["path"] for row in selected if row["path"] not in diacritized]
    if missing:
        raise SystemExit(
            f"{len(missing)} selected clips have no diacritization record. Run the diacritize stage."
        )

    utterances = []
    words: set[str] = set()
    n_empty = 0
    for row in selected:
        hypothesis = (diacritized[row["path"]].get("hypothesis") or "").strip()
        if not hypothesis:
            n_empty += 1
            continue
        normalized = normalize_transcript(hypothesis)
        tokens = normalized.split()
        if not tokens:
            n_empty += 1
            continue
        words.update(tokens)
        utterances.append({**row, "diacritized": hypothesis, "normalized": normalized, "tokens": tokens})

    print(f"Phonetising {len(words)} word types from {len(utterances)} utterances.")
    phonetiser = Phonetiser(paths.phonetiser_dir, paths.ipa_csv)
    lexicon: dict[str, str] = {}
    oov: list[str] = []
    for index, word in enumerate(sorted(words), start=1):
        phones = phonetiser.ipa_for_word(word)
        if phones:
            lexicon[word] = " ".join(phones)
        else:
            oov.append(word)
        if index % 1000 == 0 or index == len(words):
            print(f"phonetised {index}/{len(words)}", flush=True)

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    with paths.lexicon.open("w", encoding="utf-8") as handle:
        for word in sorted(lexicon):
            handle.write(f"{word}\t{lexicon[word]}\n")
    paths.oov.write_text("".join(f"{word}\n" for word in oov), encoding="utf-8")

    oov_set = set(oov)
    kept_rows = []
    n_dropped = 0
    for utterance in utterances:
        if any(token in oov_set for token in utterance["tokens"]):
            n_dropped += 1
            continue
        kept_rows.append(utterance)

    write_tsv(
        paths.kept,
        kept_rows,
        ["path", "client_id", "speaker_id", "duration", "sentence", "diacritized", "normalized"],
    )
    kept_seconds = sum(float(row["duration"]) for row in kept_rows)
    print(
        f"Wrote {len(lexicon)} lexicon entries to {paths.lexicon}. "
        f"{len(oov)} OOV types, {n_empty} empty diacritizations, "
        f"{n_dropped} utterances dropped, {len(kept_rows)} kept "
        f"({kept_seconds / 3600.0:.2f} hours)."
    )
    if kept_seconds < MIN_SECONDS:
        raise SystemExit(
            f"Pronounceable audio is {kept_seconds / 3600.0:.2f} hours, under the {MIN_HOURS:g} hour minimum."
        )


def stage_prepare(paths: Paths) -> None:
    require_file(paths.kept, "lexicon")
    require_file(paths.lexicon, "lexicon")
    clips = paths.cv_dir / "clips"
    rows = read_tsv(paths.kept)
    paths.corpus.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    for row in rows:
        audio = clips / row["path"]
        if not audio.is_file():
            raise SystemExit(f"Missing audio file {audio}")
        speaker_dir = paths.corpus / f"spk{int(row['speaker_id']):04d}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        link = speaker_dir / row["path"]
        target = audio.resolve()
        if not link.is_symlink() or link.resolve() != target:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        lab = link.with_suffix(".lab")
        lab.write_text(row["normalized"].strip() + "\n", encoding="utf-8")
        expected.add(link)
        expected.add(lab)

    removed = 0
    for dirpath, _dirnames, filenames in os.walk(paths.corpus, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in {".mp3", ".wav", ".lab", ".textgrid"} and path not in expected:
                path.unlink()
                removed += 1
    for speaker_dir in list(paths.corpus.iterdir()):
        if speaker_dir.is_dir() and not any(speaker_dir.iterdir()):
            speaker_dir.rmdir()
            removed += 1
    print(f"Prepared {len(rows)} utterances in {paths.corpus} (removed {removed} stale files).")


def _require_mfa() -> None:
    if shutil.which("mfa") is None:
        raise SystemExit("mfa was not found on PATH. Activate the aligner conda environment and retry.")


def _run_mfa(args: list[str]) -> None:
    _require_mfa()
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)


def stage_validate(paths: Paths) -> None:
    require_file(paths.lexicon, "lexicon")
    if not paths.corpus.is_dir():
        raise SystemExit(f"Missing {paths.corpus}. Run the prepare stage.")
    _run_mfa(
        [
            "mfa",
            "validate",
            str(paths.corpus),
            str(paths.lexicon),
            "--phone_set",
            "UNKNOWN",
            "--ignore_acoustics",
            "--clean",
        ]
    )


def stage_train(paths: Paths) -> None:
    require_file(paths.lexicon, "lexicon")
    if not paths.corpus.is_dir():
        raise SystemExit(f"Missing {paths.corpus}. Run the prepare stage.")
    paths.acoustic.parent.mkdir(parents=True, exist_ok=True)
    _run_mfa(
        [
            "mfa",
            "train",
            "--clean",
            "--phone_set",
            "UNKNOWN",
            str(paths.corpus),
            str(paths.lexicon),
            str(paths.acoustic),
        ]
    )


def stage_align(paths: Paths) -> None:
    require_file(paths.lexicon, "lexicon")
    require_file(paths.acoustic, "train")
    if not paths.corpus.is_dir():
        raise SystemExit(f"Missing {paths.corpus}. Run the prepare stage.")
    paths.alignments.mkdir(parents=True, exist_ok=True)
    _run_mfa(
        [
            "mfa",
            "align",
            "--clean",
            str(paths.corpus),
            str(paths.lexicon),
            str(paths.acoustic),
            str(paths.alignments),
        ]
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, help="Run only this stage.")
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default="download",
        help="Run this stage and every stage after it. Ignored when --stage is set.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help=f"Cap the select stage at this many hours. Must be at least {MIN_HOURS:g}.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for diacritization.")
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours < MIN_HOURS:
        parser.error(f"--max-hours must be at least {MIN_HOURS:g}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.stage:
        stages = [args.stage]
    else:
        stages = STAGES[STAGES.index(args.from_stage) :]
    if args.max_hours is not None and "select" not in stages:
        print("--max-hours applies only to the select stage; it was ignored.", file=sys.stderr)

    paths = Paths()
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "download": lambda: stage_download(paths),
        "select": lambda: stage_select(paths, args.max_hours),
        "diacritize": lambda: stage_diacritize(paths, args.device),
        "lexicon": lambda: stage_lexicon(paths),
        "prepare": lambda: stage_prepare(paths),
        "validate": lambda: stage_validate(paths),
        "train": lambda: stage_train(paths),
        "align": lambda: stage_align(paths),
    }
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        runners[stage]()


if __name__ == "__main__":
    main()
