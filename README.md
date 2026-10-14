# Common Voice Arabic forced alignment

This pipeline takes the validated Arabic clips from Common Voice, restores
their diacritics from the audio with [DiaCTC](https://github.com/rufaelfekadu/DiaCTC),
turns each diacritized word into IPA phones, and then trains and runs
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (MFA)
on the result. The output is a phone-level TextGrid for every clip that could
be phonetised.

## layout

| File | Purpose |
|------|---------|
| `run.sh` | Entry point. Sets the default paths and runs `run_cv_arabic.py`. |
| `run_cv_arabic.py` | The pipeline. Each stage is a function, and you can run them separately. |
| `phonetise_ipa.py` | Wraps Arabic-Phonetiser and maps its Buckwalter output to IPA. |
| `arabic_buckwalter_to_ipa.csv` | The table that maps Buckwalter phones to IPA phones. |


## Requirements

This repository does not install any of the following:

1. **Python environment for diacritization.** Needs PyTorch, torchaudio,
   transformers, librosa and [k2](https://k2-fsa.github.io/k2/installation/index.html).
   k2 must be built for the same CUDA and PyTorch versions you have installed.
2. **DiaCTC source.** Clone it into `third_party/DiaCTC`, or point
   `DIACTC_ROOT` at an existing clone. The script adds `$DIACTC_ROOT/src` to
   `sys.path`, so you don't need to `pip install` the package itself, only its
   requirements.
3. **MFA**, available as `mfa` on `PATH`, for example from a conda environment
   called `aligner`.
4. **A Mozilla Data Collective API key** in `MDC_API_KEY`. The download stage
   uses it to fetch Common Voice. You don't need it if the corpus is already
   on disk.
5. **git**. The lexicon stage clones
   [Arabic-Phonetiser](https://github.com/nawarhalabi/Arabic-Phonetiser) into
   `third_party/Arabic-Phonetiser` the first time it runs, then patches it so
   it imports under Python 3.

## Installation

Run all of these from the repository root.

### 1. Clone DiaCTC

```bash
mkdir -p third_party
git clone https://github.com/rufaelfekadu/DiaCTC.git third_party/DiaCTC
```

If you already have a clone somewhere else, skip this step and set
`DIACTC_ROOT` to its path instead.

### 2. Create the diacritization environment

DiaCTC needs Python 3.12 or newer.

```bash
conda create -n diac-wfst python=3.12 -y
conda activate diac-wfst
pip install -r third_party/DiaCTC/requirements.txt
```

This installs PyTorch 2.9.1 and torchaudio 2.9.1. On Linux the default wheels
are built for CUDA 12.8. It also installs transformers, librosa, soundfile,
pyarabic and the rest of DiaCTC's dependencies.

### 3. Install k2

k2 does the WFST decoding, and its wheel has to match both your PyTorch
version and the CUDA version it was built with. Check those versions first:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 2.9.1+cu128 12.8
```

Then install the matching wheel from the
[k2 CUDA wheel index](https://k2-fsa.github.io/k2/cuda.html). For CUDA 12.8 and
torch 2.9.1:

```bash
pip install k2==1.24.4.dev20251118+cuda12.8.torch2.9.1 -f https://k2-fsa.github.io/k2/cuda.html
```

For other versions, see the
[k2 installation guide](https://k2-fsa.github.io/k2/installation/index.html).

### 4. Check the diacritization environment

```bash
PYTHONPATH=third_party/DiaCTC/src python -c \
  "import k2; from diactc.models import Wav2Vec2DiacritizationModel; print('ok')"
```

The diacritize stage downloads the `rufaelfekadu/diactc-combined` model from
the Hugging Face Hub the first time it runs. The model is cached under
`~/.cache/huggingface` after that.

### 5. Create the MFA environment

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda activate aligner
mfa version
```

### 6. Arabic-Phonetiser (optional)

The lexicon stage clones and patches Arabic-Phonetiser on its own. To fetch it
ahead of time, for example on a machine that will run offline later, use:

```bash
git clone --depth 1 https://github.com/nawarhalabi/Arabic-Phonetiser.git third_party/Arabic-Phonetiser
```

The Python 3 patch is still applied the first time the lexicon stage runs.

## Running the pipeline

```bash
export MDC_API_KEY=...       # only needed for the download stage
./run.sh                     # run every stage, from download to align
```

`run.sh` passes all of its arguments through to `run_cv_arabic.py`:

| Flag | Effect |
|------|--------|
| `--stage NAME` | Run only that stage. |
| `--from-stage NAME` | Run that stage and all the stages after it (default `download`). |
| `--max-hours H` | Cap the `select` stage at `H` hours of audio. `H` must be at least 2. |
| `--device DEV` | Torch device for diacritization (default `cuda`). |

The diacritize stage and the MFA stages usually run in different
environments, so in practice you split the run:

```bash
conda activate diac-wfst
./run.sh --stage download
./run.sh --stage select --max-hours 10
./run.sh --stage diacritize
./run.sh --stage lexicon
./run.sh --stage prepare

conda activate aligner
./run.sh --from-stage validate   # validate, train, align
```

Every stage after `diacritize` uses only the Python standard library, so you
can run them with the Python from either environment.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CV_DIR` | `data/cv-corpus-27.0-2026-09-11/ar` | Common Voice language directory, containing `validated.tsv`, `clip_durations.tsv` and `clips/`. |
| `WORK_DIR` | `work/ar` | Where all the outputs are written. |
| `DIACTC_ROOT` | `third_party/DiaCTC` | DiaCTC checkout. |
| `PYTHON` | `python3` | Interpreter that `run.sh` uses. |
| `MDC_API_KEY` | (none) | Mozilla Data Collective token. |

## Stages

| Stage | What it does | Writes to `WORK_DIR` |
|-------|--------------|----------------------|
| `download` | Asks Mozilla Data Collective for an archive URL, downloads it and extracts the TSVs and `clips/` into `CV_DIR`. Skipped if the clips are already there. | `cv_corpus_download.*`, `_cv_extract/` |
| `select` | Reads `validated.tsv` and drops clips of 1 s or less and clips with an empty or placeholder sentence. Gives each `client_id` a numeric speaker ID. With `--max-hours`, keeps clips in sorted path order until the cap is reached. Fails if less than 2 hours of audio is left. | `selected.tsv` |
| `diacritize` | Runs the DiaCTC model `rufaelfekadu/diactc-combined` with WFST decoding constrained to the transcript. The stage can resume: clips that already have a record are skipped. | `diacritized.jsonl`, `diacritize_empty.log` |
| `lexicon` | Normalizes each hypothesis (strips punctuation and tatweel, keeps letters and harakat), phonetises every word type and maps it to IPA. Any utterance with an out-of-vocabulary word is dropped. Fails if less than 2 hours of audio is left. | `lexicon.txt`, `oov_words.txt`, `kept.tsv` |
| `prepare` | Builds the MFA corpus: one directory per speaker (`spkNNNN/`), the mp3 symlinked in, and a `.lab` file next to it with the normalized transcript. Files left over from earlier runs are removed. | `mfa_corpus/` |
| `validate` | `mfa validate --ignore_acoustics` on the corpus and lexicon. | MFA logs |
| `train` | `mfa train` creates an acoustic model from the corpus. | `acoustic/ar_diactc_acoustic.zip` |
| `align` | `mfa align` with the trained model. | `alignments/` (TextGrids) |

Each stage checks that the files it needs exist. If one is missing, it tells
you which stage to run first.

## Lexicon details

`phonetise_ipa.py` runs Arabic-Phonetiser on each diacritized word and keeps
the first pronunciation, without the leading and trailing `sil`. It then maps
the Buckwalter phones to IPA with `arabic_buckwalter_to_ipa.csv`, matching the
longest key first so that geminates and long vowels such as `bb` or `ii0` stay
a single phone. A symbol that isn't in the table is passed through one
character at a time. To change the phone inventory, edit the CSV and rerun
from `lexicon`:

```bash
./run.sh --from-stage lexicon
```

Lexicon lines have the form `word<TAB>phone phone ...`. MFA is run with
`--phone_set UNKNOWN`.
