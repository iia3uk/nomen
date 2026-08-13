# Nomen

**Production-grade brand discovery engine** for software products (Python 3.12+).

Not a fancy-word generator. A hunt pipeline: tens of thousands of candidates, offline filters, aesthetic and brand-fit ranking, collision checks against registries and domains, then a short list of *independent* names with a full audit trail.

Built for naming an AI-native CMS / developer platform / MCP platform / cloud product (originally for [Jasefly](https://github.com/iia3uk) CMS and neighbouring products).

**Diversity is a hard constraint.** The engine uses novelty search, not hill-climbing. Collapsing into one phonetic family (`Plerasta / Plerora / Plerenda…`) is treated as a bug and triggers an automatic exploration restart.

## Why

Manual lists and ChatGPT dumps fail in the same two ways: *pretty but taken*, or *free but synthetic sludge*. Generators also drift into one phonetic nest. Nomen makes diversity a hard constraint, keeps BeautyScore independent of registry signals, and scales the hunt with cache, backoff, and parallel validation — instead of spamming APIs and returning clones.

Clearing package / search / trademark checks is **not** a legal guarantee. Confirm Nice-class trademarks and domains with counsel before launch.

## Pipeline

```
Generate (quota-balanced engines)
  → Offline filters
  → Similarity
  → Beauty + Brand + Collision
  → NoveltyScore + clustering (1 per root family)
  → Generator / length / CV-pattern quotas
  → Online validation
  → Convergence detector → reseed if collapsed
  → Repeat until ≥N independent clean names
```

## Engine

- **Novelty search** — not hill-climbing to one peak; same root-family = bug → exploration restart
- **10 generators** — char LM, phoneme model, evolutionary / genetic, syllable recombination, letter graph, entropy sampler, pronounceability optimizer, winner mutation, and more; quotas by generator, length, and CV pattern
- **Offline filter bank** — phonotactics, banned morphemes, English cores, synthetic fragments, weak tails, verbatim corpus ban
- **Similarity** — Levenshtein / Damerau / Jaro-Winkler / n-gram / Metaphone + optional embeddings
- **Scoring** — BeautyScore (aesthetics, no registries), BrandScore (pronunciation, memory, typing, visual, premium feel), CollisionScore, NoveltyScore → Overall
- **Diversity selector** — one name per phonetic root; caps on generator/pattern share; min beauty / min novelty
- **Online ValidationGateway** — GitHub, npm, PyPI, crates, RubyGems, NuGet, Packagist, Docker Hub, WordPress, GitLab, RDAP domains (`.com` / `.io` / `.dev` / `.ai` …), web search (Brave / SerpAPI / Bing), OpenCorporates; 72h cache, retry/backoff, concurrency
- **Elite archive + tournament** — best-of-best shortlist with pairwise comparison
- **Ops** — checkpoint / resume, parallel CPU workers, HTTP semaphore, export of `results.json` / csv, `clean_names`, top10/20/100, `rejected.json` with reasons
- **Benchmark** — offline diversity run, no network (`python -m nomen benchmark --n 5000`)

Training learns **statistical patterns only** from SaaS, AI, cloud, developer tools, frameworks, databases, and design brands. Verbatim corpus names are never emitted.

## Stack

Python 3.12 · asyncio · httpx · Typer · Rich · pydantic · RapidFuzz · Metaphone · diskcache · NumPy · optional sentence-transformers

## Install

Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Unix:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[dev]"          # pytest
pip install -e ".[embeddings]"   # sentence-transformers (heavy)
```

## Configure

```powershell
copy .env.example .env
```

Then edit `.env` (local only, gitignored) and `config.yaml`.

Own / already-taken brands go in `nomen/data/reserved.txt`. They are never emitted as discoveries; near-matches are treated as collisions.

```
GITHUB_TOKEN=
BRAVE_SEARCH_API_KEY=
SERPAPI_KEY=
BING_SEARCH_API_KEY=
OPENCORPORATES_API_TOKEN=
WHOIS_API_KEY=
```

Keys are optional. Without them, online GitHub / search / company checks are skipped or rate-limited. Offline generation and `benchmark` still work.

## Run

Windows:

```bat
run.bat
run.bat --fresh
```

```powershell
.\run.ps1
python -m nomen --strict --target-clean 20 --min-score 92
```

Unix:

```bash
./run.sh
python -m nomen --strict --target-clean 20 --min-score 92
```

Resume is the default in `config.yaml` (`--resume`). Start from scratch with `--fresh`.

Benchmark (offline):

```powershell
python -m nomen benchmark --n 5000
```

## Output (`nomen_results/`)

Generated locally and gitignored.

- `results.json` / `results.csv` — full audit trail
- `clean_names.txt`, `top100.txt`, `top20.txt`, `top10.txt`
- `rejected.json` — every rejection reason
- `cache/` — diskcache
- `checkpoint.json` — resume state

## Tests

```powershell
pip install -e ".[dev]"
pytest -q
```

## Author

Роман · [iia3uk](https://github.com/iia3uk)
