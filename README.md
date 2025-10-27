# GitScraper

Extract structured rows (Company | Role | Location | URL) from Markdown files hosted in GitHub repositories. Useful for scraping job/role listings from README.md, jobs.md, or similar lists (tables or bullet lists).

## Quick Start

- Python 3.9+ recommended
- Install deps: `pip install -r requirements.txt`
- Optional: set a GitHub token to avoid rate limits
  - PowerShell: `$env:GITHUB_TOKEN = "<your-token>"`
  - macOS/Linux: `export GITHUB_TOKEN="<your-token>"`

## Usage

Run the CLI and point it at a repo, branch, and path.

Basic example (fetch whole file and write to `out.txt`):

```
python gitscraper.py --repo owner/repo --path README.md
```

Target a specific section and filter to Canada, include a header row, and write to a custom file:

```
python gitscraper.py \
  --repo owner/repo \
  --path jobs.md \
  --anchor roles \
  --keywords canada \
  --include-headers \
  --outfile jobs_canada.txt
```

Preview results without writing a file and see detailed logs:

```
python gitscraper.py --repo owner/repo --path listings.md --dry-run --verbose
```

## Arguments

- `--repo` (required): GitHub repository in `owner/repo` format (no protocol).
- `--branch` (optional): Branch name. Default: `main`.
- `--path` (required): Path to the Markdown file in the repo (e.g., `README.md`, `internships.md`).
- `--anchor` (optional): Target a single Markdown section by header. Accepts a slug or header text (e.g., `roles`, `#roles`, `## Roles`). If not provided, the entire file is parsed.
- `--keywords` (optional): Comma‑separated keywords used to match locations/notes (case‑insensitive). Example: `canada, toronto`.
- `--outfile` (optional): Output file path. Default: `out.txt`.
- `--include-headers` (flag): Include `Company | Role | Location | URL` header in the output file.
- `--max-rows` (optional): Cap the number of rows written.
- `--dry-run` (flag): Print sample output to stdout instead of writing a file.
- `--verbose` (flag): Enable verbose logging.

## What It Extracts

The scraper reads Markdown tables or bullet lists and outputs normalized lines in this format:

```
Company | Role | Location | URL
```

If multiple links are present, it prefers application/job‑like URLs.

## Keyword Matching Tips

- Keywords are matched against `Location` and `Notes` fields.
- Passing `canada` auto‑expands to common Canadian cities/abbreviations and "Remote (Canada)" variants.
- Negations like "US only" or "not Canada" are respected and will not match `canada`.

## Rate Limits and Auth

Unauthenticated requests are subject to GitHub rate limits. Set `GITHUB_TOKEN` to increase limits:

- PowerShell: `$env:GITHUB_TOKEN = "<your-token>"`
- macOS/Linux: `export GITHUB_TOKEN="<your-token>"`

## Exit Codes

- `0` on success
- `1` if fetching fails, parsing yields no rows, or filters match nothing
- `2` if required arguments are missing

## Development

- Run tests: `pytest -q`
