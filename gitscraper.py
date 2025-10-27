"""GitScraper CLI tool for parsing GitHub markdown listings."""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import guard for environments without requests installed
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore

# Configure module-level logger
logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)

# Keywords indicating strong application/job related links
APPLICATION_URL_HINTS = (
    "apply",
    "applicant",
    "application",
    "job",
    "jobs",
    "posting",
    "careers",
    "position",
    "opportunity",
    "boards",
)

NEGATION_PATTERNS = (
    "no canada",
    "not canada",
    "outside canada",
    "excluding canada",
)

PROVINCE_ABBREVIATIONS = {
    "ab",
    "bc",
    "mb",
    "nb",
    "nl",
    "ns",
    "nt",
    "nu",
    "on",
    "pe",
    "pei",
    "qc",
    "sk",
    "yt",
}


@dataclass
class NormalizedRow:
    company: str
    role: str
    location: str
    url: str
    notes: str = ""

    def key(self) -> Tuple[str, str, str]:
        return (self.company.lower().strip(), self.role.lower().strip(), self.url.strip())

    def to_output_line(self) -> str:
        return f"{self.company} | {self.role} | {self.location} | {self.url}"


class ScraperError(RuntimeError):
    """Custom exception for scraper errors."""


# ---------------------------------------------------------------------------
# Fetching utilities
# ---------------------------------------------------------------------------

def fetch_raw(owner: str, repo: str, branch: str, path: str, token: Optional[str] = None) -> str:
    """Fetch a raw file from GitHub with retry and backoff.

    Args:
        owner: Repository owner.
        repo: Repository name.
        branch: Branch name.
        path: Path within repository.
        token: Optional GitHub token for authenticated requests.

    Returns:
        The raw text of the file.
    """

    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    backoff = 1.0
    session = requests.Session() if requests is not None else None

    for attempt in range(1, 6):  # up to 5 attempts
        try:
            logger.debug("Fetching URL %s (attempt %d)", url, attempt)
            if session is not None:
                response = session.get(url, headers=headers, timeout=30)
                status = response.status_code
                if status == 429 or status >= 500:
                    raise ScraperError(
                        f"GitHub returned status {status} for {response.url}"
                    )
                response.raise_for_status()
                logger.debug("Fetched %d bytes from %s", len(response.text), response.url)
                return response.text
            else:
                from urllib import error, request

                req = request.Request(url, headers=headers)
                try:
                    with request.urlopen(req, timeout=30) as resp:
                        status = getattr(resp, "status", 200)
                        final_url = resp.geturl()
                        if status == 429 or status >= 500:
                            raise ScraperError(
                                f"GitHub returned status {status} for {final_url}"
                            )
                        data = resp.read()
                        text = data.decode("utf-8")
                        logger.debug("Fetched %d bytes from %s", len(text), final_url)
                        return text
                except error.HTTPError as http_error:
                    if http_error.code == 429 or http_error.code >= 500:
                        raise ScraperError(
                            f"GitHub returned status {http_error.code} for {http_error.geturl()}"
                        ) from http_error
                    raise
        except Exception as exc:  # Broad catch to unify retry logic
            logger.warning("Fetch attempt %d failed: %s", attempt, exc)
            if attempt == 5:
                raise ScraperError(
                    f"Failed to fetch {url}: {exc}"
                ) from exc
            sleep_time = backoff + random.uniform(0, 0.5)
            logger.debug("Sleeping %.2f seconds before retry", sleep_time)
            time.sleep(sleep_time)
            backoff *= 2
    raise ScraperError(f"Failed to fetch {url}")


# ---------------------------------------------------------------------------
# Markdown utilities
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text


def slice_section(markdown: str, anchor: Optional[str]) -> str:
    if not anchor:
        return markdown

    target = anchor.lstrip("#")
    lines = markdown.splitlines()
    section_lines: List[str] = []
    found = False
    current_level = None

    for idx, line in enumerate(lines):
        header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            slug = slugify(header_text)
            if slug == target:
                found = True
                current_level = level
                section_lines.append(line)
                logger.debug("Anchor '%s' matched header '%s' at line %d", anchor, header_text, idx)
                continue
            if found and current_level is not None and level <= current_level:
                break
        if found:
            section_lines.append(line)

    if not found:
        logger.warning("Anchor '%s' not found; using full document", anchor)
        return markdown

    return "\n".join(section_lines).strip()


def _strip_markdown_links(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    links: List[Tuple[str, str]] = []

    def replacement(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        url = match.group(2).strip()
        links.append((label, url))
        return label

    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", replacement, text)
    return cleaned, links


def _split_table_row(line: str) -> List[str]:
    line = line.strip().strip("|")
    parts = [part.strip() for part in re.split(r"\s*\|\s*", line)]
    return parts


def _is_table_separator(line: str) -> bool:
    line = line.strip()
    if not (line.startswith("|") or line.endswith("|")):
        return False
    return bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$", line))


HEADER_ALIASES = {
    "company": "Company",
    "organization": "Company",
    "org": "Company",
    "employer": "Company",
    "role": "Role",
    "position": "Role",
    "title": "Role",
    "job": "Role",
    "location": "Location",
    "locations": "Location",
    "city": "Location",
    "notes": "Notes",
    "details": "Notes",
    "description": "Notes",
    "link": "URL",
    "links": "URL",
    "application": "URL",
    "apply": "URL",
    "url": "URL",
    "posting": "URL",
}


def _map_headers(headers: Sequence[str]) -> Dict[int, str]:
    mapped: Dict[int, str] = {}
    for idx, header in enumerate(headers):
        key = slugify(header).replace("-", " ")
        key = key.replace(" ", "")
        if key in HEADER_ALIASES:
            mapped[idx] = HEADER_ALIASES[key]
    return mapped


def _parse_table(lines: List[str], start_idx: int) -> Tuple[List[Dict[str, str]], int]:
    header_parts = _split_table_row(lines[start_idx])
    mapped_headers = _map_headers(header_parts)
    rows: List[Dict[str, str]] = []
    idx = start_idx + 2  # skip header and separator

    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            break
        if re.match(r"^#{1,6}\s", line):
            break
        if line.strip().startswith("|"):
            parts = _split_table_row(line)
            row: Dict[str, str] = {}
            for col_idx, value in enumerate(parts):
                header = mapped_headers.get(col_idx)
                if header:
                    row[header] = value
            if row:
                rows.append(row)
            idx += 1
        else:
            break
    return rows, idx


def _parse_bullets(lines: List[str], start_idx: int) -> Tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    idx = start_idx
    # We'll handle bullets manually to allow nested content
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if re.match(r"^\s*[-*+]\s+", line):
            primary = re.sub(r"^\s*[-*+]\s+", "", line).strip()
            idx += 1
            continuation: List[str] = []
            while idx < len(lines):
                cont_line = lines[idx]
                if cont_line.strip() and re.match(r"^\s{2,}\S", cont_line):
                    continuation.append(cont_line.strip())
                    idx += 1
                else:
                    break
            cleaned_primary, links = _strip_markdown_links(primary)
            parts = [part.strip() for part in re.split(r"\s+-\s+", cleaned_primary)]
            row: Dict[str, str] = {}
            if parts:
                row["Company"] = parts[0]
            if len(parts) > 1:
                row["Role"] = parts[1]
            if len(parts) > 2:
                row["Location"] = parts[2]
            if len(parts) > 3:
                row["Notes"] = " - ".join(parts[3:])
            if links:
                # Prefer the first link as URL; store in notes if needed
                row.setdefault("URL", links[0][1])
                if len(links) > 1:
                    row.setdefault("Notes", "")
                    extra_links = ", ".join(url for _, url in links[1:])
                    row["Notes"] = (row.get("Notes", "") + f" Additional links: {extra_links}").strip()
            if continuation:
                extra_text = " ".join(continuation).strip()
                if extra_text:
                    row["Notes"] = (row.get("Notes", "") + f" {extra_text}").strip()
            if row:
                rows.append(row)
        else:
            break
    return rows, idx


def parse_markdown_to_rows(markdown: str) -> List[Dict[str, str]]:
    lines = markdown.splitlines()
    idx = 0
    rows: List[Dict[str, str]] = []

    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if re.match(r"^\s*[-*+]\s+", line):
            bullet_rows, idx = _parse_bullets(lines, idx)
            rows.extend(bullet_rows)
            continue
        if line.strip().startswith("|") and idx + 1 < len(lines) and _is_table_separator(lines[idx + 1]):
            table_rows, idx = _parse_table(lines, idx)
            rows.extend(table_rows)
            continue
        idx += 1
    return rows


# ---------------------------------------------------------------------------
# Normalization and filtering
# ---------------------------------------------------------------------------


def _extract_preferred_url(candidates: Iterable[str]) -> str:
    scored: List[Tuple[int, str]] = []
    for url in candidates:
        lower = url.lower()
        score = 0
        for hint in APPLICATION_URL_HINTS:
            if hint in lower:
                score += 1
        scored.append((score, url))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def normalize_row(row: Dict[str, str]) -> NormalizedRow:
    links: List[str] = []
    notes = row.get("Notes", "")
    normalized: Dict[str, str] = {}

    for key in ("Company", "Role", "Location", "Notes", "URL"):
        value = row.get(key, "").strip()
        if not value:
            continue
        cleaned, extracted = _strip_markdown_links(value)
        normalized[key] = cleaned.strip()
        for _, url in extracted:
            links.append(url)

    # Additional link extraction from other keys if URL missing
    if not normalized.get("URL"):
        for key in row:
            if key not in normalized:
                cleaned, extracted = _strip_markdown_links(row[key])
                row[key] = cleaned
                for _, url in extracted:
                    links.append(url)

    if normalized.get("URL"):
        links.insert(0, normalized["URL"])

    url = _extract_preferred_url(links)

    company = normalized.get("Company", "").strip()
    role = normalized.get("Role", "").strip()
    location = normalized.get("Location", "").strip()
    notes = normalized.get("Notes", notes)

    return NormalizedRow(company=company, role=role, location=location, url=url, notes=notes.strip())


def normalize_rows(rows: Sequence[Dict[str, str]]) -> List[NormalizedRow]:
    normalized = [normalize_row(row) for row in rows]
    return [row for row in normalized if any([row.company, row.role, row.location, row.url])]


def _contains_negation(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in NEGATION_PATTERNS)


def _keyword_match(text: str, keywords: Sequence[str]) -> bool:
    lower = text.lower()
    for keyword in keywords:
        keyword_clean = keyword.strip().lower()
        if not keyword_clean:
            continue
        if keyword_clean in PROVINCE_ABBREVIATIONS:
            pattern = re.compile(rf"\b{re.escape(keyword_clean)}\b", re.IGNORECASE)
            if pattern.search(text):
                return True
            continue
        if keyword_clean in lower:
            return True
    return False


def _expand_keywords(keywords: Sequence[str]) -> List[str]:
    expanded = {keyword.strip().lower() for keyword in keywords if keyword.strip()}
    if any("canada" in keyword for keyword in expanded):
        expanded.update({
            "toronto",
            "vancouver",
            "montreal",
            "ottawa",
            "calgary",
            "edmonton",
            "remote (canada)",
            "remote-canada",
            "remote in canada",
        })
        expanded.update(PROVINCE_ABBREVIATIONS)
    return list(expanded)


def match_keywords(row: NormalizedRow, keywords: Sequence[str]) -> bool:
    expanded_keywords = _expand_keywords(keywords)
    if not expanded_keywords:
        return True

    haystacks = [row.location, row.notes]
    for hay in haystacks:
        if not hay:
            continue
        if _contains_negation(hay):
            continue
        if _keyword_match(hay, expanded_keywords):
            return True
    return False


def dedupe_rows(rows: Sequence[NormalizedRow]) -> List[NormalizedRow]:
    seen = set()
    deduped: List[NormalizedRow] = []
    for row in rows:
        key = row.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def write_output(rows: Sequence[NormalizedRow], outfile: str, include_headers: bool) -> None:
    with open(outfile, "w", encoding="utf-8", newline="") as handle:
        if include_headers:
            handle.write("Company | Role | Location | URL\n")
        for row in rows:
            handle.write(row.to_output_line() + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extracts structured rows (Company, Role, Location, URL) from a Markdown file "
            "in a GitHub repo. Useful for scraping job/role listings."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic: fetch full file and write to out.txt\n"
            "  python gitscraper.py --repo owner/repo --path README.md\n\n"
            "  # Target a section by header anchor and filter for Canada\n"
            "  python gitscraper.py --repo owner/repo --path jobs.md \\\n+            \n    --anchor roles --keywords canada --include-headers --outfile jobs_canada.txt\n\n"
            "  # Preview first results without writing a file\n"
            "  python gitscraper.py --repo owner/repo --path listings.md --dry-run --verbose\n\n"
            "Notes:\n"
            "- --repo must be 'owner/repo' (no https://).\n"
            "- --anchor accepts a header slug or text (e.g., 'roles' or '#roles').\n"
            "- Set GITHUB_TOKEN to avoid rate limits (optional).\n"
        ),
    )
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/repo format")
    parser.add_argument("--branch", default="main", help="Git branch name (default: main)")
    parser.add_argument("--path", required=True, help="Path to the file in the repository")
    parser.add_argument("--anchor", help="Markdown header anchor (e.g. #-roles)")
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-separated keywords for filtering (case-insensitive)",
    )
    parser.add_argument("--outfile", default="out.txt", help="Output file path")
    parser.add_argument("--include-headers", action="store_true", help="Include header row")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to output")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # If launched with no arguments, show help and exit clearly
    if argv is None and len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        parser.exit(2)

    return parser.parse_args(argv)


def parse_args_clean(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extracts structured rows (Company, Role, Location, URL) from a Markdown file "
            "in a GitHub repo. Useful for scraping job/role listings."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/repo format")
    parser.add_argument("--branch", default="main", help="Git branch name (default: main)")
    parser.add_argument("--path", required=True, help="Path to the file in the repository")
    parser.add_argument("--anchor", help="Markdown header anchor (e.g., 'roles' or '#roles')")
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-separated keywords for filtering (case-insensitive)",
    )
    parser.add_argument("--outfile", default="out.txt", help="Output file path")
    parser.add_argument("--include-headers", action="store_true", help="Include header row")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to output")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # In interactive mode (no args), main() will prompt instead of exiting

    return parser.parse_args(argv)


def _parse_url_or_repo(input_text: str) -> Tuple[str, str, str, str, Optional[str]]:
    """Parse a GitHub repo or file URL into parts.

    Accepts formats:
    - owner/repo
    - https://github.com/owner/repo
    - https://github.com/owner/repo/blob/branch/path/to/file.md
    - https://raw.githubusercontent.com/owner/repo/branch/path/to/file.md

    Returns: (owner, repo, branch, path, anchor)
    Defaults to branch 'main' and path 'README.md' when not specified.
    """
    text = input_text.strip()
    anchor: Optional[str] = None
    try:
        if text.startswith("http://") or text.startswith("https://"):
            from urllib.parse import urlparse

            parsed = urlparse(text)
            parts = [p for p in parsed.path.split("/") if p]
            anchor = parsed.fragment or None

            if parsed.netloc == "raw.githubusercontent.com" and len(parts) >= 4:
                owner, repo, branch = parts[0], parts[1], parts[2]
                path = "/".join(parts[3:])
                return owner, repo, branch, path, anchor

            if parsed.netloc == "github.com" and len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                # blob URL with explicit branch and path
                if len(parts) >= 4 and parts[2] == "blob":
                    branch = parts[3]
                    path = "/".join(parts[4:]) if len(parts) > 4 else "README.md"
                    return owner, repo, branch, path, anchor
                # repo root URL – default path/branch
                return owner, repo, "main", "README.md", anchor

        # owner/repo short form
        if "/" in text and not text.startswith(("http://", "https://")):
            owner, repo = text.split("/", 1)
            return owner, repo, "main", "README.md", None
    except Exception:
        pass

    raise ScraperError("Unable to parse repository or URL. Provide owner/repo or a GitHub URL.")


def _prompt_interactive() -> argparse.Namespace:
    print("GitScraper interactive mode — leave blank to accept defaults where offered.")
    raw = input("GitHub repo or URL (e.g., owner/repo or https://github.com/owner/repo/blob/branch/file.md): ").strip()
    if not raw:
        raise SystemExit(2)
    owner, repo, branch, path, anchor = _parse_url_or_repo(raw)

    kw = input("Keywords (comma-separated, optional): ").strip()
    keywords = kw

    # Build an argparse-like namespace for reuse below
    return argparse.Namespace(
        repo=f"{owner}/{repo}",
        branch=branch or "main",
        path=path or "README.md",
        anchor=anchor,
        keywords=keywords,
        outfile="out.txt",
        include_headers=False,
        max_rows=None,
        dry_run=False,
        verbose=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    # If launched with no CLI args, switch to interactive prompt mode.
    if argv is None and len(sys.argv) == 1:
        try:
            args = _prompt_interactive()
        except SystemExit as e:
            return int(getattr(e, "code", 2) or 2)
        except ScraperError as exc:
            print(f"Error: {exc}")
            return 2
    else:
        args = parse_args_clean(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    owner_repo = args.repo.strip()
    if "/" not in owner_repo or owner_repo.count("/") != 1 or owner_repo.startswith("http"):
        logger.error("--repo must be in 'owner/repo' format (no protocol)")
        return 1
    owner, repo = owner_repo.split("/", 1)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.debug("No GITHUB_TOKEN detected; unauthenticated requests may be rate-limited.")

    # Friendly run summary
    logger.info(
        "Repo: %s/%s | Branch: %s | Path: %s | Anchor: %s | Filters: %s | Out: %s",
        owner,
        repo,
        args.branch,
        args.path,
        args.anchor or "<entire file>",
        args.keywords or "<none>",
        args.outfile,
    )

    try:
        markdown = fetch_raw(owner, repo, args.branch, args.path, token=token)
    except ScraperError as exc:
        logger.error(str(exc))
        return 1

    section = slice_section(markdown, args.anchor)
    parsed_rows = parse_markdown_to_rows(section)
    if not parsed_rows:
        logger.error("No rows parsed from the specified section")
        return 1
    logger.debug("Parsed %d raw rows", len(parsed_rows))

    normalized_rows = normalize_rows(parsed_rows)
    logger.debug("Normalized %d rows", len(normalized_rows))

    keywords = [keyword.strip() for keyword in args.keywords.split(",") if keyword.strip()]
    matched_rows = [row for row in normalized_rows if match_keywords(row, keywords)]
    if not matched_rows:
        logger.error("No rows matched the provided filters")
        return 1

    deduped_rows = dedupe_rows(matched_rows)
    deduped_rows.sort(key=lambda row: (row.company.lower(), row.role.lower()))

    if args.max_rows is not None:
        deduped_rows = deduped_rows[: args.max_rows]

    if args.dry_run:
        for line in deduped_rows[:10]:
            print(line.to_output_line())
    else:
        write_output(deduped_rows, args.outfile, args.include_headers)

    logger.info(
        "Parsed %d rows; matched %d; wrote %d lines to %s",
        len(parsed_rows),
        len(matched_rows),
        len(deduped_rows),
        args.outfile,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
