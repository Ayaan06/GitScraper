import os
import sys
import textwrap

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gitscraper import (
    NormalizedRow,
    dedupe_rows,
    match_keywords,
    normalize_rows,
    parse_markdown_to_rows,
)


def test_parse_markdown_table_with_canada_filter():
    markdown = textwrap.dedent(
        """
        | Company | Role | Location | Application |
        | --- | --- | --- | --- |
        | Shopify | Software Engineer Intern | Ottawa, ON (Canada) | [Apply](https://shopify.com/jobs) |
        | United States Co | SWE Intern | Austin, TX | [Apply](https://usco.example/jobs) |
        | Multinational | SWE Intern | Remote (US or Canada) | [Info](https://multi.example/info) |
        """
    ).strip()

    rows = parse_markdown_to_rows(markdown)
    normalized = normalize_rows(rows)
    keywords = ["canada"]
    filtered = [row for row in normalized if match_keywords(row, keywords)]

    assert len(filtered) == 2
    locations = {row.location for row in filtered}
    assert "Ottawa, ON (Canada)" in locations
    assert any("Remote" in location for location in locations)


def test_parse_bullet_list_extracts_fields_and_links():
    markdown = textwrap.dedent(
        """
        - [Hatch](https://hatch.com/careers) - Software Engineering Intern - Mississauga, ON (Canada)
          Apply via careers page.
        - Example Corp - Data Science Intern - Remote (US only) - [Apply](https://example.com/us)
        """
    ).strip()

    rows = parse_markdown_to_rows(markdown)
    normalized = normalize_rows(rows)

    assert normalized[0].company == "Hatch"
    assert normalized[0].role == "Software Engineering Intern"
    assert normalized[0].location == "Mississauga, ON (Canada)"
    assert normalized[0].url == "https://hatch.com/careers"

    assert normalized[1].company == "Example Corp"
    assert normalized[1].url == "https://example.com/us"


def test_match_keywords_handles_remote_us_or_canada():
    row = NormalizedRow(
        company="Remote Co",
        role="SWE Intern",
        location="Remote (US or Canada)",
        url="https://example.com",
        notes="",
    )
    assert match_keywords(row, ["canada"]) is True

    row_us_only = NormalizedRow(
        company="US Co",
        role="SWE Intern",
        location="Remote (US only)",
        url="https://example.com",
        notes="",
    )
    assert match_keywords(row_us_only, ["canada"]) is False


def test_dedupe_rows_uses_company_role_url_key():
    row1 = NormalizedRow("Shopify", "SWE Intern", "Toronto, ON", "https://example.com/apply")
    row2 = NormalizedRow("Shopify", "SWE Intern", "Toronto, ON", "https://example.com/apply")
    row3 = NormalizedRow("Shopify", "SWE Intern", "Toronto, ON", "https://example.com/careers")

    deduped = dedupe_rows([row1, row2, row3])

    assert len(deduped) == 2
    urls = {row.url for row in deduped}
    assert "https://example.com/apply" in urls
    assert "https://example.com/careers" in urls
