"""Freeze the HRCS guide for Claude's prompt (decision 38, open item 3). Writes data/guide/<date>/.

Downloads the health categories overview, the three coding-advice pages and the 21 category pages from hrcsonline.net,
keeps each page's own text word for word (only layout marks are added: # headings, - list items, | table cells) and
drops the site menu. Saves the raw pages, one guide.md and a manifest with a fingerprint of every page.
Never overwrites an earlier download: point GUIDE_FILE in settings.py at the copy to use.
"""
import hashlib
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from settings import LABELS

# ── Settings used only by this step (shared ones live in settings.py) ──
SITE = "https://hrcsonline.net"
ADVICE = ["/health-categories/", "/getting-started/general-approach-to-coding/",
          "/getting-started/general-approach-to-coding/assigning-health-categories/",
          "/getting-started/general-approach-to-coding/common-mistakes/"]
CATEGORY_SLUGS = {
    "blood": "blood", "cancer and neoplasms": "cancer-and-neoplasms", "cardiovascular": "cardiovascular",
    "congenital disorders": "congenital-disorders", "disputed aetiology and other": "disputed-aetiology-other",
    "ear": "ear", "eye": "eye", "generic health relevance": "generic-health-relevance", "infection": "infection",
    "inflammatory and immune system": "inflammatory-and-immune-system", "injuries and accidents": "injuries-and-accidents",
    "mental health": "mental-health", "metabolic and endocrine": "metabolic-and-endocrine",
    "musculoskeletal": "musculoskeletal", "neurological": "neurological",
    "oral and gastrointestinal": "oral-and-gastrointestinal", "renal and urogenital": "renal-and-urogenital",
    "reproductive health and childbirth": "reproductive-health-and-childbirth", "respiratory": "respiratory",
    "skin": "skin", "stroke": "stroke"}
CRAWL_DELAY = 10                                # seconds between requests: the site's robots.txt asks for 10
CHARS_PER_TOKEN = 2.95                          # measured on grant text in the headless test (decision 54)
CREDIT = ("Source: HRCS Online, https://hrcsonline.net. The Health Research Classification System is maintained by the "
          "UKCRC Health Research Analysis Forum. Licensed under Creative Commons Attribution-NonCommercial-NoDerivs 2.0 "
          "England & Wales. Copied word for word on {date}: the health categories overview, three coding-advice pages "
          "and the 21 category pages.")
BLOCKS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "p": "", "li": "- "}

assert list(CATEGORY_SLUGS) == LABELS, "category pages must follow the label order"
date = datetime.now().date().isoformat()
out = Path("data/guide") / date
if out.exists():
    raise SystemExit(f"{out} already exists: a frozen copy is never overwritten")
(out / "raw").mkdir(parents=True)


def page_lines(el):
    """The page's text, block by block, with the words untouched."""
    for c in el.children:
        if not getattr(c, "name", None):                 # loose text directly inside a container
            t = " ".join(str(c).split())
            if t:
                yield t
        elif c.name in BLOCKS:
            t = " ".join(c.get_text().split())
            if t:
                yield BLOCKS[c.name] + t
        elif c.name == "tr":
            yield " | ".join(" ".join(cell.get_text().split()) for cell in c.find_all(["th", "td"]))
        elif c.name not in ("script", "style"):
            yield from page_lines(c)


pages, manifest = [], []
urls = [SITE + p for p in ADVICE] + [f"{SITE}/health-categories/{s}/" for s in CATEGORY_SLUGS.values()]
for n, url in enumerate(urls):
    if n:
        time.sleep(CRAWL_DELAY)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (personal non-commercial research project)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        status, html = r.status, r.read()
    assert status == 200, f"{url}: HTTP {status}"
    slug = url.rstrip("/").split("/")[-1]
    (out / "raw" / f"{slug}.html").write_bytes(html)
    content = BeautifulSoup(html.decode("utf-8"), "html.parser").select_one("main div.content")
    assert content is not None, f"{url}: page text block not found"
    text = "\n".join(page_lines(content))
    assert text.startswith("# "), f"{url}: page does not start with its title"
    pages.append((url, text))
    manifest.append({"url": url, "status": status, "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                     "sha256_html": hashlib.sha256(html).hexdigest(), "chars": len(text)})
    lines = text.splitlines()
    print(f"{len(text):>6} chars | {lines[0][:60]:<60} | ends: {lines[-1][:70]}")

for label, (url, text) in zip(LABELS, pages[len(ADVICE):]):
    assert text.splitlines()[0][2:].lower() == label, f"{url}: title {text.splitlines()[0]!r} is not {label!r}"

guide = CREDIT.format(date=date) + "".join(f"\n\n---\n\n{text}\n\nSource: {url}" for url, text in pages) + "\n"
(out / "guide.md").write_text(guide)
(out / "manifest.json").write_text(json.dumps({"downloaded": date, "pages": manifest}, indent=1))
print(f"\n{len(pages)} pages -> {out / 'guide.md'}: {len(guide):,} characters, "
      f"about {len(guide) / CHARS_PER_TOKEN:,.0f} Claude tokens (estimate)")
