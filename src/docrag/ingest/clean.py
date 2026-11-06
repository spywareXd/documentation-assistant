"""Turn raw docs (MkDocs / Mintlify MDX / Hugo) into plain markdown.

Goals: keep prose, headings and Python code; drop site chrome, JS-only blocks,
HTML/JSX tags and shortcodes; inline code that pages pull from include files.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from docrag.ingest.sources import RawPage

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9.-]*(?:\s[^<>]*)?/?>")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
HEADING_ANCHOR = re.compile(r"(^#{1,6} .*?)\s*\{\s*#[^}]*\}\s*$", re.M)
ATTR_LIST = re.compile(r"\{\s*[.#][^}]*\}")


def split_front_matter(text: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


def _fence(code: str, lang: str = "python") -> str:
    return f"\n```{lang}\n{code.rstrip()}\n```\n"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------- mkdocs
def _mkdocs_includes(text: str, page_dir: Path) -> str:
    """FastAPI: `{* ../../docs_src/x.py hl[3] *}` and older `{!> ../../docs_src/x.py !}`."""

    def repl(m: re.Match) -> str:
        code = _read((page_dir / m.group(1)).resolve())
        return _fence(code) if code else ""

    text = re.sub(r"\{\*\s*(\S+)[^*]*\*\}", repl, text)
    return re.sub(r"\{!>?\s*(\S+?)\s*!\}", repl, text)


def _mkdocs_admonitions(text: str) -> str:
    out: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # FastAPI style:  /// tip | Title   ...   ///
        m = re.match(r"^///\s*(\w+)?\s*(?:\|\s*(.*))?$", line.strip())
        if m:
            kind, title = m.group(1), m.group(2)
            if kind and kind not in ("details",):
                out.append(f"**{(title or kind).strip().capitalize()}:**")
            elif title:
                out.append(f"**{title.strip()}**")
            i += 1
            continue
        # Material style:  !!! note "Title"  / ??? api "Title"  followed by an indented body
        m = re.match(r'^(\s*)(!!!|\?\?\?\+?)\s*(\w+)(?:\s+"([^"]*)")?\s*$', line)
        if m:
            indent, kind, title = len(m.group(1)), m.group(3), m.group(4)
            body: list[str] = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                body.append(lines[i][indent + 4:] if lines[i].strip() else "")
                i += 1
            if kind != "api":  # api blocks are just lists of reference links
                out.append(f"**{(title or kind).capitalize()}:**")
                out.extend(_mkdocs_admonitions("\n".join(body)).split("\n"))
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def clean_mkdocs(text: str, page: RawPage) -> str:
    text = _mkdocs_includes(text, page.abs_path.parent)
    text = _mkdocs_admonitions(text)
    text = re.sub(r"^:::\s+\S+.*$", "", text, flags=re.M)  # mkdocstrings directives
    text = re.sub(r"^\s*(options|members):.*$", "", text, flags=re.M)
    return text


# --------------------------------------------------------------------------- mintlify
IMPORT_LINE = re.compile(r"^import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"];?\s*$", re.M)


def _drop_language_blocks(text: str, keep: str = "python") -> str:
    """Mintlify `:::python ... :::` / `:::js ... :::` conditional blocks."""
    out, mode = [], None  # mode: None | "keep" | "drop"
    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r"^:::(\w+)$", s)
        if m:
            mode = "keep" if m.group(1) == keep else "drop"
            continue
        if s == ":::" and mode:
            mode = None
            continue
        if mode != "drop":
            out.append(line)
    return "\n".join(out)


def clean_mintlify(text: str, page: RawPage, depth: int = 0) -> str:
    imports = {name: path for name, path in IMPORT_LINE.findall(text)}
    text = IMPORT_LINE.sub("", text)
    text = _drop_language_blocks(text)

    def inline_snippet(m: re.Match) -> str:
        path = imports.get(m.group(1))
        if not path or depth > 2:
            return ""
        raw = _read(page.repo_dir / "src" / path.lstrip("/"))
        if raw is None:
            return ""
        _, body = split_front_matter(raw)
        return "\n" + clean_mintlify(body, page, depth + 1) + "\n"

    text = re.sub(r"<([A-Z]\w*)\s*/>", inline_snippet, text)
    text = re.sub(r"@\[([^\]]+)\](?:\[[^\]]*\])?", r"\1", text)  # @[`create_agent`] autorefs
    # keep titles of tabs / accordions / cards / steps as bold labels
    text = re.sub(r"<(?:Accordion|Tab|Card|Step|Expandable)\b[^>]*?title=\"([^\"]+)\"[^>]*>", r"\n**\1**\n", text)
    return text


# --------------------------------------------------------------------------- hugo
def clean_hugo(text: str, page: RawPage) -> str:
    content_root = page.repo_dir / "qdrant-landing" / "content"

    def snippet(m: re.Match) -> str:
        folder = content_root / m.group(1).strip("/")
        code = _read(folder / "python.py")
        return _fence(code) if code else ""

    text = re.sub(r"\{\{<\s*code-snippet\s+path=\"([^\"]+)\"[^>]*>\}\}", snippet, text)
    text = re.sub(r"\{\{[<%].*?[%>]\}\}", "", text, flags=re.S)  # any other shortcode
    return text


# --------------------------------------------------------------------------- common
def _strip_markup(text: str) -> str:
    """Strip tags/links outside fenced code; keep code blocks verbatim except HTML tags."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            parts[i] = HTML_TAG.sub("", part)  # e.g. <font> in FastAPI console output
            continue
        part = re.sub(r"<!--.*?-->", "", part, flags=re.S)
        part = MD_IMAGE.sub("", part)
        part = MD_LINK.sub(r"\1", part)
        part = REF_LINK.sub(r"\1", part)
        part = HTML_TAG.sub("", part)
        part = HEADING_ANCHOR.sub(r"\1", part)
        part = ATTR_LIST.sub("", part)
        parts[i] = part
    return "".join(parts)


def _dedent_fences(text: str) -> str:
    """Mintlify nests fences inside components with indentation; unindent fenced blocks."""
    out, fence_indent = [], None
    for line in text.split("\n"):
        m = re.match(r"^(\s*)```", line)
        if m and fence_indent is None:
            fence_indent = len(m.group(1))
            out.append(line[fence_indent:])
            continue
        if fence_indent is not None:
            if m:
                out.append(line.lstrip())
                fence_indent = None
            else:
                out.append(line[fence_indent:] if line[:fence_indent].strip() == "" else line)
            continue
        out.append(line)
    return "\n".join(out)


CLEANERS = {"mkdocs": clean_mkdocs, "mintlify": clean_mintlify, "hugo": clean_hugo}


def clean_page(page: RawPage) -> tuple[str, str]:
    """Return (title, clean markdown)."""
    raw = page.abs_path.read_text(encoding="utf-8")
    meta, body = split_front_matter(raw)
    body = CLEANERS[page.flavor](body, page)
    body = _dedent_fences(body)
    body = _strip_markup(body)
    body = re.sub(r"[ \t]+$", "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    title = str(meta.get("title") or "").strip()
    if not title:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        title = m.group(1).strip() if m else page.rel_path
    if not re.match(r"^#\s", body):  # make sure the page starts with its title as H1
        body = f"# {title}\n\n{body}"
    return title, body
