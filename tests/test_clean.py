from pathlib import Path

from docrag.ingest.clean import clean_page
from docrag.ingest.sources import RawPage


def _page(tmp_path: Path, rel: str, text: str, flavor: str, files: dict[str, str] | None = None) -> RawPage:
    for p, body in (files or {}).items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(body, encoding="utf-8")
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    return RawPage("s", rel, f, tmp_path, "u", "c", flavor)


def test_mkdocs_includes_admonitions_links(tmp_path):
    md = (
        "# First Steps { #first-steps }\n\n{* ../docs_src/app.py hl[1] *}\n\n"
        "/// tip | Hot reload\n\nUse `--reload`.\n\n///\n\n"
        "See [the guide](https://x.y/z) and [`Model`][pydantic.Model].\n\n"
        '!!! note "Heads up"\n    Indented body.\n'
    )
    title, text = clean_page(_page(tmp_path, "docs/a.md", md, "mkdocs", {"docs_src/app.py": "app = FastAPI()\n"}))
    assert title == "First Steps"
    assert "```python\napp = FastAPI()\n```" in text
    assert "**Hot reload:**" in text and "///" not in text
    assert "See the guide and `Model`." in text
    assert "**Heads up:**\nIndented body." in text and "{ #" not in text


def test_mintlify_keeps_python_drops_js_and_inlines_snippets(tmp_path):
    mdx = (
        "---\ntitle: Agents\n---\nimport IntroPy from '/snippets/intro-py.mdx';\n\n"
        "Text @[`create_agent`] here.\n\n:::python\n<IntroPy />\n:::\n:::js\nJS ONLY\n:::\n"
        "<Note>\nA note.\n</Note>\n"
    )
    snippet = "<CodeGroup>\n    ```python\n    agent = create_agent()\n    ```\n</CodeGroup>\n"
    title, text = clean_page(_page(tmp_path, "src/oss/a.mdx", mdx, "mintlify", {"src/snippets/intro-py.mdx": snippet}))
    assert title == "Agents" and text.startswith("# Agents")
    assert "Text `create_agent` here." in text
    assert "```python\nagent = create_agent()\n```" in text
    assert "JS ONLY" not in text and "<Note>" not in text and "A note." in text


def test_hugo_shortcodes(tmp_path):
    md = (
        "---\ntitle: Collections\n---\n# Collections\n\n"
        '{{< code-snippet path="/documentation/snip/create/" >}}\n\n{{< prompt "x" >}}\nDone.\n'
    )
    files = {"qdrant-landing/content/documentation/snip/create/python.py": "client.create_collection()\n"}
    _, text = clean_page(_page(tmp_path, "qdrant-landing/content/documentation/c.md", md, "hugo", files))
    assert "```python\nclient.create_collection()\n```" in text and "{{" not in text


def test_title_ignores_code_comments_and_falls_back_to_filename(tmp_path):
    md = "Intro text.\n\n```python\n# ignore-above\nx = 1\n```\n"
    title, text = clean_page(_page(tmp_path, "docs/concepts/json_schema.md", md, "mkdocs"))
    assert title == "Json schema" and text.startswith("# Json schema")
