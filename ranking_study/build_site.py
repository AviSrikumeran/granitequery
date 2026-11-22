"""Build the project website from results/results.json.

Run from the ranking_study folder. The page is written to ../docs/index.html,
which GitHub Pages and Vercel serve. It is a single self contained HTML file
with the results and the demo pool embedded.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def embed(obj):
    # Keep the JSON safe inside a <script> tag.
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def main():
    template = Path("site/template.html").read_text(encoding="utf8")
    results = json.loads(Path("results/results.json").read_text(encoding="utf8"))
    demo = json.loads(Path("results/demo_pool.json").read_text(encoding="utf8"))
    html = (template.replace("__RESULTS__", embed(results))
                    .replace("__DEMO__", embed(demo))
                    .replace("__STOPWORDS__", json.dumps(sorted(ENGLISH_STOP_WORDS))))
    out = Path("../docs/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf8")
    print(f"Wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
