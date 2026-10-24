"""Local-only cohort results; self-contained and no remote assets."""
import json
from pathlib import Path

from .study import clean


def report(path, summary, values, correlations):
    payload = json.dumps(clean({"summary": summary, "values": values, "correlations": correlations}),
                         allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("static") / "study.html"
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(template.read_text(encoding="utf-8").replace("__DATA__", payload))
