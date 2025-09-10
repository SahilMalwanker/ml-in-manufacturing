import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    ROOT / "README.md",
    *ROOT.glob("docs/*.md"),
    *ROOT.glob("data/*.md"),
    *ROOT.glob("plc/*.md"),
    *ROOT.glob("simit/*.md"),
]


def test_documentation_links_resolve():
    for document in DOCUMENTS:
        text = re.sub(r"```.*?```", "", document.read_text(encoding="utf-8"), flags=re.DOTALL)
        links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        links += re.findall(r'src="([^"]+)"', text)
        for link in links:
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path:
                continue
            target = (document.parent / unquote(parsed.path)).resolve()
            assert target.is_relative_to(ROOT), (document, link)
            assert target.exists(), (document, link)


def test_benchmark_has_all_configurations():
    table = pd.read_csv(ROOT / "results/benchmark.csv")
    assert len(table) == 100
    assert not table.duplicated(["scenario", "model", "window"]).any()
    assert set(table.scenario) == {"R-R", "D-D", "R-D", "D-R", "C-C"}
    assert set(table.model) == {"svc", "dt", "rf", "mlp"}
    assert set(table.window) == {0, 3, 5, 10, 20}
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert table[metric].between(0, 1).all()
    assert (table.loc[table.window == 3, "test_classes"] == 9).all()
    combined = table[(table.scenario == "C-C") & (table.model == "svc") & (table.window == 3)].iloc[0]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{combined.accuracy:.2%} accuracy / {combined.f1:.2%} macro F1" in readme


def test_documentation_images_are_valid():
    for path in [*(ROOT / "docs/images").glob("*.png"), *(ROOT / "docs/images").glob("*.gif")]:
        assert path.stat().st_size < 3_000_000, path
        with Image.open(path) as image:
            assert image.width >= 600 and image.height >= 200, path
            image.verify()
