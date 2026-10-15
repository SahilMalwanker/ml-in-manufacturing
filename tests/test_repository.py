import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

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


def test_svg_artwork_is_self_contained():
    svgs = list((ROOT / "docs/images").glob("*.svg"))
    assert svgs
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        root = ElementTree.fromstring(text)
        assert root.tag == "{http://www.w3.org/2000/svg}svg", path
        assert root.get("viewBox") and root.get("aria-label"), path
        assert path.stat().st_size < 100_000, path
        lowered = text.lower()
        for forbidden in ("<script", "<foreignobject", "javascript:", "@import", "<image"):
            assert forbidden not in lowered, (path, forbidden)
        assert not re.search(r"\son[a-z]+\s*=", lowered), path
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg\b)", text), path


def test_published_files_contain_no_network_or_user_paths():
    private_ip = re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    )
    user_path = re.compile(r"[A-Za-z]:\\Users\\|/home/[a-z]|/Users/[A-Za-z]")
    folders = (".github", "data", "docs", "plc", "results", "scripts", "simit", "src", "tests")
    files = [*ROOT.glob("*.md"), *ROOT.glob("*.toml")]
    files += [path for folder in folders for path in (ROOT / folder).rglob("*") if path.is_file()]
    for path in files:
        if path.suffix in {".png", ".gif"} or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not private_ip.search(text), (path, private_ip.search(text))
        assert not user_path.search(text), (path, user_path.search(text))
