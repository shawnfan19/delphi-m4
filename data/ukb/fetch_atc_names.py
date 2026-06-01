"""Fetch human-readable names for the prescriptions-pack ATC codes.

The `prescriptions` expansion pack tokenizes drugs as bare ATC codes (e.g.
``R05CA``). This script resolves each code to its name from the authoritative WHO
ATC/DDD index (hosted by the Norwegian Institute of Public Health at
``atcddd.fhi.no``) and writes a flat ``code: name`` mapping to
``dictionary/atc_names.yaml``.

The set of codes to resolve is read from ``dictionary/prescriptions_tokenizer.yaml``
so the output covers exactly the pack's vocabulary. Fails loudly if any code cannot
be resolved -- we never want a silent gap.

Run: ``python data/ukb/fetch_atc_names.py`` (needs network).
"""

import argparse
import html as html_lib
import re
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DICT_DIR = HERE / "dictionary"

URL = "https://atcddd.fhi.no/atc_ddd_index/?code={code}&showdescription=no"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
# Anchor whose href ``code`` exactly matches the queried code carries the level name.
# Capture the full inner HTML (names may contain inner tags, e.g. H<sub>2</sub>).
ANCHOR_RE = re.compile(r'href="[^"]*code=([A-Z0-9]+)[^"]*"[^>]*>(.*?)</a>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def fetch_name(code: str, timeout: float) -> str | None:
    """Return the ATC level name for ``code`` from the WHO index, or None."""
    req = urllib.request.Request(
        URL.format(code=code), headers={"User-Agent": USER_AGENT}
    )
    html = (
        urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    )
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    for c, inner in ANCHOR_RE.findall(html):
        if c != code:
            continue
        txt = html_lib.unescape(TAG_RE.sub("", inner))
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt.lower() != "show text from guidelines":
            return txt
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizer",
        default=str(DICT_DIR / "prescriptions_tokenizer.yaml"),
        help="tokenizer YAML whose keys are the ATC codes to resolve",
    )
    p.add_argument(
        "--out",
        default=str(DICT_DIR / "atc_names.yaml"),
        help="output YAML (flat code: name mapping)",
    )
    p.add_argument(
        "--sleep", type=float, default=0.2, help="delay between requests (s)"
    )
    p.add_argument(
        "--timeout", type=float, default=30.0, help="per-request timeout (s)"
    )
    args = p.parse_args(argv)

    with open(args.tokenizer) as f:
        codes = list(yaml.safe_load(f).keys())

    names: dict[str, str] = {}
    missing: list[str] = []
    for i, code in enumerate(codes, 1):
        name = fetch_name(code, args.timeout)
        if name is None:
            missing.append(code)
            print(f"[{i}/{len(codes)}] {code}: UNRESOLVED")
        else:
            names[code] = name
            print(f"[{i}/{len(codes)}] {code}: {name}")
        time.sleep(args.sleep)

    if missing:
        raise SystemExit(
            f"{len(missing)} ATC code(s) unresolved -- refusing to write a partial "
            f"mapping: {missing}"
        )

    with open(args.out, "w") as f:
        yaml.dump(
            names, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    print(f"wrote {len(names)} names to {args.out}")


if __name__ == "__main__":
    main()
