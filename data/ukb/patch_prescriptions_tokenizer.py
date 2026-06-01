"""Decorate the prescriptions-pack tokenizer keys with human-readable ATC names.

The `prescriptions` expansion pack tokenizes drugs as bare ATC codes (``R05CA``).
This rewrites the tokenizer keys to ``CODE_name`` (e.g.
``R05CA_expectorants``), matching the ``code_meaning`` convention used by the
``meds``/``summary_ops`` packs, while keeping every token *index* unchanged -- so
the on-disk ``data.bin`` (which stores indices) stays valid and the index<->concept
pairing is preserved by construction.

Names come from ``dictionary/atc_names.yaml`` (produced by ``fetch_atc_names.py``).
Both copies of the tokenizer are rewritten byte-identically: the canonical
``dictionary/prescriptions_tokenizer.yaml`` and the runtime copy under the data
volume (``ukb_real_data/expansion_packs/prescriptions/tokenizer.yaml``), since the
dictionary copy is canonical and the data-volume copy is its derived twin.

Idempotent: the ATC code is recovered as the prefix before the first ``_``, so
re-running on an already-patched tokenizer reproduces the same result.

Run: ``python data/ukb/patch_prescriptions_tokenizer.py``
"""

import argparse
import re
from pathlib import Path

import yaml

from delphi.env import DELPHI_DATA_READ

HERE = Path(__file__).resolve().parent
DICT_DIR = HERE / "dictionary"
DATA_TOKENIZER = (
    Path(DELPHI_DATA_READ)
    / "ukb_real_data"
    / "expansion_packs"
    / "prescriptions"
    / "tokenizer.yaml"
)


def slugify(name: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to single underscores, strip ends."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_patched(tokenizer: dict[str, int], names: dict[str, str]) -> dict[str, int]:
    """Map ``{atc_code: idx}`` -> ``{CODE_name: idx}``, preserving order and indices."""
    patched: dict[str, int] = {}
    for key, idx in tokenizer.items():
        code = key.split("_", 1)[
            0
        ]  # bare code, or recovered from an already-patched key
        if code not in names:
            raise SystemExit(f"no ATC name for code {code!r} (key {key!r})")
        patched[f"{code}_{slugify(names[code])}"] = idx

    # Invariants: indices unchanged, one new key per old key, keys unique & code-prefixed.
    if set(patched.values()) != set(tokenizer.values()):
        raise SystemExit("index set changed -- aborting")
    if len(patched) != len(tokenizer):
        raise SystemExit("key count changed (name collision?) -- aborting")
    return patched


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizer",
        default=str(DICT_DIR / "prescriptions_tokenizer.yaml"),
        help="canonical prescriptions tokenizer (read + rewritten)",
    )
    p.add_argument(
        "--names",
        default=str(DICT_DIR / "atc_names.yaml"),
        help="ATC code -> name mapping from fetch_atc_names.py",
    )
    p.add_argument(
        "--data-tokenizer",
        default=str(DATA_TOKENIZER),
        help="runtime (data-volume) copy of the tokenizer; rewritten byte-identically",
    )
    args = p.parse_args(argv)

    with open(args.tokenizer) as f:
        tokenizer = yaml.safe_load(f)
    with open(args.names) as f:
        names = yaml.safe_load(f)

    patched = build_patched(tokenizer, names)
    text = yaml.dump(
        patched, default_flow_style=False, sort_keys=False, allow_unicode=True
    )

    for path in (args.tokenizer, args.data_tokenizer):
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {len(patched)} entries to {path}")

    # The two copies must be byte-identical.
    a = Path(args.tokenizer).read_bytes()
    b = Path(args.data_tokenizer).read_bytes()
    if a != b:
        raise SystemExit("canonical and data-volume copies differ -- aborting")
    print("both copies byte-identical")


if __name__ == "__main__":
    main()
