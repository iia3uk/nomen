"""Export artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import orjson

from nomen.models import Candidate


def export_all(
    out: Path,
    checked: list[Candidate],
    rejected: list[Candidate],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    all_rows = checked + rejected

    (out / "results.json").write_bytes(
        orjson.dumps([c.model_dump(mode="json") for c in all_rows], option=orjson.OPT_INDENT_2)
    )
    (out / "rejected.json").write_bytes(
        orjson.dumps(
            [c.model_dump(mode="json") for c in rejected if not c.clean],
            option=orjson.OPT_INDENT_2,
        )
    )

    fields = [
        "name",
        "display_name",
        "generator",
        "clean",
        "overall",
        "beauty_score",
        "brand_score",
        "novelty_score",
        "collision_score",
        "phonetic_root",
        "cv_pattern",
        "pronounceability",
        "memorability",
        "typing_speed",
        "visual_balance",
        "brand_strength",
        "premium_feel",
        "international_readability",
        "collision_probability",
        "seo_uniqueness",
        "domains_registered",
        "rejection_reasons",
        "errors",
    ]
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in all_rows:
            w.writerow(
                {
                    "name": c.name,
                    "display_name": c.display_name,
                    "generator": c.generator,
                    "clean": c.clean,
                    "overall": c.scores.overall,
                    "beauty_score": c.scores.beauty_score,
                    "brand_score": c.scores.brand_score,
                    "novelty_score": c.scores.novelty_score,
                    "collision_score": c.scores.collision_score,
                    "phonetic_root": c.scores.phonetic_root,
                    "cv_pattern": c.scores.cv_pattern,
                    "pronounceability": c.scores.pronounceability,
                    "memorability": c.scores.memorability,
                    "typing_speed": c.scores.typing_speed,
                    "visual_balance": c.scores.visual_balance,
                    "brand_strength": c.scores.brand_strength,
                    "premium_feel": c.scores.premium_feel,
                    "international_readability": c.scores.international_readability,
                    "collision_probability": c.scores.collision_probability,
                    "seo_uniqueness": c.scores.seo_uniqueness,
                    "domains_registered": ",".join(c.domains_registered),
                    "rejection_reasons": " | ".join(c.rejection_reasons),
                    "errors": " | ".join(c.errors),
                }
            )

    clean = sorted(
        [c for c in checked if c.clean],
        key=lambda c: (-c.scores.overall, -c.scores.beauty_score, -c.scores.brand_score, c.name),
    )
    (out / "clean_names.txt").write_text(
        "\n".join(
            f"{c.display_name}\toverall={c.scores.overall:.1f}\t"
            f"beauty={c.scores.beauty_score:.1f}\tbrand={c.scores.brand_score:.1f}\t"
            f"nov={c.scores.novelty_score:.1f}\troot={c.scores.phonetic_root}\t"
            f"{c.scores.cv_pattern}\t{c.generator}"
            for c in clean
        )
        + ("\n" if clean else ""),
        encoding="utf-8",
    )
    for fname, n in (("top100.txt", 100), ("top20.txt", 20), ("top10.txt", 10)):
        top = clean[:n]
        (out / fname).write_text(
            "\n".join(
                f"{c.display_name}\t{c.scores.overall:.1f}\t"
                f"{c.scores.beauty_score:.1f}\t{c.scores.brand_score:.1f}\t"
                f"{c.scores.phonetic_root}"
                for c in top
            )
            + ("\n" if top else ""),
            encoding="utf-8",
        )
