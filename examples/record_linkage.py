"""Record-linkage example for polars-stringsim.

Run with:

    uv run --with polars --with pytest --with maturin \\
        python examples/record_linkage.py

(after building the plugin with `maturin develop`).

Demonstrates:
  1. Single-metric scoring expressions
  2. Hybrid multi-metric scoring (pf.hybrid_score)
  3. Per-metric breakdown (return_breakdown=True)
  4. fuzzy_join with blocking
  5. deduplicate with composite threshold
  6. pairwise_compare for tuning
"""
from __future__ import annotations

import polars as pl

import polars_stringsim as pf


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}")


def main() -> None:
    # --- Sample data: messy customer names vs a clean DB ---
    customers = pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "customer_name": [
                "Robert Smith",
                "Catherine Jones",
                "Jon Smyth",
                "William Brown",
                "Katherine Jones",
                "Rob Smith",
            ],
        }
    )
    db = pl.DataFrame(
        {
            "db_id": [101, 102, 103, 104],
            "db_name": [
                "Robert Smyth",
                "Katherine Jones",
                "William Brown",
                "Catherine Jonse",
            ],
        }
    )

    banner("1. Single-metric: Jaro-Winkler on a side-by-side frame")
    pairs = customers.join(db, how="cross").head(6)
    print(
        pairs.select("customer_name", "db_name").with_columns(
            jaro_winkler=pf.jaro_winkler("customer_name", "db_name"),
            soundex=pf.soundex_sim("customer_name", "db_name"),
        )
    )

    banner("2. Hybrid score: spelling + phonetic + token, one Rust call")
    print(
        pairs.select("customer_name", "db_name").with_columns(
            hybrid=pf.hybrid_score(
                "customer_name",
                "db_name",
                algorithms=["jaro_winkler", "double_metaphone", "trigram_jaccard"],
                weights=[0.5, 0.3, 0.2],
            )
        )
    )

    banner("3. Pre-built scorer (pf.name_default) + per-metric breakdown")
    print(
        pairs.select("customer_name", "db_name").with_columns(
            bd=pf.combine(
                [
                    pf.jaro_winkler("customer_name", "db_name"),
                    pf.double_metaphone_sim("customer_name", "db_name"),
                    pf.trigram_jaccard("customer_name", "db_name"),
                ],
                weights=[0.5, 0.3, 0.2],
                return_breakdown=True,
            )
        )
    )

    banner("4. fuzzy_join with first-chars blocking")
    joined = pf.fuzzy_join(
        customers,
        db,
        left_on="customer_name",
        right_on="db_name",
        algorithms=["jaro_winkler", "double_metaphone"],
        weights=[0.6, 0.4],
        threshold=0.75,
        top_k=1,
        block="first_chars",
        block_n=1,
    )
    print(joined.select("customer_id", "customer_name", "db_id", "db_name_right", "score"))

    banner("5. deduplicate: collapse name variants within one frame")
    messy = pl.DataFrame(
        {
            "name": [
                "Robert Smith",
                "Rob Smith",
                "Robert Smyth",
                "Catherine Jones",
                "Katherine Jones",
                "William Brown",
            ]
        }
    )
    print("Input:")
    print(messy)
    print("\nAfter deduplicate (composite_threshold=0.8):")
    print(
        pf.deduplicate(
            messy,
            on="name",
            algorithms=["jaro_winkler", "double_metaphone"],
            weights=[0.6, 0.4],
            composite_threshold=0.8,
            block="first_chars",
            block_n=1,
        )
    )

    banner("6. pairwise_compare for threshold tuning")
    print(
        pf.pairwise_compare(
            customers,
            db,
            left_on="customer_name",
            right_on="db_name",
            algorithms=["jaro_winkler", "double_metaphone", "trigram_jaccard"],
            weights=[0.5, 0.3, 0.2],
            block="first_chars",
            block_n=1,
        ).select("customer_name", "db_name_right", "combined")
    )

    banner("Done — all scoring ran in Rust, orchestrated from Python.")


if __name__ == "__main__":
    main()
