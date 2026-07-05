#!/usr/bin/env python3
"""Render a mobile-friendly power ratings graphic with team logos."""

import argparse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

RATINGS_DIR = Path(__file__).parent / "historical_ratings"
LOGO_CACHE = Path("/tmp/nfl_logos")

# ESPN logo slugs (500px). Carolina uses the dark variant — the regular one
# disappears on a dark background.
LOGO_SLUG = {t: t.lower() for t in [
    "ARI", "ATL", "BAL", "BUF", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LAC", "LV", "MIA", "MIN", "NE",
    "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN"]}
LOGO_SLUG.update({"WAS": "wsh", "LA": "lar"})
LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
LOGO_URL_DARK = "https://a.espncdn.com/i/teamlogos/nfl/500-dark/{slug}.png"

BG = "#10141c"
ROW_A = "#161b26"
TEXT = "#e8eaf0"
MUTED = "#8a93a6"


def get_logo(team: str):
    LOGO_CACHE.mkdir(exist_ok=True)
    path = LOGO_CACHE / f"{team}.png"
    if not path.exists():
        if team == "CAR":
            url = LOGO_URL_DARK.format(slug="car")
        else:
            url = LOGO_URL.format(slug=LOGO_SLUG[team])
        urllib.request.urlretrieve(url, path)
    return mpimg.imread(path)


def make_graphic(season: int, output: Path):
    df = pd.read_csv(RATINGS_DIR / f"ratings_{season}.csv")
    df = df.sort_values("rank").reset_index(drop=True)
    per_col = 16

    fig, ax = plt.subplots(figsize=(6.75, 8.5), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    lim = float(df["power_rating"].abs().max())
    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(-lim, lim)

    for i, row in df.iterrows():
        col = i // per_col
        y = per_col - 1 - (i % per_col)
        x0 = col * 1.0

        if (i % per_col) % 2 == 0:
            ax.add_patch(plt.Rectangle((x0 + 0.01, y - 0.5), 0.97, 1.0,
                                       color=ROW_A, zorder=0))

        r = row["power_rating"]
        ax.text(x0 + 0.075, y, f"{row['rank']}", ha="right", va="center",
                fontsize=10.5, color=MUTED, zorder=3)

        img = get_logo(row["team"])
        zoom = 34.0 / max(img.shape[:2])
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom),
                            (x0 + 0.155, y), frameon=False, zorder=3)
        ax.add_artist(ab)

        ax.text(x0 + 0.25, y, row["team"], ha="left", va="center",
                fontsize=12, color=TEXT, fontweight="bold", zorder=3)
        ax.text(x0 + 0.95, y, f"{r:+.1f}", ha="right", va="center",
                fontsize=12, color=cmap(norm(r)), fontweight="bold", zorder=3)

    ax.axvline(1.0, color=MUTED, lw=0.6, alpha=0.35, zorder=1)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.65, per_col - 0.35)
    ax.axis("off")

    fig.suptitle(f"{season} NFL PRESEASON POWER RATINGS",
                 fontsize=15.5, fontweight="bold", color=TEXT, y=0.965)
    ax.set_title("Expected point spread vs. an average team on a neutral field",
                 fontsize=8.5, color=MUTED, pad=12)
    fig.text(0.5, 0.015, "mpg000f.github.io/cbb_power_rating",
             ha="center", fontsize=8, color=MUTED)

    fig.tight_layout(rect=[0, 0.025, 1, 0.95])
    fig.savefig(output, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL power ratings graphic")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else \
        Path(__file__).parent / f"nfl_{args.season}_preseason_ratings.png"
    make_graphic(args.season, out)
