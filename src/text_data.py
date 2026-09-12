"""Build the *textual* side of every dataset.

Each builder returns a tidy :class:`pandas.DataFrame` with the columns

    id      unique track/clip identifier (str)
    text    the natural-language music context fed to BERT
    labels  '|'-joined multi-label tag string
    group   grouping key used to prevent artist leakage across splits
    split   official split name when the dataset defines one, else ''

plus, where available, ``valence`` / ``arousal`` regression targets (DEAM).

CLI::

    python -m src.text_data --dataset musiccaps
    python -m src.text_data --dataset all
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, path, out_dir
from .utils import LOG, clean_text, humanize, join_labels, save_json, split_labels

DATASETS = ["musiccaps", "mtat", "fma", "deam"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _top_k_labels(label_lists: list[list[str]], k: int) -> list[str]:
    counts = Counter(t for row in label_lists for t in row)
    return [t for t, _ in counts.most_common(k)]


def _restrict(label_lists: list[list[str]], vocab: list[str]) -> list[list[str]]:
    keep = set(vocab)
    return [[t for t in row if t in keep] for row in label_lists]


def labels_to_matrix(label_col: pd.Series, vocab: list[str]) -> np.ndarray:
    """'|'-joined label strings -> binary indicator matrix [N, K]."""
    index = {t: i for i, t in enumerate(vocab)}
    Y = np.zeros((len(label_col), len(vocab)), dtype=np.float32)
    for r, s in enumerate(label_col.tolist()):
        for t in split_labels(s):
            j = index.get(t)
            if j is not None:
                Y[r, j] = 1.0
    return Y


# --------------------------------------------------------------------------- #
# 1. MusicCaps -- expert captions -> tag proxy task
# --------------------------------------------------------------------------- #


def build_musiccaps(top_k: int = 50, label_source: str = "aspects") -> pd.DataFrame:
    """MusicCaps caption -> tag proxy task (Task 1 deliverable in the brief).

    ``label_source``
        ``aspects``  -- top-K phrases from the human ``aspect_list``.
                        NOTE: aspect phrases are usually *quoted verbatim* in
                        the caption, so this is an easy proxy task by design.
        ``audioset`` -- AudioSet ontology ids, which are **not** contained in
                        the caption text; a genuinely harder variant.
    """
    df = pd.read_csv(path("musiccaps_csv"))
    LOG.info("musiccaps: %d rows", len(df))

    if label_source == "aspects":
        raw = [
            [str(a).strip().lower() for a in ast.literal_eval(s)] if isinstance(s, str) else []
            for s in df["aspect_list"]
        ]
    elif label_source == "audioset":
        raw = [
            [t.strip() for t in str(s).split(",") if t.strip()] if isinstance(s, str) else []
            for s in df["audioset_positive_labels"]
        ]
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    vocab = _top_k_labels(raw, top_k)
    raw = _restrict(raw, vocab)

    out = pd.DataFrame(
        {
            "id": df["ytid"].astype(str),
            "text": df["caption"].map(clean_text),
            "labels": [join_labels(r) for r in raw],
            "group": df["ytid"].astype(str),  # one clip per video
            "split": "",
            "start_s": df["start_s"],
            "end_s": df["end_s"],
        }
    )
    out = out[out["labels"] != ""].reset_index(drop=True)
    out.attrs["vocab"] = vocab
    return out


# --------------------------------------------------------------------------- #
# 2. MagnaTagATune -- 188 tags -> top-50 after synonym merging
# --------------------------------------------------------------------------- #

# Standard MTAT tag-synonym merges used in the MIR literature.
MTAT_SYNONYMS = {
    "beat": "beats",
    "chant": "chanting",
    "choral": "choir", "chorus": "choir", "choir": "choir",
    "clasical": "classical", "classic": "classical",
    "drum": "drums",
    "electro": "electronic", "electronica": "electronic",
    "fast beat": "fast", "quick": "fast",
    "female singer": "female", "female singing": "female", "female vocal": "female",
    "female vocals": "female", "female voice": "female", "woman": "female",
    "woman singing": "female", "women": "female", "girl": "female",
    "flutes": "flute",
    "guitars": "guitar",
    "hard rock": "hard",
    "harpsicord": "harpsichord",
    "heavy metal": "heavy", "metal": "heavy",
    "horn": "horns",
    "india": "indian",
    "jazzy": "jazz",
    "male singer": "male", "male vocal": "male", "male vocals": "male",
    "male voice": "male", "man": "male", "man singing": "male", "men": "male",
    "no drums": "no beat",
    "no singer": "no vocals", "no singing": "no vocals", "no vocal": "no vocals",
    "no voice": "no vocals", "no voices": "no vocals", "instrumental": "no vocals",
    "operatic": "opera",
    "orchestral": "orchestra",
    "silence": "quiet",
    "singer": "vocals", "singing": "vocals", "vocal": "vocals",
    "voice": "vocals", "voices": "vocals",
    "strange": "weird",
    "string": "strings",
    "synthesizer": "synth",
    "violins": "violin",
    "space": "spacey",
    "harpsichord": "harpsichord",
}

# official MTAT directory split: 0-b train, c validation, d-f test
MTAT_SPLIT_BY_DIR = (
    {d: "train" for d in "0123456789ab"}
    | {"c": "val"}
    | {d: "test" for d in "def"}
)


def parse_mtat_path(mp3_path: str) -> dict[str, str]:
    """``f/artist-album-track-30-59.mp3`` -> artist / album / track / segment."""
    stem = Path(str(mp3_path)).stem
    parts = stem.split("-")
    if len(parts) >= 4:
        artist, album = parts[0], parts[1]
        track = "-".join(parts[2:-2])
        seg = f"{parts[-2]}-{parts[-1]}"
    else:  # pragma: no cover -- defensive
        artist, album, track, seg = parts[0], "", "-".join(parts[1:]), ""
    return {
        "artist": humanize(artist),
        "album": humanize(album),
        "track": humanize(track),
        "segment": seg,
    }


def mtat_text(meta: dict[str, str]) -> str:
    bits = [f"Artist: {meta['artist']}."]
    if meta["album"]:
        bits.append(f"Album: {meta['album']}.")
    if meta["track"]:
        bits.append(f"Track: {meta['track']}.")
    if meta["segment"]:
        bits.append(f"Excerpt {meta['segment']} seconds.")
    return " ".join(bits)


def build_mtat(top_k: int = 50) -> pd.DataFrame:
    df = pd.read_csv(path("mtat_annotations"), sep="\t")
    tag_cols = [c for c in df.columns if c not in ("clip_id", "mp3_path")]
    LOG.info("mtat: %d clips, %d raw tags", len(df), len(tag_cols))

    # merge synonyms into a canonical tag space
    merged: dict[str, np.ndarray] = {}
    for c in tag_cols:
        canon = MTAT_SYNONYMS.get(c, c)
        v = df[c].to_numpy(dtype=np.int8)
        merged[canon] = np.maximum(merged[canon], v) if canon in merged else v
    M = pd.DataFrame(merged)

    vocab = M.sum(0).sort_values(ascending=False).head(top_k).index.tolist()
    labels = [
        join_labels([t for t in vocab if row[t]])
        for _, row in M[vocab].iterrows()
    ]

    meta = [parse_mtat_path(p) for p in df["mp3_path"]]
    out = pd.DataFrame(
        {
            "id": df["clip_id"].astype(str),
            "text": [mtat_text(m) for m in meta],
            "labels": labels,
            "group": [m["artist"] for m in meta],
            "split": [MTAT_SPLIT_BY_DIR.get(str(p)[0], "train") for p in df["mp3_path"]],
            "mp3_path": df["mp3_path"].astype(str),
        }
    )
    out = out[out["labels"] != ""].reset_index(drop=True)
    out.attrs["vocab"] = vocab
    return out


# --------------------------------------------------------------------------- #
# 3. FMA -- track/artist/album metadata -> genre tags
# --------------------------------------------------------------------------- #


def load_fma_tracks(subset: str = "medium") -> pd.DataFrame:
    """Load ``tracks.csv`` (multi-index header) restricted to a subset."""
    t = pd.read_csv(path("fma_metadata") / "tracks.csv", index_col=0,
                    header=[0, 1], low_memory=False)
    order = ["small", "medium", "large"]
    keep = set(order[: order.index(subset) + 1])
    t = t[t[("set", "subset")].isin(keep)]
    LOG.info("fma: subset=%s -> %d tracks", subset, len(t))
    return t


def load_fma_genres() -> dict[int, str]:
    g = pd.read_csv(path("fma_metadata") / "genres.csv", index_col=0)
    return {int(i): str(r["title"]) for i, r in g.iterrows()}


def _parse_list(s) -> list:
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []


def fma_text(row: pd.Series) -> str:
    title = clean_text(row.get(("track", "title")))
    artist = clean_text(row.get(("artist", "name")))
    album = clean_text(row.get(("album", "title")))
    tags = _parse_list(row.get(("track", "tags"))) + _parse_list(row.get(("artist", "tags")))
    loc = clean_text(row.get(("artist", "location")))
    bits = []
    if title:
        bits.append(f"Track: {title}.")
    if artist:
        bits.append(f"Artist: {artist}.")
    if album:
        bits.append(f"Album: {album}.")
    if loc:
        bits.append(f"From {loc}.")
    if tags:
        bits.append("Tags: " + ", ".join(str(t) for t in tags[:12]) + ".")
    return " ".join(bits) if bits else "unknown track"


def build_fma(subset: str = "medium", top_k: int = 50,
              label_level: str = "all") -> pd.DataFrame:
    """``label_level='all'`` -> multi-label ``genres_all``;
    ``'top'`` -> single ``genre_top`` (still stored multi-label)."""
    t = load_fma_tracks(subset)
    gmap = load_fma_genres()

    if label_level == "top":
        raw = [[clean_text(v)] if isinstance(v, str) else [] for v in t[("track", "genre_top")]]
    else:
        raw = [
            [gmap[int(g)] for g in _parse_list(s) if int(g) in gmap]
            for s in t[("track", "genres_all")]
        ]

    vocab = _top_k_labels(raw, top_k)
    raw = _restrict(raw, vocab)

    split_map = {"training": "train", "validation": "val", "test": "test"}
    out = pd.DataFrame(
        {
            "id": t.index.astype(str),
            "text": [fma_text(r) for _, r in t.iterrows()],
            "labels": [join_labels(r) for r in raw],
            "group": [clean_text(v) or f"artist_{i}" for i, v in
                      zip(t.index, t[("artist", "name")])],
            "split": [split_map.get(str(s), "") for s in t[("set", "split")]],
            "genre_top": [clean_text(v) for v in t[("track", "genre_top")]],
            "subset": [str(s) for s in t[("set", "subset")]],
        }
    )
    out = out[out["labels"] != ""].reset_index(drop=True)
    out.attrs["vocab"] = vocab
    return out


# --------------------------------------------------------------------------- #
# 4. DEAM -- metadata text -> genre tags, plus valence/arousal targets
# --------------------------------------------------------------------------- #


def _strip(x) -> str:
    return clean_text(str(x).replace("\t", " ")) if pd.notna(x) else ""


def load_deam_annotations() -> pd.DataFrame:
    """Song-level averaged valence/arousal (1-9 scale) for all 1802 clips."""
    frames = []
    for key in ("deam_static_1_2000", "deam_static_2000_2058"):
        d = pd.read_csv(path(key))
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["song_id", "valence_mean", "valence_std",
                         "arousal_mean", "arousal_std"]])
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("song_id")
    ann["song_id"] = ann["song_id"].astype(int)
    return ann


def _read_ragged_csv(fp: Path) -> list[list[str]]:
    """The DEAM metadata CSVs have a variable number of trailing fields."""
    import csv

    with open(fp, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = [[c.strip().strip('"').replace("\t", " ").strip() for c in row]
                for row in csv.reader(fh)]
    return [r for r in rows[1:] if r and r[0].strip().isdigit()]


def load_deam_metadata() -> pd.DataFrame:
    """Union of the three DEAM metadata files -> artist/title/album/genre/tags.

    The files are ragged (2014 carries up to 100 trailing last.fm labels, 2015
    appends extra genre tokens), so they are parsed positionally rather than
    with ``pd.read_csv``.
    """
    mdir = path("deam_metadata_dir")
    rows: list[dict] = []

    # 2013: song_id, file_name, Artist, Song title, start, end, Genre
    for f in _read_ragged_csv(mdir / "metadata_2013.csv"):
        f = f + [""] * (7 - len(f))
        rows.append({"song_id": int(f[0]), "artist": clean_text(f[2]),
                     "title": clean_text(f[3]), "album": "",
                     "genre": clean_text(f[6]), "tags": ""})

    # 2014: Id, Artist, Album, Track, Genre, seg start, seg end, last.fm labels...
    for f in _read_ragged_csv(mdir / "metadata_2014.csv"):
        f = f + [""] * (8 - len(f))
        lastfm = [clean_text(t) for t in f[7:] if clean_text(t)]
        rows.append({"song_id": int(f[0]), "artist": clean_text(f[1]),
                     "title": clean_text(f[3]), "album": clean_text(f[2]),
                     "genre": clean_text(f[4]), "tags": ", ".join(lastfm[:15])})

    # 2015: id, Filename, title, artist, album, genre[, extra genre tokens...]
    for f in _read_ragged_csv(mdir / "metadata_2015.csv"):
        f = f + [""] * (6 - len(f))
        album = clean_text(f[4])
        genres = [clean_text(g) for g in f[5:] if clean_text(g)]
        rows.append({"song_id": int(f[0]), "artist": clean_text(f[3]),
                     "title": clean_text(f[2]),
                     "album": "" if album.lower() == "n/a" else album,
                     "genre": "-".join(genres), "tags": ""})

    return pd.DataFrame(rows).drop_duplicates("song_id")


# DEAM uses '-' as the genre separator, which collides with hyphenated genre
# names -- normalise the known compounds before splitting.
DEAM_GENRE_COMPOUNDS = {
    "hip-hop": "HipHop", "hip hop": "HipHop", "trip-hop": "TripHop",
    "r&b": "RnB", "soulrb": "SoulRnB", "rnb": "RnB",
    "singer/songwriter": "SingerSongwriter", "singer-songwriter": "SingerSongwriter",
    "drum & bass": "DrumAndBass", "drum and bass": "DrumAndBass",
    "new-age": "NewAge", "new age": "NewAge",
}


def _split_deam_genre(gen: str) -> list[str]:
    s = str(gen)
    for pat, repl in DEAM_GENRE_COMPOUNDS.items():
        s = s.replace(pat, repl).replace(pat.title(), repl).replace(pat.upper(), repl)
    s = s.replace("/", "-")
    return [g.strip().title() for g in s.split("-") if g.strip()]


def deam_text(r: pd.Series) -> str:
    """NOTE: the genre field is deliberately *excluded* -- it is the label."""
    bits = []
    if r["title"]:
        bits.append(f"Track: {r['title']}.")
    if r["artist"]:
        bits.append(f"Artist: {r['artist']}.")
    if r["album"]:
        bits.append(f"Album: {r['album']}.")
    if r["tags"]:
        bits.append(f"Listener tags: {r['tags']}.")
    return " ".join(bits) if bits else "unknown track"


def build_deam(top_k: int = 20) -> pd.DataFrame:
    ann = load_deam_annotations()
    meta = load_deam_metadata()
    df = meta.merge(ann, on="song_id", how="inner")
    LOG.info("deam: %d clips with metadata + annotations", len(df))

    raw = [_split_deam_genre(g) for g in df["genre"]]
    vocab = _top_k_labels(raw, top_k)
    raw = _restrict(raw, vocab)

    out = pd.DataFrame({
        "id": df["song_id"].astype(str),
        "text": [deam_text(r) for _, r in df.iterrows()],
        "labels": [join_labels(r) for r in raw],
        "group": [a or f"song_{i}" for a, i in zip(df["artist"], df["song_id"])],
        "split": "",
        # valence / arousal rescaled from the 1-9 annotation scale to [-1, 1]
        "valence": (df["valence_mean"].astype(float) - 5.0) / 4.0,
        "arousal": (df["arousal_mean"].astype(float) - 5.0) / 4.0,
    })
    out = out[out["labels"] != ""].reset_index(drop=True)
    out.attrs["vocab"] = vocab
    return out


# --------------------------------------------------------------------------- #
# dispatcher + persistence
# --------------------------------------------------------------------------- #

_BUILDERS = {
    "musiccaps": build_musiccaps,
    "mtat": build_mtat,
    "fma": build_fma,
    "deam": build_deam,
}


def build(dataset: str, **kwargs) -> pd.DataFrame:
    if dataset not in _BUILDERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(_BUILDERS)}")
    return _BUILDERS[dataset](**kwargs)


def _tag(dataset: str, kwargs: dict) -> str:
    if dataset == "fma":
        return f"fma_{kwargs.get('subset', 'medium')}"
    return dataset


def save(df: pd.DataFrame, dataset: str, kwargs: dict | None = None) -> Path:
    kwargs = kwargs or {}
    name = _tag(dataset, kwargs)
    d = out_dir("data", "processed", "text")
    csv = d / f"{name}.csv"
    df.to_csv(csv, index=False)
    save_json(
        {
            "dataset": name,
            "n_rows": int(len(df)),
            "vocab": df.attrs.get("vocab", []),
            "label_cardinality": float(
                np.mean([len(split_labels(s)) for s in df["labels"]])
            ),
            "builder_kwargs": kwargs,
        },
        d / f"{name}_vocab.json",
    )
    LOG.info("saved %s: %d rows, %d tags -> %s",
             name, len(df), len(df.attrs.get("vocab", [])), csv)
    return csv


def load(dataset_tag: str) -> tuple[pd.DataFrame, list[str]]:
    """Read back a saved text table plus its label vocabulary."""
    d = out_dir("data", "processed", "text")
    df = pd.read_csv(d / f"{dataset_tag}.csv", dtype={"id": str})
    df["labels"] = df["labels"].fillna("")
    df["text"] = df["text"].fillna("")
    meta = pd.read_json(d / f"{dataset_tag}_vocab.json", typ="series")
    return df, list(meta["vocab"])


def main() -> None:
    ap = argparse.ArgumentParser(description="build text/tag tables")
    ap.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    ap.add_argument("--top_k", type=int, default=None, help="size of the tag vocabulary")
    ap.add_argument("--fma_subset", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--fma_label_level", default="all", choices=["all", "top"])
    ap.add_argument("--musiccaps_labels", default="aspects", choices=["aspects", "audioset"])
    args = ap.parse_args()

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        kw: dict = {}
        if ds == "musiccaps":
            kw = {"top_k": args.top_k or CFG["task1"]["top_k_tags"],
                  "label_source": args.musiccaps_labels}
        elif ds == "mtat":
            kw = {"top_k": args.top_k or CFG["task1"]["top_k_tags"]}
        elif ds == "fma":
            kw = {"subset": args.fma_subset, "top_k": args.top_k or CFG["task1"]["top_k_tags"],
                  "label_level": args.fma_label_level}
        elif ds == "deam":
            kw = {"top_k": args.top_k or 20}
        df = build(ds, **kw)
        save(df, ds, kw)


if __name__ == "__main__":
    main()
