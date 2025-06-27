# app.py

import streamlit as st
import pandas as pd, numpy as np, re
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
LABS = ["ferritin", "vitamin_d", "hgb", "hct"]
PERCENT_DROP_YELLOW = 25

THRESHOLDS = {
    "ferritin":  {"red": 30, "yellow_low": 30, "yellow_high": 40},
    "vitamin_d": {"red": 40, "yellow_low": 40, "yellow_high": 50},
    "hgb": {
        "male":   {"red": 13.0, "yellow_delta": 0.5},
        "female": {"red": 12.0, "yellow_delta": 0.5},
    },
    "hct": {
        "male":   {"red": 38.6, "yellow_delta": 1.0},
        "female": {"red": 34.7, "yellow_delta": 1.0},
    },
}
COLOR_MAP   = {"green": "#7fd77f", "yellow": "#ffd966", "red": "#f45b69"}
FLAG_TO_INT = {"red": 0, "yellow": 1, "green": 2}
INT_TO_COLOR = [COLOR_MAP["red"], COLOR_MAP["yellow"], COLOR_MAP["green"]]

# ─────────────────────────────────────────────────────────────────────
# DATA TRANSFORM HELPERS
# ─────────────────────────────────────────────────────────────────────
def tidy_long(df, lab_key):
    regex = re.compile(fr"^{lab_key}_t\d(?:\.\d)?$", re.I)
    lab_cols = [c for c in df.columns if regex.match(c)]
    if not lab_cols:
        return pd.DataFrame(columns=["athlete","sport","sex","draw",lab_key])

    long = (
        df[["athlete","sport","sex"] + lab_cols]
          .melt(["athlete","sport","sex"], var_name="draw", value_name=lab_key)
          .dropna(subset=[lab_key])
    )
    long[lab_key] = pd.to_numeric(long[lab_key], errors="coerce")
    long["draw"] = (
        long["draw"]
            .str.extract(r"(t\d(?:\.\d)?)", expand=False)
            .astype("category")
    )
    ordered = sorted(long["draw"].dropna().unique(), key=lambda s: float(s[1:]))
    long["draw"] = long["draw"].cat.set_categories(ordered, ordered=True)
    return long


def flag_rows(df, lab):
    df = df.copy()
    if "flag" in df.columns:
        df = df.drop(columns=["flag"])

    df[lab] = pd.to_numeric(df[lab], errors="coerce")

    if lab in ["ferritin", "vitamin_d"]:
        t = THRESHOLDS[lab]
        df["flag"] = np.select(
            [df[lab] < t["red"],
             df[lab].between(t["yellow_low"], t["yellow_high"])],
            ["red", "yellow"],
            "green"
        )
        df["prev"] = df.groupby("athlete")[lab].shift(1)
        big_drop = (df["prev"] - df[lab]) / df["prev"] * 100 >= PERCENT_DROP_YELLOW
        df.loc[big_drop & (df["flag"] == "green"), "flag"] = "yellow"
        df = df.drop(columns=["prev"])
        return df

    red_cut = df["sex"].map({s: THRESHOLDS[lab][s]["red"] for s in THRESHOLDS[lab]})
    yellow_cut = df["sex"].map({
        s: THRESHOLDS[lab][s]["red"] + THRESHOLDS[lab][s]["yellow_delta"]
        for s in THRESHOLDS[lab]
    })
    df["flag"] = np.select(
        [df[lab] < red_cut,
         df[lab] < yellow_cut],
        ["red", "yellow"],
        "green"
    )
    return df

# ─────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────
def heatmap_status(df, lab):
    src = df.pivot_table(
        index="athlete", columns="draw", values="flag", aggfunc="first"
    )
    data = src.replace(FLAG_TO_INT).astype(float)
    mask = data.isna()

    fig, ax = plt.subplots(
        figsize=(data.shape[1] + 2, max(4, 0.4 * data.shape[0]))
    )
    sns.heatmap(
        data,
        cmap=ListedColormap(INT_TO_COLOR),
        vmin=0, vmax=2,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        square=True,
        mask=mask,
        ax=ax
    )
    ax.set_title(f"{lab.upper()} status", pad=12)
    ax.set_xlabel("draw")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def box_by_team(df, lab, draw):
    slc = df[df["draw"] == draw]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=slc, x="sport", y=lab, ax=ax)
    sns.stripplot(
        data=slc, x="sport", y=lab,
        color="black", jitter=True, alpha=0.6, ax=ax
    )
    ax.set_title(f"{lab.upper()} distribution – {draw}")
    ax.axhline(THRESHOLDS[lab]["red"], linestyle="--", linewidth=1)
    if lab in ["ferritin", "vitamin_d"]:
        ax.axhline(THRESHOLDS[lab]["yellow_high"], linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def athlete_trend(df, lab):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, g in df.groupby("athlete"):
        ax.plot(
            g["draw"].cat.codes,
            g[lab],
            marker="o",
            label=name,
            alpha=0.7
        )
    ax.set_xticks(range(len(df["draw"].cat.categories)))
    ax.set_xticklabels(df["draw"].cat.categories)
    ax.set_ylabel(lab)
    ax.set_title(f"{lab.upper()} trend")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Athlete-Lab Dashboard", layout="wide")
    st.title("🏃 Athlete-Lab Dashboard")

    uploaded = st.sidebar.file_uploader(
        "Upload cleaned CSV or Excel", type=["csv", "xls", "xlsx"]
    )
    if not uploaded:
        st.info("Please upload your lab data to begin.")
        return

    # READ
    try:
        if uploaded.name.lower().endswith((".xls", ".xlsx")):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # NORMALIZE COLUMNS
    raw.columns = [
        re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_")
        for c in raw.columns
    ]

    # ATHLETE ID & SEX
    raw["athlete"] = (
        raw["first_name"].fillna("") + "_" + raw["last_name"].fillna("")
    ).str.strip().replace({"^_$|^$": pd.NA}, regex=True)
    missing = raw["athlete"].isna()
    raw.loc[missing, "athlete"] = "ath" + raw.index[missing].astype(str)

    if "sex" not in raw.columns:
        raw["sex"] = "female"
    raw["sex"] = raw["sex"].str.lower()

    # SPLIT CBC if present
    if "cbc" in raw.columns:
        m = raw["cbc"].astype(str).str.extract(r"([\d.]+)\s*\(?([\d.]*)\)?")
        raw["hgb"] = pd.to_numeric(m[0], errors="coerce")
        raw["hct"] = pd.to_numeric(m[1], errors="coerce")

    # TIDY + FLAG
    long_frames = {lab: tidy_long(raw, lab) for lab in LABS}
    labs_long   = {lab: flag_rows(long_frames[lab], lab) for lab in LABS}

    # SIDEBAR CONTROLS
    lab_sel = st.sidebar.selectbox("Select lab", LABS)
    draws   = list(labs_long[lab_sel]["draw"].cat.categories)
    draw_sel = st.sidebar.selectbox("Select draw", draws)
    athletes = ["All"] + sorted(labs_long[lab_sel]["athlete"].unique())
    athlete_sel = st.sidebar.selectbox("Select athlete", athletes)

    # PLOTS
    st.subheader(f"{lab_sel.upper()} Status Heatmap")
    st.pyplot(heatmap_status(labs_long[lab_sel], lab_sel))

    st.subheader(f"{lab_sel.upper()} Distribution – {draw_sel}")
    st.pyplot(box_by_team(labs_long[lab_sel], lab_sel, draw_sel))

    st.subheader(f"{lab_sel.upper()} Trend – {athlete_sel}")
    if athlete_sel == "All":
        df_trend = labs_long[lab_sel]
    else:
        df_trend = labs_long[lab_sel][labs_long[lab_sel]["athlete"] == athlete_sel]
    st.pyplot(athlete_trend(df_trend, lab_sel))


if __name__ == "__main__":
    main()
