# app.py

import streamlit as st
import pandas as pd, numpy as np, re
import matplotlib.pyplot as plt, seaborn as sns
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
# UTILITIES
# ─────────────────────────────────────────────────────────────────────
def _value_decimals_for(lab: str) -> int:
    return 0 if lab in ["ferritin", "vitamin_d"] else 1

def _format_value(lab: str, v) -> str:
    if pd.isna(v):
        return ""
    d = _value_decimals_for(lab)
    return f"{v:.{d}f}"

def _date_candidates(lab_key: str, draw_tag: str):
    # handle both 't1.5' and normalized 't1_5' column names
    tag_dot = draw_tag.lower()
    tag_us  = tag_dot.replace(".", "_")
    cands = []
    for tag in (tag_dot, tag_us):
        cands += [
            f"{lab_key}_{tag}_date",
            f"{lab_key}_{tag}_dt",
            f"{lab_key}_{tag}_draw_date",
            f"{lab_key}_{tag}_collection_date",
            f"date_{tag}",
            f"draw_date_{tag}",
            f"{tag}_date",
            f"{tag}_draw_date",
        ]
    return cands

# ─────────────────────────────────────────────────────────────────────
# DATA TRANSFORM HELPERS
# ─────────────────────────────────────────────────────────────────────
def tidy_long(df, lab_key):
    """
    Returns long df with columns:
    ['athlete','sport','sex','draw','date', <lab_key>]
    """
    regex = re.compile(fr"^{lab_key}_t\d(?:\.\d)?$", re.I)
    lab_cols = [c for c in df.columns if regex.match(c)]
    if not lab_cols:
        return pd.DataFrame(columns=["athlete","sport","sex","draw","date",lab_key])

    parts = []
    for col in lab_cols:
        m = re.search(r"(t\d(?:\.\d)?)", col, re.I)
        if not m:
            continue
        draw_tag = m.group(1).lower()

        # find a date column for this draw (if any)
        date_col = None
        for cand in _date_candidates(lab_key, draw_tag):
            if cand in df.columns:
                date_col = cand
                break

        base = df[["athlete","sport","sex"]].copy()
        base[lab_key] = pd.to_numeric(df[col], errors="coerce")
        base["draw"] = draw_tag
        base["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
        parts.append(base)

    if not parts:
        return pd.DataFrame(columns=["athlete","sport","sex","draw","date",lab_key])

    long = pd.concat(parts, ignore_index=True)
    long["draw"] = long["draw"].astype("category")
    ordered = sorted(long["draw"].dropna().unique(), key=lambda s: float(str(s)[1:]))
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
def heatmap_status_with_values(df, lab):
    flag_piv = df.pivot_table(index="athlete", columns="draw", values="flag", aggfunc="first")
    val_piv  = df.pivot_table(index="athlete", columns="draw", values=lab,   aggfunc="first")
    date_piv = df.pivot_table(index="athlete", columns="draw", values="date", aggfunc="first")

    data = flag_piv.replace(FLAG_TO_INT).astype(float)
    mask = val_piv.isna()

    ann = []
    for r in val_piv.index:
        row = []
        for c in val_piv.columns:
            v = val_piv.loc[r, c]
            d = date_piv.loc[r, c] if c in date_piv.columns else pd.NaT
            if pd.isna(v):
                row.append("")
                continue
            txt = _format_value(lab, v)
            if pd.notna(d):
                try:
                    txt += f"\n{pd.to_datetime(d).strftime('%m/%d')}"
                except Exception:
                    pass
            row.append(txt)
        ann.append(row)

    fig, ax = plt.subplots(
        figsize=(max(6, len(val_piv.columns) + 2), max(4, 0.45 * len(val_piv.index)))
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
        annot=np.array(ann, dtype=object),
        fmt="",
        annot_kws={"fontsize": 9, "va": "center"}
    )
    ax.set_title(f"{lab.upper()} status (value & date)", pad=12)
    ax.set_xlabel("draw")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def box_by_team(df, lab, draw):
    slc = df[df["draw"] == draw]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=slc, x="sport", y=lab, ax=ax)
    sns.stripplot(data=slc, x="sport", y=lab, color="black", jitter=True, alpha=0.6, ax=ax)
    ax.set_title(f"{lab.upper()} distribution – {draw}")  # <-- fixed .upper()
    if lab in ["ferritin", "vitamin_d"]:
        ax.axhline(THRESHOLDS[lab]["red"], linestyle="--", linewidth=1)
        ax.axhline(THRESHOLDS[lab]["yellow_high"], linestyle="--", linewidth=1)
    else:
        ax.axhline(min(THRESHOLDS[lab]["female"]["red"], THRESHOLDS[lab]["male"]["red"]),
                   linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def athlete_trend(df, lab):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, g in df.groupby("athlete"):
        ax.plot(g["draw"].cat.codes, g[lab], marker="o", label=name, alpha=0.7)
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
    raw.columns = [re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_") for c in raw.columns]

    # Required-ish fields, with safe fallbacks
    if "athlete" not in raw.columns:
        if "first_name" in raw.columns and "last_name" in raw.columns:
            raw["athlete"] = (
                raw["first_name"].fillna("") + "_" + raw["last_name"].fillna("")
            ).str.strip().replace({"^_$|^$": pd.NA}, regex=True)
        else:
            raw["athlete"] = pd.NA
    missing = raw["athlete"].isna()
    raw.loc[missing, "athlete"] = "ath" + raw.index[missing].astype(str)

    if "sex" not in raw.columns:
        raw["sex"] = "female"
    raw["sex"] = raw["sex"].astype(str).str.lower().map(
        {"f": "female", "m": "male", "female": "female", "male": "male"}
    ).fillna("female")

    if "sport" not in raw.columns:
        raw["sport"] = "unspecified"

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

    athletes_all = sorted(labs_long[lab_sel]["athlete"].unique()) if not labs_long[lab_sel].empty else []
    default_index = 1 if athletes_all else 0
    athlete_sel = st.sidebar.selectbox("Select athlete", ["All"] + athletes_all, index=default_index)

    # ── INDIVIDUAL SNAPSHOT / TABLE
    if athlete_sel != "All":
        st.subheader(f"📋 {lab_sel.upper()} — Individual view: {athlete_sel}")
        df_ind = labs_long[lab_sel][labs_long[lab_sel]["athlete"] == athlete_sel].copy()
        if df_ind.empty:
            st.warning("No data for this athlete and lab.")
        else:
            df_show = df_ind.sort_values("draw")[["draw","date",lab_sel,"flag"]].copy()
            if pd.api.types.is_datetime64_any_dtype(df_show["date"]):
                df_show["date"] = df_show["date"].dt.strftime("%Y-%m-%d")
            df_show.rename(columns={lab_sel: "value"}, inplace=True)

            latest = df_show.dropna(subset=["value"]).tail(1)
            if not latest.empty:
                latest_val = latest["value"].iloc[0]
                latest_date = latest["date"].iloc[0]
                latest_flag = df_ind.dropna(subset=[lab_sel]).tail(1)["flag"].iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Latest value", _format_value(lab_sel, latest_val))
                col2.metric("Collection date", latest_date if isinstance(latest_date, str) else "-")
                col3.metric("Status", latest_flag.upper())

            st.dataframe(df_show, use_container_width=True)
            st.download_button(
                "⬇️ Download table (CSV)",
                df_show.to_csv(index=False).encode("utf-8"),
                file_name=f"{athlete_sel}_{lab_sel}_history.csv",
                mime="text/csv"
            )

            st.caption("Status by draw (value & date shown):")
            st.pyplot(heatmap_status_with_values(df_ind, lab_sel))

    # ── ROSTER HEATMAP
    st.subheader(f"{lab_sel.upper()} — Roster heatmap (values & dates visible)")
    if labs_long[lab_sel].empty:
        st.info("No data to display for this lab.")
    else:
        st.pyplot(heatmap_status_with_values(labs_long[lab_sel], lab_sel))

    # ── OPTIONAL: trends (safe-guarded)
    with st.expander("📈 Explore trends (optional)"):
        try:
            df_lab = labs_long[lab_sel]
            draws = list(df_lab["draw"].cat.categories) if "draw" in df_lab and hasattr(df_lab["draw"], "cat") else []
            if not draws:
                st.caption("No draws available for distribution view.")
            else:
                draw_sel = st.selectbox("Select draw", draws, index=len(draws)-1)
                st.pyplot(box_by_team(df_lab, lab_sel, draw_sel))

            st.pyplot(athlete_trend(df_lab if athlete_sel == "All" else df_lab[df_lab["athlete"] == athlete_sel], lab_sel))
        except Exception as e:
            st.caption("Trend view unavailable for this dataset.")
            st.exception(e)

if __name__ == "__main__":
    main()
