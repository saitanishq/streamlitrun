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
COLOR_MAP    = {"green": "#7fd77f", "yellow": "#ffd966", "red": "#f45b69"}
FLAG_TO_INT  = {"red": 0, "yellow": 1, "green": 2}
INT_TO_COLOR = [COLOR_MAP["red"], COLOR_MAP["yellow"], COLOR_MAP["green"]]

# ─────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────
def normalize_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_") for c in df.columns]
    return df

def parse_first_float(x):
    """Return the first float found in a string like '60.4 (was 25.7)', otherwise NaN."""
    if pd.isna(x):
        return np.nan
    m = re.search(r"([0-9]*\.?[0-9]+)", str(x))
    return float(m.group(1)) if m else np.nan

def parse_cbc_pair(x):
    """Parse CBC strings like '14.2/42.9' or '13.6 (40.2)' -> (hgb, hct)."""
    if pd.isna(x):
        return (np.nan, np.nan)
    s = str(x)
    m = re.search(r"^\s*([0-9]*\.?[0-9]+)\s*(?:/|\(|,)?\s*([0-9]*\.?[0-9]+)?", s)
    if not m:
        return (np.nan, np.nan)
    hgb = float(m.group(1)) if m.group(1) else np.nan
    hct = float(m.group(2)) if m.group(2) else np.nan
    return (hgb, hct)

def build_athlete(df):
    if "athlete" in df.columns:
        a = df["athlete"].astype(str)
    elif {"first_name","last_name"}.issubset(df.columns):
        a = (df["first_name"].fillna("") + "_" + df["last_name"].fillna("")).str.strip("_")
    elif "last_name" in df.columns:
        a = df["last_name"].astype(str)
    else:
        a = pd.Series([f"ath{i}" for i in range(len(df))])
    a = a.replace({"^$": np.nan, "^_$": np.nan}, regex=True)
    a = a.fillna(pd.Series([f"ath{i}" for i in range(len(df))]))
    return a

def build_sex(df):
    if "sex" in df.columns:
        s = df["sex"]
    elif "gender" in df.columns:
        s = df["gender"]
    else:
        s = "female"
    s = pd.Series(s).astype(str).str.lower().map(
        {"f":"female","female":"female","m":"male","male":"male"}
    ).fillna("female")
    return s

def value_decimals_for(lab: str) -> int:
    return 0 if lab in ["ferritin", "vitamin_d"] else 1

def fmt_value(lab: str, v) -> str:
    if pd.isna(v):
        return ""
    d = value_decimals_for(lab)
    return f"{v:.{d}f}"

# ─────────────────────────────────────────────────────────────────────
# LONG-FORM & FLAGGING
# ─────────────────────────────────────────────────────────────────────
def date_candidates(lab_key: str, draw_tag: str):
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

def tidy_from_wide(df, lab):
    """
    Build long-form for a lab.
    Accepts:
      - {lab} (single reading) → draw = t1
      - {lab}_t0, {lab}_t1, {lab}_t2_1 → draw = t0, t1, t2.1
    Captures optional per-draw date columns if present.
    """
    cols = [c for c in df.columns if c == lab or c.startswith(f"{lab}_t")]
    if not cols:
        return pd.DataFrame(columns=["athlete","sport","sex","draw","date",lab])

    parts = []
    for c in cols:
        m = re.search(r"_(t[0-9]+(?:_[0-9]+)?)$", c)
        if m:
            draw = m.group(1).lower().replace("_", ".")  # t2_1 → t2.1
        else:
            draw = "t1"  # single column case (e.g., vitamin_d)

        # date column lookup (optional)
        date_col = None
        for cand in date_candidates(lab, draw):
            if cand in df.columns:
                date_col = cand
                break

        series = pd.to_numeric(df[c].apply(parse_first_float), errors="coerce")

        base = pd.DataFrame({
            "athlete": df["athlete"],
            "sport": df.get("sport", pd.Series(["unspecified"]*len(df))),
            "sex": df["sex"],
            "draw": draw,
            lab: series,
            "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        })
        parts.append(base)

    long = pd.concat(parts, ignore_index=True)
    long["draw"] = pd.Categorical(long["draw"])
    ordered = sorted(long["draw"].dropna().unique(), key=lambda s: float(str(s)[1:]))
    long["draw"] = long["draw"].cat.set_categories(ordered, ordered=True)
    return long

def flag_df(df, lab):
    df = df.copy()
    if df.empty:
        return df.assign(flag=pd.Series(dtype="object"))
    if lab in ["ferritin","vitamin_d"]:
        t = THRESHOLDS[lab]
        df["flag"] = np.select(
            [df[lab] < t["red"],
             df[lab].between(t["yellow_low"], t["yellow_high"])],
            ["red", "yellow"],
            "green"
        )
        # Yellow if ≥ PERCENT_DROP_YELLOW from previous draw (even if green)
        df = df.sort_values(["athlete","draw"])
        df["prev"] = df.groupby("athlete")[lab].shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_drop = (df["prev"] - df[lab]) / df["prev"] * 100.0
        big_drop = (pct_drop >= PERCENT_DROP_YELLOW)
        df.loc[big_drop & (df["flag"] == "green"), "flag"] = "yellow"
        df.drop(columns=["prev"], inplace=True)
    else:
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
# VISUALS
# ─────────────────────────────────────────────────────────────────────
def heatmap_status_with_values(df, lab):
    # pivot by flag (for colors), values (for annot), and date (for annot)
    flag_piv = df.pivot_table(index="athlete", columns="draw", values="flag", aggfunc="first")
    val_piv  = df.pivot_table(index="athlete", columns="draw", values=lab,   aggfunc="first")
    date_piv = df.pivot_table(index="athlete", columns="draw", values="date", aggfunc="first")

    data = flag_piv.replace(FLAG_TO_INT).astype(float)
    mask = val_piv.isna()

    # annotation text: value + (optional) date
    ann = []
    for r in val_piv.index:
        row = []
        for c in val_piv.columns:
            v = val_piv.loc[r, c]
            d = date_piv.loc[r, c] if c in date_piv.columns else pd.NaT
            if pd.isna(v):
                row.append("")
                continue
            txt = fmt_value(lab, v)
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

# ─────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Athlete-Lab Dashboard", layout="wide")
    st.title("🏃 Athlete-Lab Dashboard")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xls", "xlsx"]
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

    # PREPROCESS
    raw = normalize_cols(raw)
    raw["athlete"] = build_athlete(raw)
    raw["sex"] = build_sex(raw)
    if "sport" not in raw.columns:
        raw["sport"] = "unspecified"

    # Expand CBC → hgb/hct
    if "cbc" in raw.columns:
        hgb_hct = raw["cbc"].apply(parse_cbc_pair).tolist()
        raw["hgb"] = [p[0] for p in hgb_hct]
        raw["hct"] = [p[1] for p in hgb_hct]

    # Long-form + flags per lab
    long_frames = {lab: tidy_from_wide(raw, lab) for lab in LABS}
    labs_long   = {lab: flag_df(long_frames[lab], lab) for lab in LABS}

    # Sidebar controls
    lab_sel = st.sidebar.selectbox("Select lab", LABS)

    # Quick filter: show only red/yellow rows in tables & heatmap mask
    show_only_at_risk = st.sidebar.checkbox("Show only at-risk (red/yellow)", value=False)

    # Build latest snapshot table (at-a-glance)
    def latest_per_athlete(flagged_lab, lab):
        df = flagged_lab
        if df.empty:
            return pd.DataFrame(columns=["athlete","lab","draw","value","flag"])
        df2 = df.dropna(subset=[lab]).sort_values(["athlete","draw"])
        last = df2.groupby("athlete").tail(1)
        out = last[["athlete","draw",lab,"flag"]].rename(columns={lab:"value"}).copy()
        out["lab"] = lab
        return out[["athlete","lab","draw","value","flag"]]

    snapshots = pd.concat(
        [latest_per_athlete(labs_long[l], l) for l in LABS],
        ignore_index=True
    )
    if show_only_at_risk and not snapshots.empty:
        snapshots = snapshots[snapshots["flag"].isin(["red","yellow"])]

    st.subheader("🧭 At-a-glance (latest per athlete × lab)")
    if snapshots.empty:
        st.info("No measurements available.")
    else:
        # sort by severity (red → yellow → green) then by lab
        sev_rank = {"red":0,"yellow":1,"green":2}
        snapshots = snapshots.assign(_sev=snapshots["flag"].map(sev_rank))
        snapshots = snapshots.sort_values(["_sev","lab","athlete"]).drop(columns=["_sev"])
        st.dataframe(snapshots, use_container_width=True)
        st.download_button(
            "⬇️ Download latest snapshot (CSV)",
            snapshots.to_csv(index=False).encode("utf-8"),
            file_name="latest_snapshot.csv",
            mime="text/csv",
        )

    # Roster heatmap (values visible; dates if provided in file)
    st.subheader(f"{lab_sel.upper()} — Roster heatmap (values & dates visible)")
    df_lab = labs_long[lab_sel]
    if df_lab.empty:
        st.info(f"No data to display for {lab_sel}.")
    else:
        if show_only_at_risk:
            df_lab = df_lab[df_lab["flag"].isin(["red","yellow"])].copy()
            if df_lab.empty:
                st.info("No red/yellow rows for this lab.")
            else:
                st.pyplot(heatmap_status_with_values(df_lab, lab_sel))
        else:
            st.pyplot(heatmap_status_with_values(df_lab, lab_sel))

    # Individual view for quick review
    athletes_all = sorted(df_lab["athlete"].unique()) if not df_lab.empty else []
    if athletes_all:
        athlete_sel = st.sidebar.selectbox("Focus athlete (optional)", ["All"] + athletes_all, index=0)
        if athlete_sel != "All":
            st.subheader(f"📋 {lab_sel.upper()} — Individual view: {athlete_sel}")
            df_ind = df_lab[df_lab["athlete"] == athlete_sel].copy()
            df_show = df_ind.sort_values("draw")[["draw","date",lab_sel,"flag"]].copy()
            if pd.api.types.is_datetime64_any_dtype(df_show["date"]):
                df_show["date"] = df_show["date"].dt.strftime("%Y-%m-%d")
            df_show.rename(columns={lab_sel: "value"}, inplace=True)
            st.dataframe(df_show, use_container_width=True)
            st.download_button(
                "⬇️ Download this athlete’s history (CSV)",
                df_show.to_csv(index=False).encode("utf-8"),
                file_name=f"{athlete_sel}_{lab_sel}_history.csv",
                mime="text/csv"
            )

    # Optional: trends
    with st.expander("📈 Explore trends (optional)"):
        try:
            if not df_lab.empty and hasattr(df_lab["draw"], "cat"):
                fig, ax = plt.subplots(figsize=(10, 5))
                for name, g in df_lab.groupby("athlete"):
                    ax.plot(g["draw"].cat.codes, g[lab_sel], marker="o", label=name, alpha=0.7)
                ax.set_xticks(range(len(df_lab["draw"].cat.categories)))
                ax.set_xticklabels(df_lab["draw"].cat.categories)
                ax.set_ylabel(lab_sel)
                ax.set_title(f"{lab_sel.upper()} trend")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
                st.pyplot(fig)
            else:
                st.caption("No draws available for trend view.")
        except Exception as e:
            st.caption("Trend view unavailable for this dataset.")

if __name__ == "__main__":
    main()
