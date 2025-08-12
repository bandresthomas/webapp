import os
import warnings
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# -----------------------------
# Constants and configuration
# -----------------------------
SEASON: int = 2024
WEEKS: List[int] = list(range(1, 19))  # NFL has 18 weeks
PRIMARY_COLOR: str = "#22c55e"  # Tailwind green-500
SECONDARY_COLOR: str = "#6b7280"  # Gray-500


def _safe_get(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> str:
    """Return the first existing column name from candidates; if none found, raise unless default.

    This avoids breaking if nfl_data_py changes column names slightly between versions.
    """
    for col in candidates:
        if col in df.columns:
            return col
    if default is not None:
        return default
    raise KeyError(f"None of the expected columns exist: {candidates}")


@st.cache_data(show_spinner=True)
def load_schedules(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    schedules = nfl.import_schedules(years=[season])
    # Build team-week to opponent map (including both home and away perspectives)
    home_col = _safe_get(schedules, ["home_team"])  # schedule has stable names
    away_col = _safe_get(schedules, ["away_team"])  # schedule has stable names
    week_col = _safe_get(schedules, ["week"]) 

    home_map = schedules[[week_col, home_col, away_col]].rename(
        columns={week_col: "week", home_col: "team", away_col: "opponent"}
    )
    away_map = schedules[[week_col, home_col, away_col]].rename(
        columns={week_col: "week", away_col: "team", home_col: "opponent"}
    )
    both = pd.concat([home_map, away_map], ignore_index=True)
    both["season"] = season

    # Ensure one row per team-week
    both = both.drop_duplicates(subset=["season", "week", "team"])  # should already be unique

    # Add BYE rows for team-weeks missing in schedule
    teams = sorted(set(both["team"].unique()))
    full = (
        pd.MultiIndex.from_product([[season], WEEKS, teams], names=["season", "week", "team"])  # type: ignore[arg-type]
        .to_frame(index=False)
    )
    full = full.merge(both, on=["season", "week", "team"], how="left")
    full["opponent"] = full["opponent"].fillna("BYE")
    return full


@st.cache_data(show_spinner=True)
def load_weekly_and_snaps(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import nfl_data_py as nfl

    weekly_cols = None  # allow package default; we'll select after
    weekly = nfl.import_weekly_data(years=[season], columns=weekly_cols, downcast=True)
    # Normalize common column names
    player_col = _safe_get(weekly, ["player_display_name", "player_name", "player"], default="player")
    team_col = _safe_get(weekly, ["recent_team", "team", "club"], default="team")
    pos_col = _safe_get(weekly, ["position", "pos"], default="position")
    week_col = _safe_get(weekly, ["week"]) 
    season_col = _safe_get(weekly, ["season"]) 
    ppr_col = _safe_get(weekly, ["fantasy_points_ppr", "ppr_points", "ppr"], default=None)

    # Keep only relevant columns
    keep = [player_col, team_col, pos_col, week_col, season_col]
    if ppr_col is not None and ppr_col in weekly.columns:
        keep.append(ppr_col)
    weekly = weekly[keep].copy()
    weekly.rename(
        columns={
            player_col: "player_name",
            team_col: "team",
            pos_col: "position",
            week_col: "week",
            season_col: "season",
            ppr_col: "ppr_points" if ppr_col in weekly.columns else None,
        },
        inplace=True,
    )
    if "ppr_points" not in weekly.columns:
        # If the installed dataset doesn't expose PPR directly, derive a conservative fallback of zeros.
        # Users can still explore snap counts and opponents.
        weekly["ppr_points"] = 0.0

    # Snap counts
    snaps = nfl.import_snap_counts(years=[season])
    # Normalize
    s_player_col = _safe_get(snaps, ["player", "player_display_name", "player_name"], default="player")
    s_team_col = _safe_get(snaps, ["team", "recent_team", "club"], default="team")
    s_week_col = _safe_get(snaps, ["week"]) 
    s_season_col = _safe_get(snaps, ["season"]) 

    # Find an offense snaps column variant
    offense_candidates = [
        "offense",  # common in nflverse snap counts
        "offense_snaps",
        "off_snaps",
    ]
    offense_col = None
    for c in offense_candidates:
        if c in snaps.columns:
            offense_col = c
            break
    if offense_col is None:
        # Fallback: if a percentage column exists, use it as proxy scaled by 100
        for c in ["off_pct", "offense_pct"]:
            if c in snaps.columns:
                offense_col = c
                break
    if offense_col is None:
        snaps["snap_counts"] = 0
    else:
        snaps["snap_counts"] = pd.to_numeric(snaps[offense_col], errors="coerce").fillna(0).astype(float)

    snaps = snaps[[s_player_col, s_team_col, s_week_col, s_season_col, "snap_counts"]].rename(
        columns={
            s_player_col: "player_name",
            s_team_col: "team",
            s_week_col: "week",
            s_season_col: "season",
        }
    )

    # Ensure types
    weekly["week"] = weekly["week"].astype(int)
    weekly["season"] = weekly["season"].astype(int)
    snaps["week"] = snaps["week"].astype(int)
    snaps["season"] = snaps["season"].astype(int)
    return weekly, snaps


@st.cache_data(show_spinner=True)
def load_rosters(season: int) -> pd.DataFrame:
    """Optional enrichment for the Player card (height, weight, age, experience)."""
    import nfl_data_py as nfl

    try:
        rosters = nfl.import_seasonal_rosters(years=[season])
        # Normalize names
        r_player_col = _safe_get(rosters, ["player_display_name", "player_name", "player"], default="player")
        r_team_col = _safe_get(rosters, ["team", "recent_team", "club"], default="team")
        meta_cols = [
            r_player_col,
            r_team_col,
        ]
        for c in ["height", "weight", "age", "experience", "position"]:
            if c in rosters.columns:
                meta_cols.append(c)
        rosters = rosters[meta_cols].rename(
            columns={r_player_col: "player_name", r_team_col: "team"}
        )
        return rosters
    except Exception:
        return pd.DataFrame(columns=["player_name", "team", "height", "weight", "age", "experience", "position"])


def prepare_player_week_grid(
    player_name: str,
    team: str,
    weekly: pd.DataFrame,
    snaps: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Create a complete 18-week grid for a selected player and team, filling gaps with 0 and BYE as required."""
    sched_team = schedules[schedules["team"] == team][["week", "opponent"]].copy()
    # player rows for that team (in case of traded players, apply team filter)
    pw = weekly[(weekly["player_name"] == player_name) & (weekly["team"] == team) & (weekly["season"] == SEASON)][["week", "ppr_points"]]
    ps = snaps[(snaps["player_name"] == player_name) & (snaps["team"] == team) & (snaps["season"] == SEASON)][["week", "snap_counts"]]

    base = pd.DataFrame({"week": WEEKS}).merge(sched_team, on="week", how="left")
    base = base.merge(pw, on="week", how="left").merge(ps, on="week", how="left")
    base["opponent"].fillna("BYE", inplace=True)
    base["ppr_points"] = pd.to_numeric(base["ppr_points"], errors="coerce").fillna(0.0)
    base["snap_counts"] = pd.to_numeric(base["snap_counts"], errors="coerce").fillna(0.0)
    return base


def plot_dual_axis(df: pd.DataFrame, player_name: str, team: str) -> go.Figure:
    fig = go.Figure()
    # Bars: PPR points (primary y)
    fig.add_trace(
        go.Bar(
            x=df["week"],
            y=df["ppr_points"],
            name="PPR Points",
            marker_color=PRIMARY_COLOR,
            hovertemplate="Week %{x}<br>PPR: %{y:.2f}<br>Opp: %{customdata}",
            customdata=df["opponent"],
        )
    )
    # Line: Snap counts (secondary y)
    fig.add_trace(
        go.Scatter(
            x=df["week"],
            y=df["snap_counts"],
            name="Snap Count",
            mode="lines+markers",
            marker=dict(size=8, color="#3b82f6"),
            line=dict(width=3, color="#3b82f6"),
            yaxis="y2",
            hovertemplate="Week %{x}<br>Snaps: %{y:.0f}<br>Opp: %{customdata}",
            customdata=df["opponent"],
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            title="Week",
            dtick=1,
            tick0=1,
            range=[0.5, 18.5],
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            title="PPR Points",
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis2=dict(
            title="Snap Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white",
    )

    # Annotations for BYE weeks
    for _, row in df.iterrows():
        if str(row["opponent"]).upper() == "BYE":
            fig.add_vrect(
                x0=row["week"] - 0.5,
                x1=row["week"] + 0.5,
                fillcolor="#f3f4f6",
                opacity=0.4,
                line_width=0,
            )

    fig.update_layout(title=f"{player_name} — {team} ({SEASON})")
    return fig


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* App-wide theming to resemble the sample UI */
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.95)), url('https://images.unsplash.com/photo-1502877338535-766e1452684a?q=80&w=2000&auto=format&fit=crop');
            background-size: cover;
        }}
        .metric-card {{
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            padding: 16px 16px 4px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .accent {{ border-top: 3px solid {PRIMARY_COLOR}; }}
        .pill {{
            display: inline-block;
            border: 1px solid #e5e7eb;
            padding: 4px 10px;
            border-radius: 999px;
            margin-right: 6px;
            background: #fff;
            color: #111827;
            font-size: 0.85rem;
        }}
        .header-title {{ font-weight: 700; font-size: 1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="NFL 2024 — PPR & Snaps", layout="wide")
    inject_css()

    st.sidebar.markdown("<div class='header-title'>NFL Dashboard</div>", unsafe_allow_html=True)
    st.sidebar.caption("Fonte de dados: nfl_data_py via nflverse")

    # Data loading
    schedules = load_schedules(SEASON)
    weekly, snaps = load_weekly_and_snaps(SEASON)
    rosters = load_rosters(SEASON)

    # Filters
    all_positions = sorted([p for p in weekly["position"].dropna().unique().tolist() if p])
    position = st.sidebar.selectbox("Posição", options=["QB", "RB", "WR", "TE"] if {"QB","RB","WR","TE"}.issubset(set(all_positions)) else all_positions)

    # Teams that had at least one player with this position
    teams_for_pos = (
        weekly[weekly["position"] == position]["team"].dropna().unique().tolist()
    )
    teams_for_pos = sorted(teams_for_pos)
    team = st.sidebar.selectbox("Time", options=teams_for_pos, index=0 if teams_for_pos else None)

    # Opponent filter list from schedule
    opponents = sorted(schedules[schedules["team"] == team]["opponent"].unique().tolist()) if team else []
    opponent_filter = st.sidebar.multiselect("Oponente", options=opponents, default=[])

    # Player list for position+team
    players_df = weekly[(weekly["position"] == position) & (weekly["team"] == team)]
    player_names = sorted(players_df["player_name"].dropna().unique().tolist())
    default_player = player_names[0] if player_names else ""
    player_name = st.sidebar.selectbox("Jogador", options=player_names, index=0 if player_names else None)

    # Main layout
    left, right = st.columns([3, 1])
    with left:
        st.markdown("<div class='header-title'>Player</div>", unsafe_allow_html=True)
        if not player_name:
            st.info("Selecione posição, time e jogador na barra lateral.")
            st.stop()

        grid = prepare_player_week_grid(player_name, team, weekly, snaps, schedules)
        if opponent_filter:
            grid = grid[grid["opponent"].isin(opponent_filter)]
            # Still show all weeks for context; fill others as nan -> then fill
            grid = (
                pd.DataFrame({"week": WEEKS}).merge(grid, on="week", how="left")
            )
            grid["opponent"] = grid["opponent"].fillna("")
            grid["ppr_points"] = grid["ppr_points"].fillna(0)
            grid["snap_counts"] = grid["snap_counts"].fillna(0)

        fig = plot_dual_axis(grid, player_name, team)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Opponent table for quick reference
        show = grid.copy()
        show["PPR"] = show["ppr_points"].round(2)
        show["Snaps"] = show["snap_counts"].astype(int)
        show = show[["week", "opponent", "PPR", "Snaps"]].rename(columns={"week": "Semana", "opponent": "Oponente"})
        st.dataframe(show, hide_index=True, use_container_width=True)

    with right:
        st.markdown("<div class='header-title'>Card</div>", unsafe_allow_html=True)
        card = st.container()
        with card:
            st.markdown("<div class='metric-card accent'>", unsafe_allow_html=True)
            st.subheader(player_name)
            meta = rosters[(rosters["player_name"] == player_name) & (rosters["team"] == team)]
            if not meta.empty:
                cols = st.columns(2)
                mrow = meta.iloc[0].to_dict()
                cols[0].metric("Posição", mrow.get("position", position))
                cols[1].metric("Time", team)

                cols2 = st.columns(2)
                if "height" in meta.columns:
                    cols2[0].metric("Altura", str(mrow.get("height", "-")))
                if "weight" in meta.columns:
                    cols2[1].metric("Peso", str(mrow.get("weight", "-")))
                cols3 = st.columns(2)
                if "age" in meta.columns:
                    cols3[0].metric("Idade", str(mrow.get("age", "-")))
                if "experience" in meta.columns:
                    cols3[1].metric("Exp.", str(mrow.get("experience", "-")))
            else:
                st.write(f"Time: {team} — Posição: {position}")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


