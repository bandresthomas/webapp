import os
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Constants and configuration
# -----------------------------
DEFAULT_SEASON: int = 2024
TABLES_DIR_NAME: str = "tables"


def _repo_root_dir() -> str:
    # Page files live under `pages/`; we want repo root to access `tables/`
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _tables_dir() -> str:
    return os.path.join(_repo_root_dir(), TABLES_DIR_NAME)


@st.cache_data(show_spinner=False)
def _load_pfr_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _tables_dir()
    passing_path = os.path.join(base, "pfr_passing_2022_2024.csv")
    receiving_path = os.path.join(base, "pfr_receiving_2022_2024.csv")
    rushing_path = os.path.join(base, "pfr_rushing_2022_2024.csv")

    # Load CSVs; rely on pandas type inference for simplicity
    passing = pd.read_csv(passing_path)
    receiving = pd.read_csv(receiving_path)
    rushing = pd.read_csv(rushing_path)

    # Normalize common columns
    # Passing lacks explicit position; assign QB for clarity
    passing = passing.rename(
        columns={
            "player": "player_name",
            "team": "team",
            "season": "season",
        }
    ).copy()
    passing["position"] = "QB"

    receiving = receiving.rename(
        columns={
            "player": "player_name",
            "tm": "team",
            "season": "season",
            "pos": "position",
        }
    ).copy()

    rushing = rushing.rename(
        columns={
            "player": "player_name",
            "tm": "team",
            "season": "season",
            "pos": "position",
        }
    ).copy()

    # Ensure expected dtypes for filters
    for df in (passing, receiving, rushing):
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        if "team" in df.columns:
            df["team"] = df["team"].astype(str)
        if "player_name" in df.columns:
            df["player_name"] = df["player_name"].astype(str)
        if "position" in df.columns:
            df["position"] = df["position"].astype(str)

    return passing, receiving, rushing


def _collect_filter_values(
    passing: pd.DataFrame, receiving: pd.DataFrame, rushing: pd.DataFrame
) -> Tuple[List[int], List[str], List[str]]:
    # Seasons across all tables
    seasons = pd.concat(
        [
            passing[["season"]],
            receiving[["season"]],
            rushing[["season"]],
        ]
    )["season"].dropna().astype(int).unique().tolist()
    seasons = sorted(seasons)

    # Positions across receiving/rushing, and QB for passing
    pos_values = set(["QB"]) | set(
        pd.concat([receiving[["position"]], rushing[["position"]]])["position"].dropna().unique().tolist()
    )
    positions = sorted([p for p in pos_values if p])

    # Teams across all
    teams = pd.concat([
        passing[["team"]], receiving[["team"]], rushing[["team"]]
    ])["team"].dropna().unique().tolist()
    teams = sorted([t for t in teams if t])

    return seasons, positions, teams


def _apply_common_filters(
    df: pd.DataFrame,
    season: Optional[int],
    positions: List[str],
    teams: List[str],
    players: List[str],
) -> pd.DataFrame:
    result = df.copy()
    if season is not None and "season" in result.columns:
        result = result[result["season"].astype("Int64") == season]
    if positions:
        if "position" in result.columns:
            result = result[result["position"].isin(positions)]
        else:
            # Passing table will pass positions=["QB"]
            result = result[result.get("position", "").isin(positions)]
    if teams:
        result = result[result["team"].isin(teams)]
    if players:
        result = result[result["player_name"].isin(players)]
    return result


def _inject_minimal_css() -> None:
    # Keep it minimal to avoid duplication with main page
    st.markdown(
        """
        <style>
        .header-title { font-weight: 700; font-size: 1.2rem; }
        .caption { color: #6b7280; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="PFR Tables — Players Stats", layout="wide")
    _inject_minimal_css()

    st.markdown("<div class='header-title'>PFR Tables</div>", unsafe_allow_html=True)
    st.caption("Tabelas de Passing, Receiving e Rushing (PFR) para exploração por temporada, posição, time e jogador.")

    passing, receiving, rushing = _load_pfr_tables()
    seasons, positions, teams_all = _collect_filter_values(passing, receiving, rushing)

    # Sidebar filters
    with st.sidebar:
        st.markdown("<div class='header-title'>Filtros</div>", unsafe_allow_html=True)
        season = st.selectbox(
            "Temporada",
            options=seasons,
            index=seasons.index(DEFAULT_SEASON) if DEFAULT_SEASON in seasons else len(seasons) - 1,
        )

        # Positions: allow multiple; default none -> show all
        pos_sel = st.multiselect("Posição", options=positions, default=[])

        # Teams available for chosen season across all tables
        teams_for_season = pd.concat([
            passing[passing["season"].astype("Int64") == season][["team"]],
            receiving[receiving["season"].astype("Int64") == season][["team"]],
            rushing[rushing["season"].astype("Int64") == season][["team"]],
        ])["team"].dropna().unique().tolist()
        teams_for_season = sorted([t for t in teams_for_season if t])
        team_sel = st.multiselect("Time", options=teams_for_season or teams_all, default=[])

        # Players available for chosen season + team/position prefilter (to keep list reasonable)
        candidate_players = pd.concat([
            _apply_common_filters(passing, season, ["QB"] if (pos_sel and "QB" in pos_sel) else [], team_sel, [])[["player_name"]],
            _apply_common_filters(receiving, season, pos_sel, team_sel, [])[["player_name"]],
            _apply_common_filters(rushing, season, pos_sel, team_sel, [])[["player_name"]],
        ], ignore_index=True)["player_name"].dropna().unique().tolist()
        candidate_players = sorted(candidate_players)
        player_sel = st.multiselect("Jogador", options=candidate_players, default=[])

    tab1, tab2, tab3 = st.tabs(["Passing", "Receiving", "Rushing"])

    with tab1:
        st.subheader("Passing")
        df = _apply_common_filters(passing, season, ["QB"] if (not pos_sel or "QB" in pos_sel) else [], team_sel, player_sel)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Baixar CSV (Passing)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"pfr_passing_{season}.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Receiving")
        df = _apply_common_filters(receiving, season, pos_sel, team_sel, player_sel)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Baixar CSV (Receiving)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"pfr_receiving_{season}.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Rushing")
        df = _apply_common_filters(rushing, season, pos_sel, team_sel, player_sel)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Baixar CSV (Rushing)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"pfr_rushing_{season}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


