# fantasy_adp_prob_app.py

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from math import erf, sqrt
import requests
from datetime import datetime, timezone
import time
import re

# =============================
# ---------- Helpers ----------
# =============================

def normal_cdf(z):
    """Vectorized CDF of the standard normal. Accepts scalars or numpy arrays."""
    z_arr = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(erf)
    return 0.5 * (1.0 + erf_vec(z_arr / np.sqrt(2.0)))

def prob_available_vector(mu: np.ndarray, sigma: np.ndarray, K: int) -> np.ndarray:
    """P(X > K) for X ~ N(mu, sigma^2). Clipped to [0,1]. Safe for arrays."""
    sigma_safe = np.where(~np.isfinite(sigma) | (sigma <= 0), 1.0, sigma)
    z = (K - mu) / sigma_safe
    p = 1.0 - normal_cdf(z)
    return np.clip(p, 0.0, 1.0)

# =============================
# ---- Sleeper Integration ----
# =============================

API_BASE = "https://api.sleeper.app/v1"
HTTP_TIMEOUT_SEC = 6.0
POLL_SEC_DEFAULT = 5
PLAYERS_DICT_TTL_SEC = 24 * 3600  # 1 day
DRAFT_META_TTL_SEC = 120
PICKS_TTL_SEC = 2
MAX_BACKOFF_SEC = 60


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _player_key(name: str, pos: str) -> str:
    return f"{_normalize_name(name)}|{str(pos).upper()}"


def sleeper_get(path: str, params: Dict | None = None) -> Dict | List | None:
    url = f"{API_BASE}{path}"
    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=PLAYERS_DICT_TTL_SEC)
def sleeper_players_dict() -> Dict:
    data = sleeper_get("/players/nfl")
    return data if isinstance(data, dict) else {}


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_user(username: str) -> Dict | None:
    if not username:
        return None
    return sleeper_get(f"/user/{username}")


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_user_leagues(user_id: str, season: str) -> List[Dict]:
    if not user_id:
        return []
    leagues = sleeper_get(f"/user/{user_id}/leagues/nfl/{season}")
    return leagues if isinstance(leagues, list) else []


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_league_drafts(league_id: str) -> List[Dict]:
    if not league_id:
        return []
    drafts = sleeper_get(f"/league/{league_id}/drafts")
    return drafts if isinstance(drafts, list) else []


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_draft_meta(draft_id: str) -> Dict | None:
    if not draft_id:
        return None
    data = sleeper_get(f"/draft/{draft_id}")
    return data if isinstance(data, dict) else None


@st.cache_data(show_spinner=False, ttl=PICKS_TTL_SEC)
def sleeper_draft_picks(draft_id: str) -> List[Dict]:
    if not draft_id:
        return []
    picks = sleeper_get(f"/draft/{draft_id}/picks")
    return picks if isinstance(picks, list) else []


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_league_rosters(league_id: str) -> List[Dict]:
    if not league_id:
        return []
    rosters = sleeper_get(f"/league/{league_id}/rosters")
    return rosters if isinstance(rosters, list) else []


@st.cache_data(show_spinner=False, ttl=DRAFT_META_TTL_SEC)
def sleeper_league_users(league_id: str) -> List[Dict]:
    if not league_id:
        return []
    users = sleeper_get(f"/league/{league_id}/users")
    return users if isinstance(users, list) else []


def build_slot_names(draft_id: str, league_id: str | None) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not draft_id or not league_id:
        return names
    meta = sleeper_draft_meta(draft_id) or {}
    slot_to_roster = meta.get("slot_to_roster_id") or meta.get("slot_to_roster_id_map") or {}
    rosters = sleeper_league_rosters(league_id)
    users = sleeper_league_users(league_id)
    roster_id_to_owner: Dict[int, str] = {}
    for r in rosters:
        try:
            roster_id_to_owner[int(r.get("roster_id"))] = str(r.get("owner_id"))
        except Exception:
            continue
    user_id_to_name: Dict[str, str] = {}
    for u in users:
        user_id_to_name[str(u.get("user_id"))] = str(u.get("display_name") or u.get("username") or "")
    for k, v in (slot_to_roster or {}).items():
        try:
            slot = int(k)
            roster_id = int(v)
            owner_id = roster_id_to_owner.get(roster_id)
            disp = user_id_to_name.get(str(owner_id)) if owner_id else None
            if disp:
                names[slot] = disp
        except Exception:
            continue
    return names


def resolve_draft_id_flow(username: str, league_id: str, draft_id: str, season: str) -> Tuple[str | None, str | None]:
    # Returns (resolved_draft_id, resolved_league_id)
    if draft_id:
        return draft_id, league_id or None
    resolved_league = league_id
    if not resolved_league and username:
        u = sleeper_user(username)
        user_id = u.get("user_id") if isinstance(u, dict) else None
        leagues = sleeper_user_leagues(user_id, season) if user_id else []
        if leagues:
            # Choose the most recently created league
            leagues_sorted = sorted(leagues, key=lambda x: x.get("created", 0), reverse=True)
            resolved_league = leagues_sorted[0].get("league_id")
    if resolved_league:
        drafts = sleeper_league_drafts(resolved_league)
        if drafts:
            # Prefer in_progress or most recent
            def _draft_score(d):
                status = str(d.get("status", ""))
                created = d.get("created", 0)
                return (1 if status == "in_progress" else 0, created)
            best = sorted(drafts, key=_draft_score, reverse=True)[0]
            return best.get("draft_id"), resolved_league
    return None, resolved_league


def get_official_num_teams(draft_id: str, league_id: str | None) -> int | None:
    meta = sleeper_draft_meta(draft_id) if draft_id else None
    if isinstance(meta, dict):
        settings = meta.get("settings") or {}
        num_teams = settings.get("teams") or settings.get("slots")
        if isinstance(num_teams, int) and num_teams > 0:
            return int(num_teams)
    if league_id:
        league = sleeper_get(f"/league/{league_id}")
        if isinstance(league, dict):
            settings = league.get("settings") or {}
            num_teams = settings.get("num_teams") or settings.get("teams")
            if isinstance(num_teams, int) and num_teams > 0:
                return int(num_teams)
    # Fallback: infer from picks
    if draft_id:
        picks = sleeper_draft_picks(draft_id)
        slots = [p.get("draft_slot") for p in picks if isinstance(p, dict) and p.get("draft_slot")]
        if slots:
            return int(max(slots))
    return None


def build_local_lookup(players_df: pd.DataFrame) -> Dict[str, str]:
    # Map normalized name + pos => local player_id
    lookup: Dict[str, str] = {}
    for _, row in players_df.iterrows():
        name = str(row.get("Player", ""))
        pos = str(row.get("POS", "")).upper()
        pid = str(row.get("player_id", ""))
        key = _player_key(name, pos)
        if key and pid:
            lookup[key] = pid
    return lookup


def try_match_sleeper_to_local(sp: Dict, lookup: Dict[str, str], players_df: pd.DataFrame) -> str | None:
    # sp keys: player_id, metadata including full_name/team/position
    full_name = sp.get("metadata", {}).get("full_name") or sp.get("player_name") or ""
    position = (sp.get("metadata", {}).get("position") or sp.get("position") or "").upper()
    key = _player_key(full_name, position)
    pid = lookup.get(key)
    if pid:
        return pid
    # Fallback: try last name + pos if unique
    name_norm = _normalize_name(full_name)
    last = name_norm.split(" ")[-1] if name_norm else ""
    if last:
        candidates = players_df[players_df["POS"].str.upper() == position]
        candidates = candidates[candidates["Player"].str.lower().str.contains(last, na=False)]
        if len(candidates) == 1:
            return str(candidates.iloc[0]["player_id"])
    return None


def apply_sleeper_sync(players_df: pd.DataFrame, draft: 'DraftState', picks: List[Dict], manual_map: Dict[str, str]) -> Tuple[pd.DataFrame, 'DraftState', List[Tuple[str, str, str]]]:
    # Returns updated (players_df, draft, unresolved_list[(sleeper_id, name, pos)])
    df = players_df.copy()
    if df.empty:
        return df, draft, []
    df["picked"] = False

    lookup = build_local_lookup(df)
    # Direct mapping via sleeper_id column when available
    sid_to_local: Dict[str, str] = {}
    if "sleeper_id" in df.columns:
        try:
            sid_series = df["sleeper_id"].astype(str).str.strip()
            for local_pid, sid in zip(df["player_id"], sid_series):
                if sid and sid.lower() != "nan":
                    sid_to_local[str(sid)] = str(local_pid)
        except Exception:
            sid_to_local = {}
    unresolved: List[Tuple[str, str, str]] = []

    # Prepare teams structure from picks
    teams_map: Dict[int, List[Tuple[str, str]]] = {}
    history: List[Tuple[int, str, str]] = []

    for idx, p in enumerate(sorted(picks, key=lambda x: (x.get("round", 0), x.get("pick", 0), x.get("overall", 0)))):
        sleeper_pid = str(p.get("player_id", ""))
        meta = p.get("metadata", {}) if isinstance(p.get("metadata"), dict) else {}
        full_name = meta.get("full_name") or p.get("player_name") or ""
        position = (meta.get("position") or p.get("position") or "").upper()
        draft_slot = int(p.get("draft_slot", 0) or 0)

        if position not in {"QB", "RB", "WR", "TE"}:
            continue

        # Priority 1: manual override
        local_pid = manual_map.get(sleeper_pid)
        # Priority 2: direct sleeper_id column mapping
        if not local_pid and sleeper_pid and sid_to_local:
            local_pid = sid_to_local.get(sleeper_pid)
        # Priority 3: heuristic matching
        if not local_pid:
            local_pid = try_match_sleeper_to_local(p, lookup, df)

        if local_pid and (df["player_id"] == local_pid).any():
            # mark picked
            df.loc[df["player_id"] == local_pid, "picked"] = True
            row = df.loc[df["player_id"] == local_pid].iloc[0]
            history.append((len(history) + 1, local_pid, str(row["POS"])) )
            if draft_slot not in teams_map:
                teams_map[draft_slot] = []
            teams_map[draft_slot].append((str(row["Player"]), str(row["POS"])) )
        else:
            unresolved.append((sleeper_pid, str(full_name), str(position)))

    # Rebuild draft teams in slot order
    if isinstance(draft.N, int) and draft.N > 0:
        max_slot = max([0] + list(teams_map.keys()))
        num_teams = max(draft.N, max_slot)
    else:
        num_teams = max([0] + list(teams_map.keys()))
    teams: List[TeamState] = [TeamState() for _ in range(max(1, num_teams))]
    for slot, plist in teams_map.items():
        if 1 <= slot <= len(teams):
            teams[slot - 1].players = plist.copy()
            # recompute taken counts
            taken = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
            for _, pos in plist:
                if pos in taken:
                    taken[pos] += 1
            teams[slot - 1].taken = taken

    draft.teams = teams
    draft.history = history
    draft.current_pick = len(history) + 1
    draft.N = len(teams)

    return df, draft, unresolved

# =============================
# ---------- Draft Types ------
# =============================

@dataclass
class RosterRules:
    slots: Dict[str, int]     # {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1}
    flex_map: Dict[str, float]  # {"RB":1,"WR":1,"TE":1}

@dataclass
class TeamState:
    taken: Dict[str, int] = field(default_factory=lambda: {"QB": 0, "RB": 0, "WR": 0, "TE": 0})
    players: List[Tuple[str, str]] = field(default_factory=list)  # (player_name, pos)

@dataclass
class DraftState:
    N: int
    user_slot: int           # 1..N
    current_pick: int        # global pick (1-indexed)
    order_snake: bool = True
    roster_rules: RosterRules = None
    teams: List[TeamState] = None
    history: List[Tuple[int, str, str]] = field(default_factory=list)  # (pick_no, player_id, POS)

    def next_pick_for_slot(self, slot: int, from_pick: int = None) -> int:
        if from_pick is None:
            from_pick = self.current_pick
        r = ((from_pick - 1) // self.N) + 1
        p = from_pick
        while True:
            if r % 2 == 1:
                k = (r - 1) * self.N + slot
            else:
                k = (r - 1) * self.N + (self.N - slot + 1)
            if k > p:
                return k
            r += 1

    def user_next_pick(self) -> int:
        return self.next_pick_for_slot(self.user_slot, self.current_pick)

# =============================
# -------- Data Hygiene -------
# =============================

def sanitize_player_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize common column names
    df.rename(columns={"adp_std": "ADP_STD", "avg": "AVG", "fpts": "FPTS"}, inplace=True)

    # POS validation
    if "POS" not in df.columns:
        raise ValueError("A coluna POS é obrigatória (QB/RB/WR/TE).")
    df["POS"] = df["POS"].astype(str).str.upper().str.strip()
    valid_pos = {"QB", "RB", "WR", "TE"}
    df = df[df["POS"].isin(valid_pos)]

    # Ensure numeric
    for c in ["ADP", "ADP_STD", "FPTS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If ADP missing but sources exist, compute
    source_cols = [c for c in ["ESPN", "Sleeper", "NFL", "RTSports", "FFC", "Fantrax"] if c in df.columns]
    for c in source_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ADP" not in df.columns and source_cols:
        df["ADP"] = df[source_cols].mean(axis=1, skipna=True)
    if "ADP_STD" not in df.columns and source_cols:
        df["ADP_STD"] = df[source_cols].std(axis=1, ddof=0, skipna=True)

    # Remove non-finite ADP/STD
    for c in ["ADP", "ADP_STD"]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df[df["ADP"].notna() & df["ADP_STD"].notna()]

    # Practical caps
    df["ADP"] = df["ADP"].clip(lower=1)
    df["ADP_STD"] = df["ADP_STD"].clip(lower=0.5)

    # Minimum columns
    if "Player" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'Player'.")
    if "Team" not in df.columns:
        df["Team"] = ""
    if "FPTS" not in df.columns:
        df["FPTS"] = np.nan  # optional
    if "tier" not in df.columns:
        df["tier"] = np.nan  # optional

    # Optional: sleeper_id for direct mapping to Sleeper API
    if "sleeper_id" in df.columns:
        df["sleeper_id"] = df["sleeper_id"].astype(str).str.strip()

    if "player_id" not in df.columns:
        df["player_id"] = (
            df["Player"].astype(str).str.lower().str.replace(" ", "_", regex=False)
            + "_" + df["Team"].astype(str).str.lower()
        )

    # ---- Tier handling ----
    if "tier" in df.columns:
        df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    else:
        df["tier"] = np.nan

    # Auto-tier por posição quando faltar muitos tiers
    missing_ratio = df["tier"].isna().mean()
    if missing_ratio >= 0.5:
        N_TIERS = 6
        def _assign_tiers_pos(sub: pd.DataFrame) -> pd.DataFrame:
            sub = sub.copy()
            ranks = sub["ADP"].rank(method="first", ascending=True)
            try:
                labels = list(range(1, N_TIERS + 1))
                sub["tier"] = pd.qcut(ranks, q=N_TIERS, labels=labels)
            except Exception:
                sub["tier"] = 1
            sub["tier"] = pd.to_numeric(sub["tier"], errors="coerce")
            return sub
        df = df.groupby("POS", group_keys=False).apply(_assign_tiers_pos)

    df["tier"] = pd.to_numeric(df["tier"], errors="coerce")
    return df

# =============================
# ------- Core Calculator -----
# =============================

def compute_probabilities(players_df: pd.DataFrame,
                          draft: DraftState,
                          chaos: float = 0.3,
                          sigma_min: float = 2.0,
                          gamma_by_pos: Dict[str, float] = {"RB": 1.0, "WR": 0.9, "QB": 0.7, "TE": 1.1},
                          run_window: int = 12,
                          baseline_rate: Dict[str, float] = {"RB": 0.33, "WR": 0.42, "QB": 0.13, "TE": 0.12},
                          tau_hot: float = 3.0,
                          run_z_threshold: float = 1.0) -> pd.DataFrame:
    """Simplified, robust probability computation."""
    alive = players_df.loc[~players_df["picked"].astype(bool)].copy()
    alive = alive[np.isfinite(alive["ADP"]) & np.isfinite(alive["ADP_STD"])]
    if alive.empty:
        return alive.assign(mu_adj=[], sigma_adj=[], prob_available_next_pick=[])

    K_user = draft.user_next_pick()
    picks_to_user = max(0, K_user - draft.current_pick)

    pos_list = ["QB","RB","WR","TE"]
    need_counts = {p: baseline_rate.get(p, 0.25) * picks_to_user for p in pos_list}

    recent = draft.history[-run_window:]
    if len(recent) > 0:
        counts = {p:0 for p in pos_list}
        for _,_,p in recent:
            if p in counts:
                counts[p]+=1
        n = len(recent)
        hot = {}
        for p in pos_list:
            base = baseline_rate.get(p, 0.0)
            var = n*base*(1-base)+1e-6
            z = (counts[p] - base*n)/math.sqrt(var)
            hot[p] = z >= run_z_threshold
    else:
        hot = {p: False for p in pos_list}

    mu = alive["ADP"].to_numpy(float)
    sig = alive["ADP_STD"].to_numpy(float)
    pos = alive["POS"].to_numpy(str)

    mu_adj = mu.copy()
    sigma_adj = np.maximum(sig * (1 + chaos), sigma_min)

    demand_by_pos = {p: need_counts.get(p,0.0) for p in pos_list}
    for i in range(len(alive)):
        p = pos[i]
        shift = gamma_by_pos.get(p,1.0) * demand_by_pos.get(p,0.0) / max(1, len(pos_list))
        mu_adj[i] = mu[i] - shift
        if hot.get(p, False):
            sigma_adj[i] = np.sqrt(sigma_adj[i]**2 + tau_hot**2)

    probs = prob_available_vector(mu_adj, sigma_adj, K_user)

    out = alive.copy()
    out["mu_adj"] = mu_adj
    out["sigma_adj"] = sigma_adj
    out["prob_available_next_pick"] = probs
    return out

# =============================
# --------- Data Loading ------
# =============================

def load_players(default_path: str = None, uploaded: bytes = None) -> pd.DataFrame:
    # 1) Upload do usuário tem prioridade
    if uploaded is not None:
        df = pd.read_csv(uploaded)

    # 2) Caminho padrão: aceita URL (http/https) ou arquivo local
    elif default_path:
        if isinstance(default_path, str) and default_path.lower().startswith(("http://", "https://")):
            df = pd.read_csv(default_path)
        elif os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            raise FileNotFoundError(
                f"Arquivo de jogadores não encontrado em '{default_path}'. "
                f"Envie um CSV ou corrija o caminho/URL."
            )
    else:
        raise FileNotFoundError("Arquivo de jogadores não encontrado. Forneça o CSV.")

    # Mapear colunas para ADP/ADP_STD quando necessário
    if "AVG" in df.columns:
        df["ADP"] = pd.to_numeric(df["AVG"], errors="coerce")
    source_cols = [c for c in ["ESPN","Sleeper","NFL","RTSports","FFC","Fantrax"] if c in df.columns]
    if "ADP" not in df.columns and source_cols:
        df["ADP"] = df[source_cols].mean(axis=1, skipna=True)
    if "adp_std" in df.columns:
        df["ADP_STD"] = pd.to_numeric(df["adp_std"], errors="coerce")
    elif source_cols:
        df["ADP_STD"] = df[source_cols].std(axis=1, ddof=0, skipna=True)

    df = sanitize_player_data(df)
    df["picked"] = False
    base_cols = ["player_id","Player","Team","POS","ADP","ADP_STD","FPTS","tier","picked"]
    if "sleeper_id" in df.columns:
        base_cols.insert(1, "sleeper_id")  # keep close to ids
    return df[[c for c in base_cols if c in df.columns]]


# =============================
# ----- Card calc + UI --------
# =============================

def compute_card_for_pos(result_df: pd.DataFrame, pos: str, chosen_main: str | None = None) -> Dict[str, str]:
    """
    Calcula os campos do card para uma POS específica usando result_df (já com probas).
    - chosen_main: se fornecido, usa este jogador como principal (se existir na posição).
    - 'Próximo' = primeiro com P(sobrar) >= 50% (fallback: próximo da lista).
    """
    dpos = result_df[result_df["POS"] == pos].copy()
    if dpos.empty:
        return {"main":"—","psobrar":"—","fpts1":"—","next_name":"—","fpts2":"—","custo":"—","risk":"—"}

    # Ordenação base: FPTS desc quando houver; senão ADP asc
    if "FPTS" in dpos.columns and dpos["FPTS"].notna().any():
        dpos = dpos.sort_values(["FPTS","ADP"], ascending=[False, True])
    else:
        dpos = dpos.sort_values(["ADP","Player"], ascending=[True, True])

    # Principal: escolhido ou default top
    if chosen_main and chosen_main in dpos["Player"].values:
        j1 = dpos.loc[dpos["Player"] == chosen_main].iloc[0]
    else:
        j1 = dpos.iloc[0]

    name1 = str(j1["Player"])
    p1 = float(j1.get("prob_available_next_pick", np.nan))
    fpts1 = float(j1.get("FPTS", np.nan))

    # Próximo = primeiro por ADP com P(sobrar) >= 50% (pode ser o próprio principal)
    next_thresh = 0.50
    dpos_adp = dpos.sort_values(["ADP", "Player"], ascending=[True, True]).copy()

    if "prob_available_next_pick" in dpos_adp.columns:
        dpos_adp["p_next"] = pd.to_numeric(dpos_adp["prob_available_next_pick"], errors="coerce").fillna(0.0)
    else:
        dpos_adp["p_next"] = 0.0

    cands = dpos_adp[dpos_adp["p_next"] >= next_thresh]

    if not cands.empty:
        j2 = cands.iloc[0]  # primeiro por ADP com P>=50%
    else:
        j2 = dpos_adp.iloc[0]  # fallback: primeiro por ADP

    name2 = str(j2["Player"])
    fpts2 = float(j2.get("FPTS", np.nan)) if np.isfinite(j2.get("FPTS", np.nan)) else (fpts1 if np.isfinite(fpts1) else np.nan)

    # Formats
    if np.isfinite(p1):
        ps = int(round(np.clip(p1,0,1)*100))
        psobrar_txt = f"{ps}%"
    else:
        psobrar_txt = "N/A"

    fpts1_txt = f"{fpts1:.1f}" if np.isfinite(fpts1) else "N/A"
    fpts2_txt = f"{fpts2:.1f}" if np.isfinite(fpts2) else "N/A"

    # EV if pass ≈ p1*FPTS1 + (1-p1)*FPTS2  (usa p1 do principal)
    if np.isfinite(p1) and np.isfinite(fpts1) and np.isfinite(fpts2):
        ev_pass = p1*fpts1 + (1-p1)*fpts2
        custo = fpts1 - ev_pass
        custo_txt = f"{custo:.1f} pts"
    else:
        custo_txt = "N/A"

    # Risk missing tier
    tier1 = j1.get("tier", np.nan)
    if np.isfinite(tier1):
        same_tier = dpos[dpos.get("tier").astype(float) == float(tier1)]
        if not same_tier.empty and "prob_available_next_pick" in same_tier.columns:
            probs = same_tier["prob_available_next_pick"].astype(float).clip(0,1)
            miss = float(np.prod(1 - probs.values))
            risk_txt = f"{int(round(miss*100))}%"
        else:
            risk_txt = "—"
    else:
        risk_txt = "N/A"

    return {"main":name1, "psobrar":psobrar_txt, "fpts1":fpts1_txt, "next_name":name2, "fpts2":fpts2_txt, "custo":custo_txt, "risk":risk_txt}

def render_cards(cards: Dict[str, Dict[str, str]]):
    st.markdown(
        """
        <style>
          .card {border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.06) 0%,rgba(255,255,255,.02) 100%);backdrop-filter:blur(6px);height:100%}
          .card h4{margin:0 0 8px 0;font-size:1.05rem;letter-spacing:.5px;text-transform:uppercase;opacity:.85}
          .card .main{font-weight:700;font-size:1.05rem;margin-bottom:10px}
          .grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;font-size:.92rem}
          .kv{opacity:.9}.val{font-weight:600;text-align:right}
          .cost{margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,.15);display:flex;justify-content:space-between;font-size:1rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    labels = {"QB":"QUARTERBACK","RB":"RUNNING BACK","WR":"WIDE RECEIVER","TE":"TIGHT END"}
    cols = st.columns(4)
    for i, pos in enumerate(["QB","RB","WR","TE"]):
        c = cards.get(pos, {})
        with cols[i]:
            st.markdown(f"""
            <div class=card>
              <h4>{labels.get(pos,pos)}</h4>
              <div class=main>{c.get('main','—')}</div>
              <div class=grid>
                <div class=kv>P(sobrar)</div><div class=val>{c.get('psobrar','—')}</div>
                <div class=kv>FPTS Pick</div><div class=val>{c.get('fpts1','—')}</div>
                <div class=kv>Próximo</div><div class=val>{c.get('next_name','—')}</div>
                <div class=kv>FPTS Next</div><div class=val>{c.get('fpts2','—')}</div>
                <div class=kv>Risco Missing Tier</div><div class=val>{c.get('risk','—')}</div>
              </div>
              <div class=cost><div>Custo de Passe</div><div class=val>{c.get('custo','—')}</div></div>
            </div>
            """, unsafe_allow_html=True)

# =============================
# ------------- UI ------------
# =============================

st.set_page_config(page_title="ADP Prob Draft", layout="wide")
st.title("🔮 Probabilidade de Jogador Disponível — Draft NFL (com runs e necessidades)")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Configurações da Liga")
    N = st.number_input("Nº de times", min_value=4, max_value=16, value=10, step=1)
    user_slot = st.number_input("Seu slot (1=Nº times)", min_value=1, max_value=int(N), value=1, step=1)

    st.markdown("---")
    st.header("Parâmetros do Modelo")
    chaos = st.slider("Caos (imprevisibilidade)", 0.0, 0.8, 0.3, 0.05)
    run_window = st.slider("Janela de run (últimas picks)", 6, 20, 12, 1)
    sigma_min = st.slider("Desvio mínimo (picks)", 0.5, 6.0, 2.0, 0.5)
    tau_hot = st.slider("Inflar variância quando em run (τ)", 0.0, 8.0, 3.0, 0.5)
    run_z_threshold = st.slider("Z-score p/ marcar run", 0.5, 2.5, 1.0, 0.1)

    st.markdown("---")
    st.header("Dados")
    default_path = "https://raw.githubusercontent.com/bandresthomas/webapp/main/tables/adp_app_table.csv"
    st.caption(f"Caminho padrão: {default_path}")
    uploaded = st.file_uploader("Ou envie o CSV aqui", type=["csv"])  # opcional

    st.markdown("---")
    st.header("Integração Sleeper")
    if "sleeper" not in st.session_state:
        st.session_state["sleeper"] = {
            "enabled": False,
            "paused": False,
            "username": "",
            "league_id": "",
            "draft_id": "",
            "season": str(datetime.now().year),
            "poll_sec": POLL_SEC_DEFAULT,
            "last_sync_ts": 0.0,
            "last_error_ts": 0.0,
            "backoff_sec": 0,
            "status": "idle",
            "logs": [],
            "manual_map": {},  # sleeper_player_id -> local player_id
            "unresolved": [],  # list of tuples (sleeper_id, name, pos)
        }

    sleeper_state = st.session_state["sleeper"]

    sleeper_state["enabled"] = st.checkbox("Habilitar sync com Sleeper", value=sleeper_state.get("enabled", False))
    sleeper_state["paused"] = st.checkbox("Pausar sincronização", value=sleeper_state.get("paused", False), disabled=not sleeper_state["enabled"]) if sleeper_state["enabled"] else sleeper_state["paused"]

    sleeper_state["username"] = st.text_input("Sleeper username (opcional)", value=sleeper_state.get("username", ""))
    col_sid1, col_sid2 = st.columns(2)
    with col_sid1:
        sleeper_state["league_id"] = st.text_input("league_id (opcional)", value=sleeper_state.get("league_id", ""))
    with col_sid2:
        sleeper_state["draft_id"] = st.text_input("draft_id (opcional)", value=sleeper_state.get("draft_id", ""))
    col_season, col_poll = st.columns(2)
    with col_season:
        sleeper_state["season"] = st.text_input("Season", value=sleeper_state.get("season", str(datetime.now().year)))
    with col_poll:
        sleeper_state["poll_sec"] = int(st.slider("Polling (s)", 3, 20, sleeper_state.get("poll_sec", POLL_SEC_DEFAULT), 1))

    col_btns1, col_btns2, col_btns3 = st.columns(3)
    if col_btns1.button("Resolver Draft ID") and sleeper_state["enabled"]:
        rid, rleague = resolve_draft_id_flow(sleeper_state.get("username", ""), sleeper_state.get("league_id", ""), sleeper_state.get("draft_id", ""), sleeper_state.get("season", str(datetime.now().year)))
        if rid:
            sleeper_state["draft_id"] = rid
            if rleague:
                sleeper_state["league_id"] = rleague
            sleeper_state["status"] = f"Conectado ao draft_id {rid}"
            sleeper_state["logs"].append(f"[{_now_utc_iso()}] Resolved draft_id={rid} league_id={rleague}")
        else:
            sleeper_state["status"] = "Não foi possível resolver o draft"
            sleeper_state["logs"].append(f"[{_now_utc_iso()}] Falha ao resolver draft")

    if col_btns2.button("Sync agora") and sleeper_state["enabled"]:
        sleeper_state["last_sync_ts"] = 0.0  # force immediate sync in main loop
        sleeper_state["last_error_ts"] = 0.0
        sleeper_state["backoff_sec"] = 0
        st.rerun()

    if col_btns3.button("Limpar mapeamentos"):
        sleeper_state["manual_map"] = {}
        sleeper_state["logs"].append(f"[{_now_utc_iso()}] Limpei mapeamentos manuais")

# ---------- Init/Load Data ----------
if "players_df" not in st.session_state:
    try:
        st.session_state.players_df = load_players(default_path=default_path, uploaded=uploaded)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

# allow reload if user uploads after start
if uploaded is not None and st.session_state.get("_uploaded_sha") != hash(uploaded.getvalue()):
    try:
        st.session_state.players_df = load_players(uploaded=uploaded)
        st.session_state["_uploaded_sha"] = hash(uploaded.getvalue())
        for key in ["draft", "pick_log"]:
            st.session_state.pop(key, None)
        st.rerun()
    except Exception as e:
        st.error(f"Erro ao recarregar dados enviados: {e}")
        st.stop()

players_df = st.session_state.players_df

## Background sync moved below after draft initialization

# Status/Logs panel
with st.expander("Status & Logs — Sleeper"):
    s = st.session_state.get("sleeper", {})
    if s.get("enabled"):
        st.write({
            "status": s.get("status"),
            "draft_id": s.get("draft_id"),
            "league_id": s.get("league_id"),
            "paused": s.get("paused"),
            "poll_sec": s.get("poll_sec"),
            "backoff_sec": s.get("backoff_sec"),
            "last_sync_ts": s.get("last_sync_ts"),
            "unresolved": len(s.get("unresolved", [])),
        })
        if s.get("unresolved"):
            st.markdown("Jogadores não mapeados do Sleeper:")
            df_unr = pd.DataFrame(s.get("unresolved"), columns=["sleeper_id", "name", "pos"]) if isinstance(s.get("unresolved"), list) else pd.DataFrame()
            if not df_unr.empty:
                st.dataframe(df_unr, use_container_width=True)
                # Mapping UI
                col_map1, col_map2 = st.columns([2, 3])
                with col_map1:
                    sel_unr = st.selectbox("Selecionar sleeper_id p/ mapear", df_unr["sleeper_id"].tolist())
                with col_map2:
                    # Suggest candidates by position
                    pos_mask = players_df["POS"].isin(df_unr.loc[df_unr["sleeper_id"] == sel_unr, "pos"].tolist())
                    candidates = players_df.loc[~players_df["picked"] & pos_mask, ["player_id", "Player", "POS", "Team", "ADP"]].copy()
                    candidates.sort_values(["ADP", "Player"], inplace=True)
                    options = candidates.apply(lambda r: f"{r['Player']} ({r['POS']}-{r['Team']}) • ADP {int(r['ADP']) if pd.notna(r['ADP']) else '-'}", axis=1).tolist()
                    ids = candidates["player_id"].tolist()
                    map_choice = st.selectbox("Mapear para jogador local", options)
                    if st.button("Salvar mapeamento"):
                        if map_choice and ids and options:
                            pid = ids[options.index(map_choice)] if map_choice in options else None
                            if pid:
                                st.session_state["sleeper"]["manual_map"][str(sel_unr)] = str(pid)
                                st.session_state["sleeper"]["logs"].append(f"[{_now_utc_iso()}] Mapeei sleeper {sel_unr} -> {pid}")
                                # Trigger resync soon
                                st.session_state["sleeper"]["last_sync_ts"] = 0.0
                                st.rerun()
        st.markdown("Logs:")
        for line in list(s.get("logs", []))[-200:]:
            st.code(line)
    else:
        st.write("Integração desabilitada.")
# ---------- Initialize Draft State ----------
if "draft" not in st.session_state or st.session_state.draft.N != N or st.session_state.draft.user_slot != user_slot:
    teams = [TeamState() for _ in range(int(N))]
    roster = RosterRules(slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, flex_map={"RB": 1, "WR": 1, "TE": 1})
    st.session_state.draft = DraftState(N=int(N), user_slot=int(user_slot), current_pick=1, roster_rules=roster, teams=teams)
    st.session_state.pick_log = []  # for undo

draft = st.session_state.draft

# ---------- Sleeper: Background Sync (after draft init) ----------
if st.session_state.get("sleeper", {}).get("enabled"):
    sleeper_state = st.session_state["sleeper"]

    if not sleeper_state.get("draft_id"):
        rid, rleague = resolve_draft_id_flow(sleeper_state.get("username", ""), sleeper_state.get("league_id", ""), None, sleeper_state.get("season", str(datetime.now().year)))
        if rid:
            sleeper_state["draft_id"] = rid
            if rleague:
                sleeper_state["league_id"] = rleague
            sleeper_state["logs"].append(f"[{_now_utc_iso()}] Auto-resolve draft_id={rid} league_id={rleague}")

    if sleeper_state.get("draft_id"):
        official_N = get_official_num_teams(sleeper_state["draft_id"], sleeper_state.get("league_id"))
        if isinstance(official_N, int) and official_N > 0 and official_N != st.session_state.draft.N:
            st.session_state.draft.N = int(official_N)
            # If N changed, ensure teams size matches
            if len(st.session_state.draft.teams) != int(official_N):
                st.session_state.draft.teams = [TeamState() for _ in range(int(official_N))]

    now_ts = time.time()
    due = (now_ts - sleeper_state.get("last_sync_ts", 0)) >= max(1, sleeper_state.get("poll_sec", POLL_SEC_DEFAULT))
    blocked_by_backoff = (now_ts - sleeper_state.get("last_error_ts", 0)) < sleeper_state.get("backoff_sec", 0)

    if not sleeper_state.get("paused", False) and sleeper_state.get("draft_id") and due and not blocked_by_backoff:
        try:
            picks = sleeper_draft_picks(sleeper_state["draft_id"])
            if isinstance(picks, list):
                new_df, new_draft, unresolved = apply_sleeper_sync(players_df, st.session_state.draft, picks, sleeper_state.get("manual_map", {}))
                st.session_state.players_df = new_df
                st.session_state.draft = new_draft
                players_df = new_df
                draft = new_draft
                sleeper_state["unresolved"] = unresolved
                sleeper_state["status"] = f"OK: {len(new_draft.history)} picks sincronizados"
                sleeper_state["logs"].append(f"[{_now_utc_iso()}] Sync ok: {len(new_draft.history)} picks; unresolved={len(unresolved)}")
                sleeper_state["last_sync_ts"] = now_ts
                sleeper_state["backoff_sec"] = 0
            else:
                raise RuntimeError("Resposta inválida da API de picks")
        except Exception as e:
            sleeper_state["status"] = f"Erro: {e}"
            sleeper_state["logs"].append(f"[{_now_utc_iso()}] Erro ao sync: {e}")
            sleeper_state["last_error_ts"] = now_ts
            new_backoff = sleeper_state.get("backoff_sec", 0) * 2 if sleeper_state.get("backoff_sec", 0) else sleeper_state.get("poll_sec", POLL_SEC_DEFAULT)
            sleeper_state["backoff_sec"] = int(min(MAX_BACKOFF_SEC, max(3, new_backoff)))

    if sleeper_state.get("enabled") and not sleeper_state.get("paused"):
        try:
            st.autorefresh(interval=int(max(3, sleeper_state.get("poll_sec", POLL_SEC_DEFAULT)) * 1000), key="sleeper_autorefresh")
        except Exception:
            pass

# ---------- Controls: Draft Picks (busca rápida) ----------
col_pick, col_info = st.columns([1, 2])
with col_pick:
    r = ((draft.current_pick - 1) // draft.N) + 1
    if r % 2 == 1:
        on_clock_slot = ((draft.current_pick - 1) % draft.N) + 1
    else:
        on_clock_slot = draft.N - ((draft.current_pick - 1) % draft.N)

    st.subheader(f"🕒 Agora: Pick #{draft.current_pick} — Time {on_clock_slot}")

    alive = players_df[~players_df["picked"].astype(bool)].copy()

    q = st.text_input("Buscar jogador")
    pos_filter = st.multiselect("Filtrar posições", ["QB", "RB", "WR", "TE"], default=["RB", "WR", "QB", "TE"])
    if q:
        alive = alive[alive["Player"].str.contains(q, case=False, na=False)]
    if pos_filter:
        alive = alive[alive["POS"].isin(pos_filter)]

    # compute probabilities preview for selector
    preview_df = players_df.copy()
    probs_preview = compute_probabilities(
        preview_df,
        draft,
        chaos=chaos,
        sigma_min=sigma_min,
        run_window=run_window,
        tau_hot=tau_hot,
        run_z_threshold=run_z_threshold,
    )
    preview_small = probs_preview.set_index("player_id")["prob_available_next_pick"] if not probs_preview.empty else pd.Series(dtype=float)

    alive = alive.copy()
    alive["prob"] = alive["player_id"].map(preview_small)
    alive.sort_values(["prob", "ADP"], ascending=[False, True], inplace=True)

    options = alive.head(300)[["player_id", "Player", "POS", "Team", "ADP", "prob"]]
    options_display = options.apply(lambda r: f"{r['Player']} ({r['POS']}-{r['Team']}) • ADP {int(r['ADP']) if pd.notna(r['ADP']) else '-'} • P(sobrar) {r['prob']:.2f}", axis=1)
    label_to_id = dict(zip(options_display, options["player_id"]))

    selected_label = st.selectbox("Selecionar (busca rápida)", list(label_to_id.keys()) if len(label_to_id) > 0 else ["Sem opções"])

    col_btn1, col_btn2 = st.columns(2)
    sleeper_active = st.session_state.get("sleeper", {}).get("enabled") and not st.session_state.get("sleeper", {}).get("paused", False)
    if sleeper_active:
        st.caption("Draft manual desabilitado enquanto o sync do Sleeper está ativo. Pause o sync para draftar manualmente.")
    if col_btn1.button("✅ Draftar (busca)", disabled=bool(sleeper_active)) and selected_label in label_to_id:
        pid = label_to_id[selected_label]
        row = players_df.loc[players_df["player_id"] == pid].iloc[0]
        players_df.loc[players_df["player_id"] == pid, "picked"] = True
        draft.history.append((draft.current_pick, pid, row["POS"]))
        draft.teams[on_clock_slot - 1].players.append((row["Player"], row["POS"]))
        draft.teams[on_clock_slot - 1].taken[row["POS"]] += 1
        st.session_state.pick_log.append(pid)
        draft.current_pick += 1
        st.rerun()

    if col_btn2.button("↩️ Desfazer última pick", disabled=bool(sleeper_active)):
        if st.session_state.pick_log:
            last_pid = st.session_state.pick_log.pop()
            draft.current_pick = max(1, draft.current_pick - 1)
            r_back = ((draft.current_pick - 1) // draft.N) + 1
            if r_back % 2 == 1:
                slot_back = ((draft.current_pick - 1) % draft.N) + 1
            else:
                slot_back = draft.N - ((draft.current_pick - 1) % draft.N)
            row = players_df.loc[players_df["player_id"] == last_pid].iloc[0]
            team_players = draft.teams[slot_back - 1].players
            for idx in range(len(team_players) - 1, -1, -1):
                if team_players[idx][0] == row["Player"]:
                    team_players.pop(idx)
                    break
            draft.teams[slot_back - 1].taken[row["POS"]] = max(0, draft.teams[slot_back - 1].taken[row["POS"]] - 1)
            players_df.loc[players_df["player_id"] == last_pid, "picked"] = False
            if draft.history and draft.history[-1][1] == last_pid:
                draft.history.pop()
            st.rerun()

with col_info:
    st.subheader("🎯 Seu próximo pick")
    K_user = draft.user_next_pick()
    st.metric("Pick global", K_user)

    result_df = compute_probabilities(
        players_df.copy(),
        draft,
        chaos=chaos,
        sigma_min=sigma_min,
        run_window=run_window,
        tau_hot=tau_hot,
        run_z_threshold=run_z_threshold,
    )
    
    # ======= Estado dos jogadores principais por posição (cards) =======
    if "chosen_main_by_pos" not in st.session_state:
        st.session_state["chosen_main_by_pos"] = {"QB": None, "RB": None, "WR": None, "TE": None}

    # ========= Escolha do jogador principal por posição (cards) =========
    st.markdown("#### 🔍 Focar jogador por posição (opcional)")
    def _options_for(pos: str):
        opts = ["(auto)"] + sorted(result_df.loc[result_df["POS"] == pos, "Player"].dropna().unique().tolist())
        cur = st.session_state["chosen_main_by_pos"].get(pos)
        idx = opts.index(cur) if cur in opts else 0
        return opts, idx

    sel_cols = st.columns(4)

    with sel_cols[0]:
        opts_qb, idx_qb = _options_for("QB")
        sel_qb = st.selectbox("QB", opts_qb, index=idx_qb, key="sel_qb")
        st.session_state["chosen_main_by_pos"]["QB"] = None if sel_qb == "(auto)" else sel_qb

    with sel_cols[1]:
        opts_rb, idx_rb = _options_for("RB")
        sel_rb = st.selectbox("RB", opts_rb, index=idx_rb, key="sel_rb")
        st.session_state["chosen_main_by_pos"]["RB"] = None if sel_rb == "(auto)" else sel_rb

    with sel_cols[2]:
        opts_wr, idx_wr = _options_for("WR")
        sel_wr = st.selectbox("WR", opts_wr, index=idx_wr, key="sel_wr")
        st.session_state["chosen_main_by_pos"]["WR"] = None if sel_wr == "(auto)" else sel_wr

    with sel_cols[3]:
        opts_te, idx_te = _options_for("TE")
        sel_te = st.selectbox("TE", opts_te, index=idx_te, key="sel_te")
        st.session_state["chosen_main_by_pos"]["TE"] = None if sel_te == "(auto)" else sel_te

    # Usa sempre o estado para montar os cards
    chosen_QB = st.session_state["chosen_main_by_pos"]["QB"]
    chosen_RB = st.session_state["chosen_main_by_pos"]["RB"]
    chosen_WR = st.session_state["chosen_main_by_pos"]["WR"]
    chosen_TE = st.session_state["chosen_main_by_pos"]["TE"]

    # ==== Cards de custo por posição ====
    cards = {
        "QB": compute_card_for_pos(result_df, "QB", chosen_QB),
        "RB": compute_card_for_pos(result_df, "RB", chosen_RB),
        "WR": compute_card_for_pos(result_df, "WR", chosen_WR),
        "TE": compute_card_for_pos(result_df, "TE", chosen_TE),
    }
    render_cards(cards)

    # ==== Tabela com seleção direta (REFEITA conforme #8) ====
    st.markdown("**Tabela de probabilidades (vivos):**")

    filtered_df = result_df.copy()
    if pos_filter:
        filtered_df = filtered_df[filtered_df["POS"].isin(pos_filter)]

    show_df = filtered_df.copy()
    if show_df.empty:
        st.info("Sem jogadores para exibir com os filtros atuais.")
    else:
        # Nomes legíveis + arredondamentos
        show_df["ADP"] = pd.to_numeric(show_df["ADP"], errors="coerce").round(1)
        show_df["ADP_STD"] = pd.to_numeric(show_df["ADP_STD"], errors="coerce").round(2)
        show_df["ADP ajustado"] = pd.to_numeric(show_df["mu_adj"], errors="coerce").round(2)
        show_df["Desvio ajustado"] = pd.to_numeric(show_df["sigma_adj"], errors="coerce").round(2)
        show_df["Tier"] = pd.to_numeric(show_df.get("tier"), errors="coerce").astype("Int64")
        show_df["FPTS"] = pd.to_numeric(show_df.get("FPTS"), errors="coerce")

        prob_series = (pd.to_numeric(show_df["prob_available_next_pick"], errors="coerce") * 100).round(0)
        prob_series = prob_series.clip(0, 100).fillna(0).astype(int)
        show_df["Prob próximo pick (%)"] = prob_series

        # Ordenação base por ADP e nome
        show_df = show_df.sort_values(["ADP", "Player"], ascending=[True, True])

        # ORDEM SOLICITADA (#8)
        cols_order = [
            "player_id", "Player", "POS", "Tier", "FPTS", "ADP",
            "Prob próximo pick (%)", "Selecionar", "ADP_STD", "ADP ajustado", "Desvio ajustado"
        ]
        # Garante coluna Selecionar
        show_df["Selecionar"] = False

        show_df = show_df[[c for c in cols_order if c in show_df.columns]]
        show_df = show_df.set_index("player_id", drop=True)

        edited = st.data_editor(
            show_df,
            use_container_width=True,
            height=520,
            key="table_editor_right",
            hide_index=True,
            column_config={
                "ADP": st.column_config.NumberColumn(format="%.1f"),
                "FPTS": st.column_config.NumberColumn(format="%.1f"),
                "ADP_STD": st.column_config.NumberColumn(format="%.2f"),
                "ADP ajustado": st.column_config.NumberColumn(format="%.2f"),
                "Desvio ajustado": st.column_config.NumberColumn(format="%.2f"),
                "Prob próximo pick (%)": st.column_config.NumberColumn(format="%d%%"),
                "Selecionar": st.column_config.CheckboxColumn(help="Marque o jogador que você quer draftar agora"),
            },
            disabled=["Player", "POS", "Tier", "FPTS", "ADP", "ADP_STD", "ADP ajustado", "Desvio ajustado", "Prob próximo pick (%)"],
        )

        col_tbl_btn1, col_tbl_btn2, col_tbl_btn3 = st.columns([1, 1, 1])

        if col_tbl_btn1.button("✅ Draftar selecionado (tabela)", disabled=bool(sleeper_active)):
            sel_rows = edited[edited["Selecionar"] == True] if isinstance(edited, pd.DataFrame) else pd.DataFrame()
            if not sel_rows.empty:
                pid = sel_rows.index[0]
                row = players_df.loc[players_df["player_id"] == pid].iloc[0]
                r_cur = ((draft.current_pick - 1) // draft.N) + 1
                if r_cur % 2 == 1:
                    on_slot2 = ((draft.current_pick - 1) % draft.N) + 1
                else:
                    on_slot2 = draft.N - ((draft.current_pick - 1) % draft.N)
                players_df.loc[players_df["player_id"] == pid, "picked"] = True
                draft.history.append((draft.current_pick, pid, row["POS"]))
                draft.teams[on_slot2 - 1].players.append((row["Player"], row["POS"]))
                draft.teams[on_slot2 - 1].taken[row["POS"]] += 1
                st.session_state.pick_log.append(pid)
                draft.current_pick += 1
                st.rerun()

        if col_tbl_btn2.button("↩️ Desfazer (tabela)", disabled=bool(sleeper_active)):
            if st.session_state.pick_log:
                last_pid = st.session_state.pick_log.pop()
                draft.current_pick = max(1, draft.current_pick - 1)
                r_back = ((draft.current_pick - 1) // draft.N) + 1
                if r_back % 2 == 1:
                    slot_back = ((draft.current_pick - 1) % draft.N) + 1
                else:
                    slot_back = draft.N - ((draft.current_pick - 1) % draft.N)
                row = players_df.loc[players_df["player_id"] == last_pid].iloc[0]
                team_players = draft.teams[slot_back - 1].players
                for idx in range(len(team_players) - 1, -1, -1):
                    if team_players[idx][0] == row["Player"]:
                        team_players.pop(idx)
                        break
                draft.teams[slot_back - 1].taken[row["POS"]] = max(0, draft.teams[slot_back - 1].taken[row["POS"]] - 1)
                players_df.loc[players_df["player_id"] == last_pid, "picked"] = False
                if draft.history and draft.history[-1][1] == last_pid:
                    draft.history.pop()
                st.rerun()

        # >>> Definir como principal do card
        if col_tbl_btn3.button("⭐ Definir como principal do card"):
            sel_rows = edited[edited["Selecionar"] == True] if isinstance(edited, pd.DataFrame) else pd.DataFrame()
            if not sel_rows.empty:
                pid = sel_rows.index[0]
                row = players_df.loc[players_df["player_id"] == pid].iloc[0]
                pos_sel = str(row["POS"])
                name_sel = str(row["Player"])
                if "chosen_main_by_pos" not in st.session_state:
                    st.session_state["chosen_main_by_pos"] = {"QB": None, "RB": None, "WR": None, "TE": None}
                if pos_sel in st.session_state["chosen_main_by_pos"]:
                    st.session_state["chosen_main_by_pos"][pos_sel] = name_sel
                st.rerun()

# ---------- Board (Times x Rounds) ----------
st.markdown("---")
st.subheader("📋 Board (Times x Rounds)")

max_picks = max(draft.current_pick - 1, 0)
rounds_completed = ((max_picks) // draft.N) + (1 if (max_picks % draft.N) else 0)
rounds_to_show = max(8, rounds_completed + 1)

cols_board = st.columns(int(draft.N))
# Build slot names once
slot_names = {}
sstate = st.session_state.get("sleeper", {})
if sstate.get("enabled") and sstate.get("league_id") and sstate.get("draft_id"):
    try:
        slot_names = build_slot_names(sstate.get("draft_id"), sstate.get("league_id"))
    except Exception:
        slot_names = {}
for i in range(int(draft.N)):
    with cols_board[i]:
        title = slot_names.get(i+1) or f"Time {i+1}"
        st.markdown(f"**{title}**")
        team_players = draft.teams[i].players
        for r in range(rounds_to_show):
            if r < len(team_players):
                nm, ps = team_players[r]
                st.markdown(f"R{r+1}: **{nm}** ({ps})")
            else:
                st.markdown(f"R{r+1}: —")

st.markdown(
    "<small>Coloque o arquivo `adp_app_table.csv` no caminho padrão (ou faça upload na barra lateral). O app recalcula as probabilidades a cada pick conforme runs e necessidades dos adversários.</small>",
    unsafe_allow_html=True,
)
