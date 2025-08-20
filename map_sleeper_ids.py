import argparse
import json
import re
import sys
from typing import Dict, List, Tuple

import pandas as pd
import requests

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
HTTP_TIMEOUT_SEC = 10.0


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_sleeper_indexes(players: Dict) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]], Dict[Tuple[str, str, str], List[str]], Dict[str, Dict]]:
    # Returns multiple indexes + raw lookup by id
    name_pos_to_ids: Dict[Tuple[str, str], List[str]] = {}
    name_to_ids: Dict[str, List[str]] = {}
    last_team_pos_to_ids: Dict[Tuple[str, str, str], List[str]] = {}
    id_to_player: Dict[str, Dict] = {}

    for sid, p in players.items():
        if not isinstance(p, dict):
            continue
        pos = (p.get("position") or "").upper()
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        full_name = p.get("full_name") or p.get("search_full_name") or ""
        team = (p.get("team") or p.get("team_abbr") or "").upper()
        nid = str(p.get("player_id") or sid)
        norm = normalize_name(full_name)
        if not norm:
            continue
        id_to_player[nid] = p

        name_pos_to_ids.setdefault((norm, pos), []).append(nid)
        name_to_ids.setdefault(norm, []).append(nid)
        last = norm.split(" ")[-1]
        if last:
            last_team_pos_to_ids.setdefault((last, team, pos), []).append(nid)

    return name_pos_to_ids, name_to_ids, last_team_pos_to_ids, id_to_player


def choose_unique(candidates: List[str]) -> str | None:
    if len(candidates) == 1:
        return candidates[0]
    return None


def match_row(row: pd.Series,
              name_pos_to_ids: Dict[Tuple[str, str], List[str]],
              name_to_ids: Dict[str, List[str]],
              last_team_pos_to_ids: Dict[Tuple[str, str, str], List[str]],
              id_to_player: Dict[str, Dict]) -> str | None:
    fp_name = str(row.get("Player", ""))
    pos = str(row.get("POS", "")).upper()
    team = str(row.get("Team", "")).upper()
    norm = normalize_name(fp_name)
    if not norm or pos not in {"QB", "RB", "WR", "TE"}:
        return None

    # 1) Exact name+pos
    sid = choose_unique(name_pos_to_ids.get((norm, pos), []))
    if sid:
        return sid

    # 2) Name only filtered by pos
    ids_by_name = name_to_ids.get(norm, [])
    if ids_by_name:
        filtered = [i for i in ids_by_name if (id_to_player.get(i, {}).get("position") or "").upper() == pos]
        sid = choose_unique(filtered)
        if sid:
            return sid

    # 3) Last name + team + pos
    last = norm.split(" ")[-1]
    if last:
        sid = choose_unique(last_team_pos_to_ids.get((last, team, pos), []))
        if sid:
            return sid

    # 4) Name only filtered by team
    if ids_by_name:
        filtered = [i for i in ids_by_name if (str(id_to_player.get(i, {}).get("team") or "").upper()) == team]
        sid = choose_unique(filtered)
        if sid:
            return sid

    # 5) Last name + pos (unique)
    if last and ids_by_name:
        # restrict to same last name set across that pos
        same_last = [i for i in ids_by_name if (id_to_player.get(i, {}).get("position") or "").upper() == pos]
        sid = choose_unique(same_last)
        if sid:
            return sid

    return None


def main():
    parser = argparse.ArgumentParser(description="Append Sleeper IDs to FantasyPros ADP CSV")
    parser.add_argument("--input", default="tables/adp_app_table.csv", help="Path to input CSV")
    parser.add_argument("--output", default="tables/adp_app_table_with_sleeper.csv", help="Path to write enriched CSV")
    parser.add_argument("--unmatched", default="tables/adp_unmatched.csv", help="Path to write unmatched rows report")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        resp = requests.get(SLEEPER_PLAYERS_URL, timeout=HTTP_TIMEOUT_SEC)
        resp.raise_for_status()
        sleeper_players = resp.json()
        if not isinstance(sleeper_players, dict):
            raise ValueError("Unexpected Sleeper players payload")
    except Exception as e:
        print(f"Failed to fetch Sleeper players: {e}", file=sys.stderr)
        sys.exit(2)

    name_pos_to_ids, name_to_ids, last_team_pos_to_ids, id_to_player = build_sleeper_indexes(sleeper_players)

    df = df.copy()
    df["sleeper_id"] = None
    unmatched_rows: List[Dict] = []

    for idx, row in df.iterrows():
        sid = match_row(row, name_pos_to_ids, name_to_ids, last_team_pos_to_ids, id_to_player)
        if sid:
            df.at[idx, "sleeper_id"] = sid
        else:
            unmatched_rows.append({
                "Player": row.get("Player"),
                "POS": row.get("POS"),
                "Team": row.get("Team"),
            })

    try:
        df.to_csv(args.output, index=False)
        print(f"Wrote enriched CSV: {args.output}")
    except Exception as e:
        print(f"Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        if unmatched_rows:
            pd.DataFrame(unmatched_rows).to_csv(args.unmatched, index=False)
            print(f"Wrote unmatched report: {args.unmatched} ({len(unmatched_rows)} rows)")
        else:
            print("All rows matched.")
    except Exception as e:
        print(f"Failed to write unmatched report: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


