from __future__ import annotations
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# Get match city from match JSON file
def city(match_json: str | Path) -> str:
    with open(match_json) as fp:
        match_data = json.load(fp)
    return match_data["info"]["city"]


# Get all cities in the dataset
def cities_gen(json_files: Iterable[Path]):
    for json_file in json_files:
        yield city(json_file)


# Toss Result and Decision
# winner = 1 if team_1 wins and 0 if team_0 wins
# decision = 0 if bat first and 1 if field first.
def toss(match_data: dict) -> tuple[int, int | None]:
    winner = int(match_data["info"]["toss"]["winner"] == match_data["info"]["teams"][1])
    if match_data["info"]["toss"]["decision"] == "bat":
        decision = 0
    elif match_data["info"]["toss"]["decision"] == "field":
        decision = 1
    else:
        decision = None
    return winner, decision


# Get score details for an over.
def score_from_over(over: dict):
    runs = 0
    wickets = 0
    deliveries = 0
    extras = 0
    for ball in over["deliveries"]:
        runs += ball["runs"]["total"]
        extras += ball["runs"]["extras"]
        if "wickets" in ball.keys():
            wickets += 1
        deliveries += 1
    return runs, wickets, extras, deliveries


# Get Scores at the end of any over in an innings.
def scorecard_after_over(match_data: dict, innings_num: int = 1, overno: int = 50):
    if len(match_data["innings"]) >= innings_num:
        innings = match_data["innings"][innings_num - 1]
        team = innings["team"]
        overs = [over for over in innings["overs"] if over["over"] < overno]
    else:
        raise KeyError(f"Innings not present: {innings_num}")
    if overno > 0:
        lastovdelv = divmod(score_from_over(overs[-1])[-1], 6)
        score = {
            "team": team,
            "runs": sum(score_from_over(over)[0] for over in overs),
            "wickets": sum(score_from_over(over)[1] for over in overs),
            "overs": (len(overs) + lastovdelv[0] - 1, lastovdelv[1]),
            "extras": sum(score_from_over(over)[2] for over in overs),
        }
    else:
        score = {
            "team": team,
            "runs": 0,
            "wickets": 0,
            "overs": (0, 0),
            "extras": 0,
        }
    return score


# Get final scores for the match
def get_scores(match_data):
    w, d = toss(match_data)
    # We use exclusive or to find the batting first team.
    batfir = int((w or d) and (not w or not d))
    batsec = int(not batfir)
    if len(match_data["innings"]) == 2:
        scores = {
            batfir: scorecard_after_over(match_data, innings_num=1),
            batsec: scorecard_after_over(match_data, innings_num=2),
        }
    else:
        raise ValueError("Match not completed")
    return scores


# Match Result
def results(match_data) -> int | None:
    teams = match_data["info"]["teams"]
    outcome = match_data["info"]["outcome"]
    teams.sort()
    if "winner" in outcome.keys():
        result = int(outcome["winner"] == teams[1])
        return result
    return None
