#!/usr/bin/env python3

"""
Script making the pairings for FIDE binance tournament
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import requests
import os
import time
import re
import sqlite3
import sys

from enum import IntEnum
from argparse import RawTextHelpFormatter
from dataclasses import dataclass
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Dict, List, Optional, Tuple

#############
# Constants #
#############

load_dotenv()

PLAYER_REGEX = re.compile(
    r"^(?P<table_number>\d+)"
    + r"\s+"
    + r"(?P<player1>\w+)"
    + r"\s+.+\s+"
    + r"(?P<player2>\w+)"
    + r"\s+\d+"
)

G_DOC_PATH = "round_{}.txt"
LOG_PATH = "pair.log"

BASE = "https://lichess.org"
if __debug__:
    BASE = "http://localhost:9663"
PAIRING_API = BASE + "/api/challenge/admin/{}/{}"
LOOKUP_API = BASE + "/games/export/_ids"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('TOKEN')}",
    "Accept": "application/x-ndjson",
}

RETRY_STRAT = Retry(
    total=5,
    backoff_factor=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
ADAPTER = HTTPAdapter(max_retries=RETRY_STRAT)


########
# Logs #
########

log = logging.getLogger("pair")
log.setLevel(logging.DEBUG)

req_log = logging.getLogger('urllib3')
req_log.setLevel(logging.WARN)

format_string = "%(asctime)s | %(levelname)-8s | %(message)s"

# 12_500_000 bytes = 12.5Mb
handler = logging.handlers.RotatingFileHandler(
    LOG_PATH, maxBytes=12_500_000, backupCount=3, encoding="utf8"
)
handler.setFormatter(logging.Formatter(format_string))
handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(handler)

handler_2 = logging.StreamHandler(sys.stdout)
handler_2.setFormatter(logging.Formatter(format_string))
handler_2.setLevel(logging.INFO)
if __debug__:
    handler_2.setLevel(logging.DEBUG)
logging.getLogger().addHandler(handler_2)

###########
# Classes #
###########


class GameResult(IntEnum):
    BLACK_WINS = 0
    WHITE_WINS = 1
    DRAW = 2

    @staticmethod
    def from_game(game: Dict[str, str]) -> Optional[GameResult]:
        winner = game.get("winner")
        status = game["status"]
        if winner == "white":
            return GameResult.WHITE_WINS
        elif winner == "black":
            return GameResult.BLACK_WINS
        elif winner is None and status == "draw" or status == "stalemate":
            return GameResult.DRAW
        return None


class Db:
    def __init__(self: Db) -> None:
        self.con = sqlite3.connect("FIDE_binance.db", isolation_level=None)
        self.cur = self.con.cursor()

    def create_db(self: Db) -> None:
        # Since the event is divided in two parts, `round_nb` will first indicate the round_nb number in the round-robin then advancement in the knockdown event
        # `result` 0 = black wins, 1 = white wins, 2 = draw, 3 = unknown (everything else)
        # `rowId` is the primary key and is create silently
        self.cur.execute(
            """CREATE TABLE rounds
               (
               white_player description VARCHAR(30) NOT NULL, 
               black_player description VARCHAR(30) NOT NULL, 
               lichess_game_id CHAR(8), 
               result INT,
               round_nb INT)"""
        )

    def show(self: Db) -> None:
        tables = self.cur.execute(
            """SELECT name 
            FROM sqlite_master 
            WHERE type ='table' AND 
            name NOT LIKE 'sqlite_%';"""
        )
        clean_tables = [t[0] for t in tables]
        log.info(f"List of table's names: {clean_tables}")
        for table in clean_tables:
            struct = self.cur.execute(
                f"PRAGMA table_info({table})"
            )  # discouraged but qmark does not work here for some reason
            log.info(f"{table} structure: {[t for t in struct]}")
            rows = self.cur.execute(f"SELECT * from {table}")
            log.info(f"{table} rows: {[t for t in rows]}")

    def add_players(self: Db, pair: Pair, round_nb: int) -> None:
        self.cur.execute(
            """INSERT INTO rounds
            (
            white_player,
            black_player,
            round_nb
            ) VALUES (?, ?, ?)
            """,
            (pair.white_player, pair.black_player, round_nb),
        )

    def get_unpaired_players(self: Db, round_nb: int) -> List[Tuple[int, Pair]]:
        raw_data = list(
            self.cur.execute(
                """SELECT 
                    rowId, 
                    white_player,
                    black_player
                    FROM rounds
                    WHERE lichess_game_id IS NULL AND round_nb = ?
                    """,
                (round_nb,),
            )
        )
        log.info(f"Round {round_nb}, {len(raw_data)} games to be created")
        return [
            (int(row_id), Pair(white_player, black_player))
            for row_id, white_player, black_player in raw_data
        ]

    def add_lichess_game_id(self: Db, row_id: int, game_id: str) -> None:
        self.cur.execute(
            """UPDATE rounds
                SET lichess_game_id = ?
                WHERE
                rowId = ?""",
            (game_id, row_id),
        )

    def get_unfinished_games(self: Db, round_nb: int) -> Dict[str, int]:
        raw_data = list(
            self.cur.execute(
                """SELECT 
                    rowId, 
                    lichess_game_id
                    FROM rounds
                    WHERE lichess_game_id IS NOT NULL AND result IS NULL AND round_nb = ?""",
                (round_nb,),
            )
        )
        log.info(f"Round {round_nb}, {len(raw_data)} games unfinished")
        return {game_id: int(row_id) for row_id, game_id in raw_data}

    def get_game_ids(self: Db, round_nb: int) -> str:
        raw_data = list(
            self.cur.execute(
                """SELECT 
                    lichess_game_id
                    FROM rounds
                    WHERE lichess_game_id IS NOT NULL AND round_nb = ?
                    """,
                (round_nb,),
            )
        )
        log.info(f"Round {round_nb}, {len(raw_data)} games started")
        log.debug(raw_data)
        return " ".join((x[0] for x in raw_data))

    def add_game_result(self: Db, row_id: int, result: GameResult) -> None:
        self.cur.execute(
            """UPDATE rounds
                SET result = ?
                WHERE
                rowId = ?""",
            (result, row_id),
        )


class FileHandler:
    def __init__(self: FileHandler, db: Db) -> None:
        self.db = db

    def get_pairing(self: FileHandler, round_nb: int) -> List[Pair]:
        pairs: List[Pair] = []
        with open(G_DOC_PATH.format(round_nb)) as f:
            for line in (line.strip() for line in f if line.strip()):
                match = PLAYER_REGEX.match(line)
                if match is None:
                    log.warn(f"Failed to line: {line}")
                    continue
                log.debug(match.groups())
                table_number = int(match.group("table_number"))
                player1 = match.group("player1")
                player2 = match.group("player2")
                if int(table_number) % 2:  # odd numbers have white player on left
                    pair = Pair(white_player=player1, black_player=player2)
                else:
                    pair = Pair(white_player=player2, black_player=player1)
                log.debug(pair)
                pairs.append(pair)
        return pairs

    def fetch(self: FileHandler, round_nb: int) -> None:
        for pair in self.get_pairing(round_nb):
            self.db.add_players(pair, round_nb)


@dataclass
class Pair:
    white_player: str
    black_player: str


class Pairing:
    def __init__(self: Pairing, db: Db) -> None:
        self.db = db
        self.http = requests.Session()
        self.http.mount("https://", ADAPTER)
        self.http.mount("http://", ADAPTER)
        self.dep = time.time()

    def tl(self: Pairing) -> float:
        """time elapsed"""
        return time.time() - self.dep

    def pair_all_players(self: Pairing, round_nb: int) -> None:
        for row_id, pair in self.db.get_unpaired_players(round_nb):
            game_id = self.create_game(pair)
            self.db.add_lichess_game_id(row_id, game_id)

    def create_game(self: Pairing, pair: Pair) -> str:
        """Return the lichess game id of the game created"""
        url = PAIRING_API.format(pair.white_player, pair.black_player)
        payload = {
            "rated": "true",
            "clock.limit": 600,
            "clock.increment": 2,
            "color": "white",
            "message": "FIDE Binance: Your game with {opponent} is ready: {game}"
        }
        r = self.http.post(url, data=payload, headers=HEADERS).json()
        log.debug(r)
        return r["game"]["id"]

    def check_all_results(self: Pairing, round_nb: int) -> None:
        games_dic = self.db.get_unfinished_games(round_nb)
        # Not streaming since at most 128 games so ~6s and it avoid parsing ndjson.
        r = self.http.post(
            LOOKUP_API,
            data=",".join(games_dic.keys()),
            headers=HEADERS,
            params={"moves": "false"},
        )
        games = r.text.splitlines()
        log.debug(games)
        for raw_game in games:
            log.debug(raw_game)
            game = json.loads(raw_game)
            result = GameResult.from_game(game)
            id_ = game["id"]
            log.info(f"Game {id_}, result: {result}")
            if result is not None:
                self.db.add_game_result(id_, result)

    def test(self: Pairing):
        games_id = ["11tHUbnm", "ETSYCv5R", "KVPzep34", "uXxcDewp"]
        # Not streaming since at most 128 games so ~6s and it avoid parsing ndjson.
        r = self.http.post(
            LOOKUP_API,
            data=",".join(games_id),
            headers=HEADERS,
            params={"moves": "false"},
        )
        games = r.text.splitlines()
        log.debug(games)
        for raw_game in games:
            log.debug(raw_game)
            game = json.loads(raw_game)
            result = GameResult.from_game(game)
            id_ = game["id"]
            log.info(f"Game {id_}, result: {result}")


#############
# Functions #
#############


def create_db() -> None:
    """Setup the sqlite database, should be run once first when getting the script"""
    db = Db()
    db.create_db()


def show() -> None:
    """Show the current state of the database. For debug purpose only"""
    db = Db()
    db.show()


def test() -> None:
    db = Db()
    p = Pairing(db)
    p.test()


def fetch(round_nb: int) -> None:
    """Takes the raw dump from the `G_DOC_PATH` copied document and store the pairings in the db, without launching the challenges"""
    f = FileHandler(Db())
    f.fetch(round_nb)


def pair(round_nb: int) -> None:
    """Create a challenge for every couple of players that has not been already paired"""
    db = Db()
    p = Pairing(db)
    p.pair_all_players(round_nb)


def result(round_nb: int) -> None:
    """Fetch all games from that round_nb, check if they are finished, and print the results"""
    db = Db()
    p = Pairing(db)
    p.check_all_results(round_nb)


def broadcast(round_nb: int) -> None:
    """Return game ids of the round `round_nb` separated by a space"""
    db = Db()
    print(db.get_game_ids(round_nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_usage())
    subparsers = parser.add_subparsers()
    BASIC_COMMANDS = {
        "create_db": create_db,
        "show": show,
        "test": test,
    }
    ROUND_COMMANDS = {
        "fetch": fetch,
        "pair": pair,
        "result": result,
        "broadcast": broadcast,
    }

    def fix(fn, args_fn=None):
        if args_fn is None:
            return lambda args: fn()
        else:
            return lambda args: fn(args_fn(args))

    for name, fn in BASIC_COMMANDS.items():
        p = subparsers.add_parser(name, help=fn.__doc__)
        p.set_defaults(func=fix(fn))

    for name, fn2 in ROUND_COMMANDS.items():
        p = subparsers.add_parser(name, help=fn2.__doc__)
        p.set_defaults(func=fix(fn2, lambda args: args.round_nb))
        p.add_argument(
            "round_nb",
            metavar="ROUND",
            default=0,
            type=int,
            help="The round number related to the action you want to do",
        )

    args = parser.parse_args()
    args.func(args)


########
# Main #
########

if __name__ == "__main__":
    main()
