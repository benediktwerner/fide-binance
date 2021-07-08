#!/usr/bin/env python3

"""
Script making the pairings for FIDE binance tournament
"""

from __future__ import annotations

import argparse
import csv
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

CHALLENGE_SETTINGS = {
    "rated": "true",
    "clock.limit": 600,
    "clock.increment": 2,
    "color": "white",
    "message": "FIDE Binance: Your game with {opponent} is ready: {game}",
}

CSV_DELIM = ";"

LOG_PATH = "out.log"

BASE = "http://localhost:9663" if __debug__ else "https://lichess.org"
PAIRING_API = BASE + "/api/challenge/admin/{}/{}"
LOOKUP_API = BASE + "/games/export/_ids"
DB_FILE = "FIDE_binance_test.db" if __debug__ else "FIDE_binance_prod.db"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('TOKEN_TEST' if __debug__ else 'TOKEN_PROD')}",
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

req_log = logging.getLogger("urllib3")
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
        self.con = sqlite3.connect(DB_FILE, isolation_level=None)
        self.cur = self.con.cursor()

    def create_db(self: Db) -> None:
        # Since the event is divided in two parts, `round_nb` will first indicate the round_nb number in the round-robin then advancement in the knockdown event
        # `result` 0 = black wins, 1 = white wins, 2 = draw
        # `rowId` is the primary key and is create silently
        self.cur.execute(
            """CREATE TABLE rounds (
                white_player description VARCHAR(30) NOT NULL, 
                black_player description VARCHAR(30) NOT NULL, 
                lichess_game_id CHAR(8), 
                result INT,
                round_nb INT
            )"""
        )

    def show(self: Db) -> None:
        tables = self.cur.execute(
            """SELECT name 
                FROM sqlite_master 
                WHERE type ='table'
                AND name NOT LIKE 'sqlite_%'
            """
        )
        table_names = [t[0] for t in tables]
        log.info(f"Tables: {table_names}")
        for table in table_names:
            struct = self.cur.execute(
                f"PRAGMA table_info({table})"
            )  # discouraged but qmark does not work here for some reason
            log.info(f"'{table}' structure:")
            for col in struct:
                log.info(f"    {col}")
            rows = self.cur.execute(f"SELECT * from {table}")
            log.info(f"'{table}' rows:")
            for row in rows:
                log.info(f"    {row}")

    def remove_round(self: Db, round_nb: int, force=False) -> None:
        force_select = "" if force else "AND lichess_game_id is NULL"
        self.cur.execute(
            f"""DELETE FROM rounds
                WHERE round_nb = ? {force_select}
            """,
            (round_nb,),
        )

    def add_players(self: Db, pair: Pair, round_nb: int) -> None:
        self.cur.execute(
            """INSERT INTO rounds (
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
        return [
            (int(row_id), Pair(white_player, black_player))
            for row_id, white_player, black_player in raw_data
        ]

    def add_lichess_game_id(self: Db, row_id: int, game_id: str) -> None:
        self.cur.execute(
            """UPDATE rounds
                SET lichess_game_id = ?
                WHERE rowId = ?
            """,
            (game_id, row_id),
        )

    def get_unfinished_games(self: Db, round_nb: int) -> List[str]:
        raw_data = list(
            self.cur.execute(
                """SELECT 
                    lichess_game_id
                   FROM rounds
                   WHERE lichess_game_id IS NOT NULL
                   AND result IS NULL
                   AND round_nb = ?
                """,
                (round_nb,),
            )
        )
        return [r[0] for r in raw_data]

    def get_game_ids(self: Db, round_nb: int) -> List[str]:
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
        return [x[0] for x in raw_data]

    def add_game_result(self: Db, game_id: str, result: GameResult) -> None:
        self.cur.execute(
            """UPDATE rounds
                SET result = ?
                WHERE lichess_game_id = ?
            """,
            (result, game_id),
        )


class FileHandler:
    def __init__(self: FileHandler, db: Optional[Db] = None) -> None:
        self.db = Db() if db is None else db

    def __read_pairings_txt(self: FileHandler, path: str) -> List[Pair]:
        pairs: List[Pair] = []
        with open(path) as f:
            for line in (line.strip() for line in f if line.strip()):
                match = PLAYER_REGEX.match(line)
                if match is None:
                    log.warn(f"Failed to match line: {line}")
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

    def __read_pairings_csv(self: FileHandler, path: str) -> List[Pair]:
        pairs: List[Pair] = []
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=CSV_DELIM)
            for row in reader:
                log.debug(row)
                table_number, player1, player2 = row
                if int(table_number) % 2:  # odd numbers have white player on left
                    pair = Pair(white_player=player1, black_player=player2)
                else:
                    pair = Pair(white_player=player2, black_player=player1)
                log.debug(pair)
                pairs.append(pair)
        return pairs

    def insert(self: FileHandler, round_nb: int, path: str) -> None:
        if path.endswith(".txt"):
            pairs = self.__read_pairings_txt(path)
        elif path.endswith(".csv"):
            pairs = self.__read_pairings_csv(path)
        else:
            return log.error(f"Unsupported pairings file: {path}. Must be .txt or .csv")

        for pair in pairs:
            self.db.add_players(pair, round_nb)

        log.info(f"Round {round_nb}: {len(pairs)} pairings inserted")


@dataclass
class Pair:
    white_player: str
    black_player: str


class Pairing:
    def __init__(self: Pairing, db: Optional[Db] = None) -> None:
        self.db = Db() if db is None else db
        self.http = requests.Session()
        self.http.mount("https://", ADAPTER)
        self.http.mount("http://", ADAPTER)
        self.dep = time.time()

    def pair_all_players(self: Pairing, round_nb: int) -> None:
        unpaired = self.db.get_unpaired_players(round_nb)
        log.info(f"Round {round_nb}: {len(unpaired)} games to be created")
        for row_id, pair in unpaired:
            game_id = self.create_game(pair)
            self.db.add_lichess_game_id(row_id, game_id)

    def create_game(self: Pairing, pair: Pair) -> str:
        """Returns the lichess game id of the game created"""
        url = PAIRING_API.format(pair.white_player, pair.black_player)
        r = self.http.post(url, data=CHALLENGE_SETTINGS, headers=HEADERS).json()
        log.debug(r)
        return r["game"]["id"]

    def check_all_results(self: Pairing, round_nb: int) -> None:
        games = self.db.get_unfinished_games(round_nb)
        log.info(f"Round {round_nb}: Checking {len(games)} previously unfinished games")
        # Not streaming since at most 128 games so ~6s and it avoid parsing ndjson.
        r = self.http.post(
            LOOKUP_API,
            data=",".join(games),
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
            log.info(f"Game {id_}: {result!s}")
            if result is not None:
                self.db.add_game_result(id_, result)

    def test(self: Pairing):
        games_id = ["mPxblLH3", "Hu5hui7d", "KVPzep34", "uXxcDewp"]
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
            log.info(f"Game {id_}: {result!s}")


#############
# Functions #
#############


def create_db() -> None:
    """Setup the sqlite database, should be run once first when getting the script"""
    log.debug("cmd: create_db")
    Db().create_db()


def show() -> None:
    """Show the current state of the database. For debug purpose only"""
    log.debug("cmd: show")
    Db().show()


def test() -> None:
    log.debug("cmd: test")
    Pairing().test()


def insert(round_nb: int, path: str) -> None:
    """Store pairings from a .txt or .csv file in the db, without launching the challenges"""
    log.debug(f"cmd: fetch {round_nb} {path}")
    FileHandler().insert(round_nb, path)


def pair(round_nb: int) -> None:
    """Create a challenge for every couple of players that has not been already paired"""
    log.debug(f"cmd: pair {round_nb}")
    Pairing().pair_all_players(round_nb)


def result(round_nb: int) -> None:
    """Fetch all games from that round_nb, check if they are finished, and print the results"""
    log.debug(f"cmd: result {round_nb}")
    Pairing().check_all_results(round_nb)


def broadcast(round_nb: int) -> None:
    """Return game ids of the round `round_nb` separated by a space"""
    log.debug(f"cmd: broadcast {round_nb}")
    game_ids = Db().get_game_ids(round_nb)
    log.info(f"Round {round_nb}: {len(game_ids)} games started")
    log.info(f"Game ids: {' '.join(game_ids)}")


def reset(round_nb: int, force=False) -> None:
    """Reset a round by removing all its pairings. Use `force` to remove pairings with created games."""
    log.debug(f"cmd: reset {round_nb} force={force}")
    Db().remove_round(round_nb, force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda **kwargs: parser.print_usage())
    subparsers = parser.add_subparsers()
    BASIC_COMMANDS = {
        "create_db": create_db,
        "show": show,
        "test": test,
    }
    ROUND_COMMANDS = {
        "insert": insert,
        "pair": pair,
        "result": result,
        "broadcast": broadcast,
        "reset": reset,
    }

    for name, fn in BASIC_COMMANDS.items():
        p = subparsers.add_parser(name, help=fn.__doc__)
        p.set_defaults(func=fn)

    for name, fn2 in ROUND_COMMANDS.items():
        p = subparsers.add_parser(name, help=fn2.__doc__)
        p.set_defaults(func=fn2)
        p.add_argument(
            "round_nb",
            metavar="ROUND",
            default=0,
            type=int,
            help="The round number related to the action you want to do",
        )

    subparsers.choices["insert"].add_argument(
        "path", help="File to read pairings from (.txt or .csv)",
    )

    subparsers.choices["reset"].add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Also delete pairings if their games were already created",
    )

    args = parser.parse_args()
    func = args.func
    params = vars(args)
    if "func" in params:
        del params["func"]
    func(**params)


########
# Main #
########

if __name__ == "__main__":
    main()
