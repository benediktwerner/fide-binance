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

from dataclasses import dataclass
from dotenv import load_dotenv
from enum import IntEnum
from requests import Response
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Dict, List, Optional, Tuple, Set

#############
# Constants #
#############

load_dotenv()

PLAYER_REGEX = re.compile(
    r"^(?P<table_number>\d+)\t"
    + r"[^\t]+\t"  # Name 1
    + r"\d+\t"  # Elo 1
    + r"(?P<player1>[a-zA-Z0-9_-]+)\t"
    + r"[^\t]+\t"  # Score
    + r"[^\t]+\t"  # Name 2
    + r"\d+\t"  # Elo 2
    + r"(?P<player2>[a-zA-Z0-9_-]+)"
)

GAME_SETTINGS = {
    "rated": "true",
    "clock.limit": 600,
    "clock.increment": 2,
    "message": "FIDE Binance: Your game is ready: {game}",
}

START_CLOCKS_AFTER_SECS = 60

CSV_DELIM = ";"
TOKENS_PATH = "tokens.txt"

LOG_PATH = "out.log"

BASE = "http://localhost:9663" if __debug__ else "https://lichess.org"
BULK_CREATE_API = BASE + "/api/bulk-pairing"
BULK_START_CLOCKS_API = BASE + "/api/bulk-pairing/{}/start-clocks"
CHALLENGE_API = BASE + "/api/challenge/admin/{}/{}"
CHALLENGE_START_CLOCKS_API = BASE + "/api/challenge/{}/start-clocks"
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


@dataclass
class Pair:
    white_player: str
    black_player: str

    def __init__(self: Pair, white_player: str, black_player: str):
        self.white_player = white_player.lower()
        self.black_player = black_player.lower()


@dataclass
class Game:
    round_nb: int
    pair: Pair
    game_id: str
    bulk_id: Optional[str] = None


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
                round_nb INT,
                bulk_id CHAR(8)
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
    
    def show_round(self: Db, round_nb: int) -> None:
        raw_data = self.cur.execute(
            """SELECT 
                    rowId,
                    white_player,
                    black_player,
                    lichess_game_id,
                    result,
                    bulk_id
                   FROM rounds
                   WHERE round_nb = ?
                """,
            (round_nb,),
        )
        log.info(f"No  |DB Row{'White':>30} vs {'Black':30} Game Id  Bulk Id  State")
        for i, (rowId, white, black, game_id, result, bulk_id) in enumerate(raw_data):
            if game_id is None:
                state = "Not created"
            elif result is None:
                state = "Started"
            else:
                state = ["Black wins 0-1", "White wins 1-0", "Draw       ½-½"][result]
            game_id = game_id or "--------"
            bulk_id = bulk_id or "--------"
            log.info(f"{i:>4}|{rowId:<5} {white:>30} vs {black:30} {game_id} {bulk_id} {state}")

    def remove_round(self: Db, round_nb: int, force=False) -> None:
        force_select = "" if force else "AND lichess_game_id is NULL"
        self.cur.execute(
            f"""DELETE FROM rounds
                WHERE round_nb = ? {force_select}
            """,
            (round_nb,),
        )
        leftover = len(
            list(
                self.cur.execute(
                    """SELECT 1
                   FROM rounds
                   WHERE round_nb = ?
                """,
                    (round_nb,),
                )
            )
        )
        if leftover > 0:
            log.info(f"Round {round_nb}: {leftover} remaining pairings")
        else:
            log.info(f"Round {round_nb}: Removed all pairings")

    def add_pairs(self: Db, round_nb: int, pairs: List[Pair]) -> None:
        self.cur.executemany(
            """INSERT INTO rounds (
                white_player,
                black_player,
                round_nb
               ) VALUES (?, ?, ?)
            """,
            [(pair.white_player, pair.black_player, round_nb) for pair in pairs],
        )

    def get_pairs(self: Db, round_nb: int) -> List[Pair]:
        raw_data = self.cur.execute(
            """SELECT 
                    white_player,
                    black_player
                   FROM rounds
                   WHERE round_nb = ?
                """,
            (round_nb,),
        )
        return [
            Pair(white_player, black_player) for white_player, black_player in raw_data
        ]

    def get_unpaired_players(self: Db, round_nb: int) -> List[Pair]:
        raw_data = self.cur.execute(
            """SELECT 
                    white_player,
                    black_player
                   FROM rounds
                   WHERE lichess_game_id IS NULL AND round_nb = ?
                """,
            (round_nb,),
        )
        return [
            Pair(white_player, black_player) for white_player, black_player in raw_data
        ]

    def add_lichess_game(self: Db, game: Game) -> None:
        self.cur.execute(
            """UPDATE rounds SET
                bulk_id = ?,
                lichess_game_id = ?,
                result = NULL
               WHERE white_player = ?
               AND black_player = ?
               AND round_nb = ?
            """,
            (
                game.bulk_id,
                game.game_id,
                game.pair.white_player,
                game.pair.black_player,
                game.round_nb,
            ),
        )

    def add_lichess_games(self: Db, games: List[Game]) -> None:
        self.cur.executemany(
            """UPDATE rounds SET
                bulk_id = ?,
                lichess_game_id = ?,
                result = NULL
               WHERE white_player = ?
               AND black_player = ?
               AND round_nb = ?
            """,
            [
                (
                    game.bulk_id,
                    game.game_id,
                    game.pair.white_player,
                    game.pair.black_player,
                    game.round_nb,
                )
                for game in games
            ],
        )

    def get_unfinished_games(self: Db, round_nb: int) -> List[Game]:
        raw_data = self.cur.execute(
            """SELECT 
                    white_player,
                    black_player,
                    lichess_game_id,
                    bulk_id
                   FROM rounds
                   WHERE lichess_game_id IS NOT NULL
                   AND result IS NULL
                   AND round_nb = ?
                """,
            (round_nb,),
        )
        return [
            Game(round_nb, Pair(white, black), game_id, bulk_id)
            for white, black, game_id, bulk_id in raw_data
        ]

    def get_game_ids(self: Db, round_nb: int) -> List[str]:
        raw_data = self.cur.execute(
            """SELECT 
                    lichess_game_id
                   FROM rounds
                   WHERE lichess_game_id IS NOT NULL AND round_nb = ?
                """,
            (round_nb,),
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

    def read_pairings_txt(self: FileHandler, path: str) -> List[Pair]:
        pairs: List[Pair] = []
        with open(path) as f:
            for line in (line.strip() for line in f if line.strip()):
                match = PLAYER_REGEX.match(line)
                if match is None:
                    log.warning(f"Failed to match line: {line}")
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

    def read_pairings_csv(self: FileHandler, path: str) -> List[Pair]:
        pairs: List[Pair] = []
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=CSV_DELIM)
            for row in reader:
                log.debug(row)
                table_number, _name1, _elo1, player1, _sep, _name2, _elo2, player2 = row
                if int(table_number) % 2:  # odd numbers have white player on left
                    pair = Pair(white_player=player1, black_player=player2)
                else:
                    pair = Pair(white_player=player2, black_player=player1)
                log.debug(pair)
                pairs.append(pair)
        return pairs

    def insert(
        self: FileHandler, round_nb: int, path: str, force: bool = False
    ) -> None:
        if path.endswith(".txt"):
            pairs = self.read_pairings_txt(path)
        elif path.endswith(".csv"):
            pairs = self.read_pairings_csv(path)
        else:
            return log.error(f"Unsupported pairings file: {path}. Must be .txt or .csv")

        existing = set()
        for pair in self.db.get_pairs(round_nb):
            existing.add(pair.white_player)
            existing.add(pair.black_player)
        filtered_pairs = [
            pair
            for pair in pairs
            if not pair.white_player in existing or pair.black_player in existing
        ]
        if len(filtered_pairs) != len(pairs):
            if force:
                log.warning("Force inserting pairs for already paired players")
            else:
                log.warning("Skipped inserting players which are already paired:")
                for pair in set(pairs) - set(filtered_pairs):
                    log.warning(f"  {pair}")
                log.warning("Use --force to insert them anyway")
                pairs = filtered_pairs

        self.db.add_pairs(round_nb, pairs)
        log.info(f"Round {round_nb}: {len(pairs)} pairings inserted")
        log.info(f"Total pairs: {len(self.db.get_pairs(round_nb))}")


def check_response(response: Response, error_msg: str) -> bool:
    if not response.ok:
        log.error(error_msg)
        log.error(f"{response.request.method} {response.url} => {response.status_code} {response.reason}")
        try:
            log.error(f"JSON response: {response.json()}")
        except Exception:
            log.error(f"Text response: {response.text}")
        return False
    return True


class Tokens:
    def __init__(self: Tokens):
        self.tokens: Dict[str, str] = {}
        with open(TOKENS_PATH) as f:
            for line in f:
                name, token = (x.strip() for x in line.split(":"))
                self.tokens[name.lower()] = token

    def get_tokens(self: Tokens, pair: Pair) -> Optional[Tuple[str, str]]:
        t1 = self.tokens.get(pair.white_player)
        t2 = self.tokens.get(pair.black_player)
        if t1 is None:
            log.error(f"No token for: {pair.white_player}")
            return None
        if t2 is None:
            log.error(f"No token for: {pair.black_player}")
            return None
        return t1, t2


class Pairing:
    def __init__(self: Pairing, db: Optional[Db] = None) -> None:
        self.db = Db() if db is None else db
        self.tokens = Tokens()
        self.http = requests.Session()
        self.http.mount("https://", ADAPTER)
        self.http.mount("http://", ADAPTER)

    def create_games_bulk(self: Pairing, round_nb: int) -> None:
        unpaired = self.db.get_unpaired_players(round_nb)
        log.info(f"Round {round_nb}: {len(unpaired)} games to be created")

        players = []

        for pair in unpaired:
            token_pair = self.tokens.get_tokens(pair)
            if token_pair is None:
                return
            players.append(":".join(token_pair))

        now = int(time.time()) * 1000
        data = GAME_SETTINGS.copy()
        data["players"] = ",".join(players)
        data["pairAt"] = now
        if START_CLOCKS_AFTER_SECS is not None:
            data["startClocksAt"] = now + START_CLOCKS_AFTER_SECS * 1000

        rep = self.http.post(BULK_CREATE_API, data=data, headers=HEADERS)
        if not check_response(rep, "Failed to create games"):
            return

        r = rep.json()
        log.debug(r)

        bulk_id: str = r["id"]
        games = [
            Game(round_nb, Pair(game["white"], game["black"]), game["game_id"], bulk_id)
            for game in r["games"]
        ]

        self.db.add_lichess_games(games)

        log.info(f"Created {len(games)} games")
        log.info(f"Total games in round: {len(self.db.get_game_ids(round_nb))}")

    def create_game(self: Pairing, pair: Pair) -> Optional[Tuple[str, str, str]]:
        """Returns (lichess game id, white, black) or None on failure"""
        # tokens = self.tokens.get_tokens(pair)
        # if tokens is None:
        #     return None

        url = CHALLENGE_API.format(pair.white_player, pair.black_player)
        data = GAME_SETTINGS.copy()
        data["color"] = "white"
        rep = self.http.post(
            url,
            data=data,
            # params={"token1": tokens[0], "token2": tokens[1]},
            headers=HEADERS,
        )

        if not check_response(rep, f"Failed to create game: {pair}"):
            return None

        r = rep.json()
        log.debug(r)
        return (r["game"]["id"], r["challenger"]["id"], r["destUser"]["id"])

    def create_games_single(self: Pairing, round_nb: int) -> None:
        unpaired = self.db.get_unpaired_players(round_nb)
        log.info(f"Round {round_nb}: {len(unpaired)} games to be created")

        for pair in unpaired:
            result = self.create_game(pair)
            if result is None:
                continue
            game_id, white, black = result
            if white != pair.white_player or black != pair.black_player:
                log.error(f"Round {round_nb}: Challenge was created incorrectly")
                log.error(f"Wanted: {pair}")
                log.error(f"Got: {white} vs {black} ({game_id})")
                log.error("Aborting")
                return
            self.db.add_lichess_game(Game(round_nb, pair, game_id))

        log.info(f"Created {len(unpaired)} games")
        log.info(f"Total games in round: {len(self.db.get_game_ids(round_nb))}")

    def start_clock(self: Pairing, game: Game) -> None:
        tokens = self.tokens.get_tokens(game.pair)
        if tokens is None:
            return
        rep = self.http.post(
            CHALLENGE_START_CLOCKS_API.format(game.game_id),
            params={"token1": tokens[0], "token2": tokens[1]},
            headers=HEADERS,
        )
        check_response(rep, f"Failed to start clock for: {game.game_id}")

    def start_clocks(self: Pairing, round_nb: int) -> None:
        games = self.db.get_unfinished_games(round_nb)
        bulk_ids: Set[str] = set()
        for game in games:
            if game.bulk_id is not None:
                bulk_ids.add(game.bulk_id)
            else:
                self.start_clock(game)

        for bulk in bulk_ids:
            rep = self.http.post(BULK_START_CLOCKS_API.format(bulk), headers=HEADERS)
            check_response(rep, "Failed to bulk start clocks")

    def fetch_results(self: Pairing, round_nb: int) -> None:
        games = self.db.get_unfinished_games(round_nb)
        log.info(f"Round {round_nb}: Checking {len(games)} previously unfinished games")
        # Not streaming since at most 128 games so ~6s and it avoid parsing ndjson.
        rep = self.http.post(
            LOOKUP_API,
            data=",".join(game.game_id for game in games),
            headers=HEADERS,
            params={"moves": "false"},
        )
        if not check_response(rep, "Failed to read games"):
            return
        r = rep.text.splitlines()
        log.debug(results)
        for raw_game in r:
            log.debug(raw_game)
            game = json.loads(raw_game)
            result = GameResult.from_game(game)
            game_id = game["id"]
            white = game["players"]["white"]["user"]["name"]
            black = game["players"]["black"]["user"]["name"]
            log.info(f"Game {game_id} - {white:>30} vs {black:30}: {result!s}")
            if result is not None:
                self.db.add_game_result(game_id, result)

        still_unfinished = len(self.db.get_unfinished_games(round_nb))
        log.info(f"Round {round_nb}: {still_unfinished} games still unfinished")

    def test(self: Pairing) -> None:
        pass


#############
# Functions #
#############


def create_db() -> None:
    """Setup the sqlite database, should be run once first when getting the script"""
    log.debug("cmd: create_db")
    Db().create_db()


def show_db() -> None:
    """Show the current state of the database. For debug purpose only"""
    log.debug("cmd: show_db")
    Db().show()

def show(round_nb: int) -> None:
    """Show the state of the round"""
    log.debug("cmd: show")
    Db().show_round(round_nb)


def test() -> None:
    log.debug("cmd: test")
    Pairing().test()


def insert(round_nb: int, path: str, force: bool = False) -> None:
    """Store pairings from a .txt or .csv file in the db, without launching the challenges"""
    log.debug(f"cmd: fetch {round_nb} {path}")
    FileHandler().insert(round_nb, path, force)


def create_games_bulk(round_nb: int) -> None:
    """Bulk create games for all pairs that have not been created already"""
    log.debug(f"cmd: create_games_bulk {round_nb}")
    Pairing().create_games_bulk(round_nb)


def create_games_single(round_nb: int) -> None:
    """Create games for all pairs that have not been created already using the challenge API"""
    log.debug(f"cmd: create_games_single {round_nb}")
    Pairing().create_games_single(round_nb)


def start_clocks(round_nb: int) -> None:
    """Start the clock of all games in the round"""
    log.debug(f"cmd: start_clocks {round_nb}")
    Pairing().start_clocks(round_nb)


def results(round_nb: int) -> None:
    """Fetch all games from that round_nb, check if they are finished, and print the results"""
    log.debug(f"cmd: results {round_nb}")
    Pairing().fetch_results(round_nb)


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
        "show_db": show_db,
        "test": test,
    }
    ROUND_COMMANDS = {
        "show": show,
        "insert": insert,
        "create_games_bulk": create_games_bulk,
        "create_games_single": create_games_single,
        "results": results,
        "broadcast": broadcast,
        "reset": reset,
        "start_clocks": start_clocks,
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

    subparsers.choices["insert"].add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Also insert pairings with players that already have one",
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
