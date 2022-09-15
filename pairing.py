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
import sqlite3
import sys

from dataclasses import dataclass
from dotenv import load_dotenv
from enum import IntEnum
from requests import Response
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from typing import Callable, Dict, List, Optional, Tuple, Iterator, Any

#############
# Constants #
#############

load_dotenv()


GAME_SETTINGS: Dict[str, Any] = {
    "rated": "true",
    "clock.limit": 5 * 60,  # starting time in seconds
    "clock.increment": 3,  # increment in seconds
    "variant": "chess960",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Leave out to get the default message: Your game with {opponent} is ready: {game}.
    # Must contain {game} which will be replaced with the game link. Can also contain {player} and {opponent}.
    # "message": "Your game for X tournament is ready: {game}"
}

# Only works with the bulk API. You can always manually start clocks though.
START_CLOCKS_AFTER_SECS = None

MESSAGE_FN: Optional[Callable[[int], str]] = None
# MESSAGE_FN = lambda rnd: f"FIDE Binance, Round {rnd}: Your game is ready: {{game}}"

ARMAGEDDON_SETTINGS = {
    "rated": "true",
    "clock.limit": 5 * 60,
    "clock.increment": 3,
    # "message": "FIDE Binance, Armageddon: Your game is ready: {game}",
}
ARMAGEDDON_EXTRA_TIME = 0  # in seconds for black
ARMAGEDDON_ROUND = 1337


CSV_DELIM = ";"
COLUMN_PLAYER1 = 0
COLUMN_PLAYER2 = 1

PLAYER_MAP = {
    "Karjakin": "Sergey_Karjakin",
    "Svidler": "Moose959",
    "Dubov": "SVODMEVKO",
    "Shimanov": "Multibrendovyi",
    "Fedoseev": "Feokl1995",
    "Grischuk": "STL_Grischuk",
    "Gelfand": "BGRishon",
    "Kramnik": "VB_Kramnik",
    "Bene": "BenWerner",
    "Test": "bentest",
}
# PLAYER_MAP = None
if PLAYER_MAP:
    REVERSE_PLAYER_MAP = {nick.lower(): real for real, nick in PLAYER_MAP.items()}
    PLAYER_MAP = {real.lower(): nick.lower() for real, nick in PLAYER_MAP.items()}
else:
    REVERSE_PLAYER_MAP = {}


TOKENS_PATH = "tokens.txt"
LOG_PATH = "out.log"

BASE = "http://localhost:9663" if __debug__ else "https://lichess.org"
BULK_CREATE_API = BASE + "/api/bulk-pairing"
BULK_START_CLOCKS_API = BASE + "/api/bulk-pairing/{}/start-clocks"
CHALLENGE_API = BASE + "/api/challenge/{}"
CHALLENGE_START_CLOCKS_API = BASE + "/api/challenge/{}/start-clocks"
CHALLENGE_TOKENS_API = BASE + "/api/token/admin-challenge"
TOKEN_TEST_API = BASE + "/api/token/test"
ADD_TIME_API = BASE + "/api/round/{}/add-time/{}"
LOOKUP_API = BASE + "/games/export/_ids"
DB_FILE = "FIDE_binance_test.db" if __debug__ else "FIDE_binance_prod.db"

HEADERS = {
    "Accept": "application/x-ndjson",
}
TOKEN = os.getenv("TOKEN_TEST" if __debug__ else "TOKEN_PROD")
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

RETRY_STRAT = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
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


def get_player(name: str) -> str:
    if PLAYER_MAP:
        username = PLAYER_MAP.get(name.lower())
        if not username:
            log.error(f"Unknown player name: {name}")
            exit(1)
        return username
    return name.lower()


@dataclass(frozen=True)
class Pair:
    white_player: str
    black_player: str


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
    ABORTED = 3

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
        elif status not in ("created", "started"):
            return GameResult.ABORTED
        return None


class Db:
    def __init__(self: Db) -> None:
        self.con = sqlite3.connect(DB_FILE, isolation_level=None)
        self.cur = self.con.cursor()

    def create_db(self: Db) -> None:
        # Since the event is divided in two parts, `round_nb` will first indicate the round_nb number in the round-robin then advancement in the knockdown event
        # `result` 0 = black wins, 1 = white wins, 2 = draw, 3 = aborted
        # `rowId` is the primary key and is create silently
        self.cur.execute(
            """CREATE TABLE rounds (
                white_player description VARCHAR(30) NOT NULL, 
                black_player description VARCHAR(30) NOT NULL, 
                lichess_game_id CHAR(8), 
                result INT,
                status VARCHAR(30),
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
                    status,
                    bulk_id
                   FROM rounds
                   WHERE round_nb = ?
                """,
            (round_nb,),
        )
        log.info(f"No  |DB Row{'White':>30} vs {'Black':30} Game Id  Bulk Id  State")
        for i, (rowId, white, black, game_id, result, status, bulk_id) in enumerate(
            raw_data
        ):
            if game_id is None:
                status = "Not created"
            elif result is not None:
                result_txt = [
                    "Black wins 0-1",
                    "White wins 1-0",
                    "Draw       ½-½",
                    "Aborted",
                ][result]
                status = f"{status:<12}{result_txt}"
            game_id = game_id or "--------"
            bulk_id = bulk_id or "--------"
            white = REVERSE_PLAYER_MAP.get(white, white)
            black = REVERSE_PLAYER_MAP.get(black, black)
            log.info(
                f"{i+1:>4}|{rowId:<5} {white:>30} vs {black:30} {game_id} {bulk_id} {status}"
            )

    def remove_round(self: Db, round_nb: int, force: bool = False) -> None:
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

    def get_uncreated_pairs(self: Db, round_nb: int) -> List[Pair]:
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
                result = NULL,
                status = "created"
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
                result = NULL,
                status = "created"
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

    def get_games(self: Db, round_nb: int) -> List[Tuple[str, str, str]]:
        raw_data = self.cur.execute(
            """SELECT 
                    lichess_game_id,
                    white_player,
                    black_player
                   FROM rounds
                   WHERE lichess_game_id IS NOT NULL AND round_nb = ?
                """,
            (round_nb,),
        )
        return list(raw_data)

    def get_game_from_id(self: Db, game_id: str) -> Optional[Game]:
        raw_data = self.cur.execute(
            """SELECT 
                    round_nb,
                    white_player,
                    black_player,
                    bulk_id
                   FROM rounds
                   WHERE lichess_game_id IS ?
            """,
            (game_id,),
        )
        if raw_data:
            round_nb, white, black, bulk_id = next(raw_data)
            return Game(round_nb, Pair(white, black), game_id, bulk_id)
        return None

    def update_game_status(
        self: Db, game_id: str, status: str, result: Optional[GameResult]
    ) -> None:
        self.cur.execute(
            """UPDATE rounds SET
                result = ?,
                status = ?
               WHERE lichess_game_id = ?
            """,
            (result, status, game_id),
        )


class FileHandler:
    def __init__(self: FileHandler, db: Optional[Db] = None) -> None:
        self.db = Db() if db is None else db

    def read_pairings_txt(self: FileHandler, path: str) -> Iterator[Tuple[str, str]]:
        with open(path) as f:
            for line in filter(bool, (line.strip() for line in f)):
                try:
                    row = line.split()
                    yield row[COLUMN_PLAYER1], row[COLUMN_PLAYER2]
                except Exception:
                    log.warning(f"Skipped line: {line}")
                    pass

    def read_pairings_csv(self: FileHandler, path: str) -> Iterator[Tuple[str, str]]:
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=CSV_DELIM)
            for row in reader:
                log.debug(row)
                try:
                    yield row[COLUMN_PLAYER1], row[COLUMN_PLAYER2]
                except Exception:
                    log.warning(f"Skipped row: {row}")
                    pass

    def insert(
        self: FileHandler, round_nb: int, path: str, force: bool = False
    ) -> None:
        if path.endswith(".txt"):
            pairs_iter = self.read_pairings_txt(path)
        elif path.endswith(".csv"):
            pairs_iter = self.read_pairings_csv(path)
        else:
            return log.error(f"Unsupported pairings file: {path}. Must be .txt or .csv")

        pairs: List[Pair] = []
        for player1, player2 in pairs_iter:
            player1, player2 = map(get_player, (player1, player2))
            pair = Pair(white_player=player1, black_player=player2)
            log.debug(pair)
            pairs.append(pair)

        existing = set()
        for pair in self.db.get_pairs(round_nb):
            existing.add(pair.white_player)
            existing.add(pair.black_player)
        filtered_pairs = [
            pair
            for pair in pairs
            if not pair.white_player in existing and not pair.black_player in existing
        ]
        if len(filtered_pairs) != len(pairs):
            if force:
                log.warning("Force inserting pairs for already paired players")
            else:
                log.warning("Skipped inserting players which are already paired:")
                for pair in set(pairs) - set(filtered_pairs):
                    log.warning(f"{pair.white_player:>30} vs {pair.black_player:<30}")
                log.warning("Use --force to insert them anyway")
                pairs = filtered_pairs

        self.db.add_pairs(round_nb, pairs)
        log.info(f"Round {round_nb}: {len(pairs)} pairings inserted")
        log.info(f"Total pairs: {len(self.db.get_pairs(round_nb))}")


def check_response(response: Response, error_msg: str) -> bool:
    if not response.ok:
        log.error(error_msg)
        log.error(
            f"{response.request.method} {response.url} => {response.status_code} {response.reason}"
        )
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
                name, token = map(str.strip, line.split(":"))
                self.tokens[name.lower()] = token

    def get(self: Tokens, player: str) -> Optional[str]:
        return self.tokens.get(player)

    def get_tokens(self: Tokens, pair: Pair) -> Optional[Tuple[str, str]]:
        t1 = self.get(pair.white_player)
        t2 = self.get(pair.black_player)
        if t1 is None:
            log.error(f"No token for: {pair.white_player}")
            return None
        if t2 is None:
            log.error(f"No token for: {pair.black_player}")
            return None
        return t1, t2

    def insert(self, tokens: Dict[str, str]) -> None:
        for name, token in tokens.items():
            self.tokens[name.lower()] = token

    def save(self) -> None:
        longest = max(map(len, self.tokens))
        with open(TOKENS_PATH, "w") as f:
            for name, token in sorted(self.tokens.items()):
                name += " " * (len(name) - longest + 1)
                f.write(f"{name}: {token}\n")


class Pairing:
    def __init__(self: Pairing, db: Optional[Db] = None) -> None:
        self.db = Db() if db is None else db
        self.tokens = Tokens()
        self.http = requests.Session()
        self.http.mount("https://", ADAPTER)
        self.http.mount("http://", ADAPTER)

    def create_games_bulk(self: Pairing, round_nb: int) -> None:
        unpaired = self.db.get_uncreated_pairs(round_nb)
        log.info(f"Round {round_nb}: {len(unpaired)} games to be created")

        players = []

        for pair in unpaired:
            token_pair = self.tokens.get_tokens(pair)
            if token_pair is None:
                return
            players.append(":".join(token_pair))

        now = int(time.time()) * 1000
        data = GAME_SETTINGS.copy()
        if MESSAGE_FN is not None:
            data["message"] = MESSAGE_FN(round_nb)
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
            Game(round_nb, Pair(game["white"], game["black"]), game["id"], bulk_id)
            for game in r["games"]
        ]

        self.db.add_lichess_games(games)

        log.info(f"Created {len(games)} games")
        log.info(f"Total games in round: {len(self.db.get_game_ids(round_nb))}")

    def create_game(
        self: Pairing, round_nb: int, pair: Pair, settings: Dict[str, Any]
    ) -> Optional[str]:
        """Returns the lichess game id on success or None on failure"""
        tokens = self.tokens.get_tokens(pair)
        if not tokens:
            return None
        url = CHALLENGE_API.format(pair.black_player)
        settings["color"] = "white"
        settings["acceptByToken"] = tokens[1]
        rep = self.http.post(
            url, data=settings, headers={"Authorization": f"Bearer {tokens[0]}"}
        )

        if not check_response(rep, f"Failed to create game: {pair}"):
            return None

        r = rep.json()
        log.debug(r)
        game_id = r["game"]["id"]
        self.db.add_lichess_game(Game(round_nb, pair, game_id))
        return game_id

    def create_games_single(self: Pairing, round_nb: int) -> None:
        unpaired = self.db.get_uncreated_pairs(round_nb)
        log.info(f"Round {round_nb}: {len(unpaired)} games to be created")
        count = 0

        for pair in unpaired:
            settings = GAME_SETTINGS.copy()
            if MESSAGE_FN is not None:
                settings["message"] = MESSAGE_FN(round_nb)
            if self.create_game(round_nb, pair, settings) is not None:
                count += 1

        log.info(f"Created {count} games")
        log.info(f"Total games in round: {len(self.db.get_game_ids(round_nb))}")

    def start_clock(self: Pairing, game: Game) -> None:
        log.debug(f"Starting clock for {game.game_id}")
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
        log.info(f"Round {round_nb}: Starting clock for {len(games)} games")

        bulk_ids = set()
        for game in games:
            if game.bulk_id is not None:
                bulk_ids.add(game.game_id)
            else:
                self.start_clock(game)

        for bulk_id in bulk_ids:
            log.debug(f"Bulk starting clocks for bulk {bulk_id}")
            rep = self.http.post(BULK_START_CLOCKS_API.format(bulk_id), headers=HEADERS)
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
        log.debug(r)
        for raw_game in r:
            log.debug(raw_game)
            game = json.loads(raw_game)
            result = GameResult.from_game(game)
            self.db.update_game_status(game["id"], game["status"], result)

        still_unfinished = len(self.db.get_unfinished_games(round_nb))
        log.info(f"Round {round_nb}: {still_unfinished} games still unfinished")

    def create_armageddon(self: Pairing, player1: str, player2: str) -> None:
        pair = Pair(*map(get_player, (player1, player2)))
        token_white = self.tokens.get(pair.white_player)

        if token_white is None:
            log.error(f"No token for {pair.white_player}")
            return

        self.db.add_pairs(ARMAGEDDON_ROUND, [pair])

        game_id = self.create_game(ARMAGEDDON_ROUND, pair, ARMAGEDDON_SETTINGS.copy())
        if game_id is None:
            return

        log.info(f"Created armageddon: https://lichess.org/{game_id}")

        if ARMAGEDDON_EXTRA_TIME > 0:
            rep = self.http.post(
                ADD_TIME_API.format(game_id, ARMAGEDDON_EXTRA_TIME),
                headers={"Authorization": f"Bearer {token_white}"},
            )
            check_response(rep, "Failed to add time for armageddon")

    def ensure_tokens(self, round_nb: int, create: Optional[str]) -> None:
        players = set()
        for pair in self.db.get_pairs(round_nb):
            players.add(pair.white_player)
            players.add(pair.black_player)

        missing = set()
        tokens = {}
        for p in players:
            token = self.tokens.get(p)
            if token is None:
                missing.add(p)
            else:
                tokens[token] = p

        logfn = log.error if create is None else log.info
        if missing:
            logfn("Missing tokens for:")
            for p in sorted(missing):
                logfn(p)

        now = int(time.time()) * 1000 + 24 * 60 * 60 + 1000
        resp = self.http.post(TOKEN_TEST_API, data=",".join(tokens.keys()))
        if check_response(resp, "Failed to validate tokens"):
            for token, data in resp.json().items():
                user = tokens[token]
                if data is None:
                    logfn(f"Bad token for {user}.")
                    missing.add(user)
                elif data["userId"] != user:
                    logfn(f"Token for {user} is actually for another user.")
                    missing.add(user)
                elif "challenge:write" not in data["scopes"]:
                    logfn(f"Missing 'challenge:write' scope for {user}.")
                    missing.add(user)
                elif data["expires"] < now:
                    logfn(f"Token for {user} is expired.")
                    missing.add(user)

        if not missing:
            log.info("All tokens present")
            return

        if create is None:
            return

        if len(create) < 10:
            log.error(
                "Please provide a proper description for the token (at least 10 characters)"
            )
            return

        if "Authorization" not in HEADERS:
            log.error("No appropriate token in .env file")
            return

        log.info("Creating tokens...")
        resp = self.http.post(
            CHALLENGE_TOKENS_API,
            headers=HEADERS,
            data={"users": ",".join(missing), "description": create},
        )
        if not check_response(resp, "Failed to create tokens"):
            return

        result = resp.json()
        self.tokens.insert(result)
        self.tokens.save()

        log.info("Created tokens.")

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


def start_clock(game_id: str) -> None:
    """Start the clock for a single game"""
    log.debug(f"cmd: start_clock {game_id}")
    db = Db()
    game = db.get_game_from_id(game_id)
    if not game:
        log.error(f"No game with id {game_id}")
        return
    Pairing(db).start_clock(game)


def create_armageddon(player1: str, player2: str) -> None:
    """Start an armageddon game between two players"""
    log.debug(f"cmd: create_armageddon {player1} {player2}")
    Pairing().create_armageddon(player1, player2)


def results(round_nb: int) -> None:
    """Fetch all games from that round_nb, check if they are finished, and print the results"""
    log.debug(f"cmd: results {round_nb}")
    p = Pairing()
    p.fetch_results(round_nb)
    p.db.show_round(round_nb)


def broadcast(round_nb: int) -> None:
    """Return game ids of the round `round_nb` separated by a space"""
    log.debug(f"cmd: broadcast {round_nb}")
    game_ids = Db().get_game_ids(round_nb)
    log.info(f"Round {round_nb}: {len(game_ids)} games started")
    log.info(f"Game ids: {' '.join(game_ids)}")


def game_urls(round_nb: int) -> None:
    """Return game URLs of the round `round_nb`"""
    log.debug(f"cmd: game_urls {round_nb}")
    games = Db().get_games(round_nb)
    log.info(f"Round {round_nb}: {len(games)} games started")
    for game_id, white, black in games:
        white = REVERSE_PLAYER_MAP.get(white, white)
        black = REVERSE_PLAYER_MAP.get(black, black)
        print(f"{white} - {black}: https://lichess.org/{game_id}")


def reset(round_nb: int, force: bool = False) -> None:
    """Reset a round by removing all its pairings. Use `--force` to remove pairings with created games."""
    log.debug(f"cmd: reset {round_nb} force={force}")
    Db().remove_round(round_nb, force)


def ensure_tokens(round_nb: int, create: Optional[str] = None) -> None:
    """Ensure tokens for all players in the round are present. Use `--create "Token Description"` to create missing tokens via the Challenge Admin API."""
    log.debug(f"cmd: ensure_tokens {round_nb} create={create}")
    Pairing().ensure_tokens(round_nb, create)


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
        "game_urls": game_urls,
        "reset": reset,
        "start_clocks": start_clocks,
        "ensure_tokens": ensure_tokens,
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

    subparsers.choices["ensure_tokens"].add_argument(
        "-c",
        "--create",
        dest="create",
        metavar="DESCRIPTION",
        type=str,
        help="Create missing tokens via Challenge Admin API. Pass token description as argument.",
    )

    p = subparsers.add_parser("start_clock", help=start_clock.__doc__)
    p.set_defaults(func=start_clock)
    p.add_argument(
        "game_id", metavar="GAME_ID", help="ID of the game to start clocks for",
    )

    p = subparsers.add_parser("create_armageddon", help=create_armageddon.__doc__)
    p.set_defaults(func=create_armageddon)
    p.add_argument(
        "player1", metavar="WHITE_PLAYER", help="The white player of the armageddon",
    )
    p.add_argument(
        "player2", metavar="BLACK_PLAYER", help="The black player of the armageddon",
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
