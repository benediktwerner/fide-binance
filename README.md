# Pairings Creator

## Setup

1. Have a recent version of Python installed
2. *Optional:* Create a venv and activate it: `python3 -m venv venv && source venv/bin/activate` 
3. Run `pip install -r requirements.txt`
4. *If using the Bulk or Challenge Admin API:* Copy `.env.example` to `.env` and replace the example tokens
5. Edit `PLAYER_MAP` in `pairing.py`:
   - Mapping `name: username`, then use `name` to refer to that player in pairings, usernames won't work
   - Alternatively, disable it by removing all the names or uncommenting the `PLAYER_MAP = None` line and refer to them by usernames instead. This won't catch typos as easily though.
6. *If not using Challenge Admin API to create tokens:* Create `tokens.txt` file with one line per player `username:token` with a `challenge:write` token for that player.

## Usage

By default, the script will run against a local Lichess instance and with `TOKEN_TEST`.
Run as `python3 -O pairing.py` to run against Lichess proper and with `TOKEN_PROD`.

To adjust game settings, modify the `GAME_SETTINGS` dictionary near the top of `pairing.py`.

For commands with longer names, the repo contains empty files with their names which should allow your shell to autocomplete them.

- Create database: `pairing.py create_db`
- Help: `pairing.py -h`

- Create round: `pairing.py insert 42 round1.txt`
  - This will set up a round in the local database. It won't make any calls to Lichess
  - `42` is the round number (used in other commands later on to specify the round)
  - Can load pairings from:
    - `.csv` file: `whitePlayer;blackPlayer` one game per line
    - `.txt` file: `whitePlayer blackPlayer` one game per line separated by whitespace
- View round details: `pairing.py show 42`
- Ensure valid tokens are present for all players in a round: `pairing.py ensure_tokens 42`
  - Automatically create missing tokens via the Challenge Admin API: `pairing.py ensure_tokens 42 --create "Description for the token"`
  - This requires a `web:mod` token from an account with the Challenge Admin permission in the `.env` file.
- Create games from round 42:
  - Using Bulk API: `pairing.py create_games_bulk 42`
  - Using challenge API: `pairing.py create_games_single 42`
- See current state of games in round: `pairing.py results 42`
- Get IDs of games for broadcasting: `pairing.py broadcast 42`
- Get game URLs: `pairing.py game_urls 42`
- Reset/Delete round: `pairing.py reset 42`
  - This won't delete already created games. To delete them anyway, use: `pairing.py reset --force 42`
- Start clocks of all games in round: `pairing.py start_clocks 42`
- Start clock of a single game: `pairing.py start_clock abcdefgh` (`abcdefgh` is the Lichess ID of the game)
- Create armageddon: `pairing.py create_armageddon <whitePlayer> <blackPlayer>`
  - Edit `ARMAGEDDON_SETTINGS` and the values below it in the code to adjust armageddon settings.
  - You can use the above commands with round number 1337 to view or act on armageddon games
