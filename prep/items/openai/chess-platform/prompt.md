Design the real-time backend for a one-on-one online chess platform. A registered player joins a queue for a time control (for example, 5 minutes with a 3-second increment added after each move) and a mode (rated or casual); the system pairs them with an opponent of comparable rating and starts a game. Once a game starts, both players submit moves over a persistent connection and must see the opponent's move appear almost immediately. Each player has an individual chess clock, and the server is the sole authority over both clocks: a player's on-screen timer is only a display and is never trusted to decide when time has run out. A dropped connection never pauses a game: the clock of the side to move keeps running whether or not either player is connected, and a player who reconnects must resume from the exact current position and the exact current clock reading. The server also decides how a game ends: a game that is under way ends in exactly one of seven ways — checkmate; stalemate (the side to move has no legal move and is not in check); resignation; a clock reaching zero; an agreed draw; threefold repetition (the same position occurring for the third time); and the fifty-move rule (fifty moves by each side with neither a capture nor a pawn move). The last two are declared as soon as they happen; a player does not have to claim them. A finished game's full move history is recorded permanently, and both players' ratings are updated.

Scale this design for:

- 5,000,000 players play at least one game on a typical day.
- At peak, 300,000 players are online at once (6% of daily players), which puts 150,000 games in progress simultaneously.
- A game runs between about 1 and 30 minutes, averaging about 8 minutes and about 80 moves in total, 40 by each side (a "move" here is one player's turn — chess notation calls this a *ply*).
- Target match wait time: 95th percentile under 4 seconds.
- Target move relay latency, from one player submitting a move to the opponent's screen showing it: 95th percentile under 120 ms.
- Target clock-timeout detection: the server must declare a timeout within 50 ms of the instant a player's remaining time actually reaches zero.
- Target availability: 99.9%.
- Correctness requirement: no move the server has acknowledged is ever lost, and no move is applied twice or accepted out of turn, even when a client retries a request or reconnects mid-game.

In scope: joining a queue and matchmaking by rating; validating, applying, and relaying moves for a single game in real time; the server-authoritative clock and timeout detection; resignation, draw offers, and the drawn endings the server declares on its own; handling disconnects and reconnects; and persisting a finished game's move history while updating both players' ratings. Out of scope: spectating an in-progress game, tournaments, a computer opponent, cheat or engine-assistance detection, chat, and the rating formula itself — assume a rating-update function that takes two ratings and a game result and returns the two updated ratings is available as a library call.

Produce:

- Requirements and a scale estimate: the peak rate of moves relayed per second, the number of concurrent WebSocket connections and how many connection-serving instances that requires (state your assumption for how many connections one instance can hold), and the daily storage growth from persisting move history.
- A data model (players, games, move records, queue entries) and the APIs, with the REST endpoints and the WebSocket messages listed separately.
- An architecture diagram, and a walk-through of one move from the moment a player submits it to the moment the opponent's client shows it.
- Deep dives into: the server-authoritative clock; keeping moves consistent and routed correctly within a single game; and matchmaking.
