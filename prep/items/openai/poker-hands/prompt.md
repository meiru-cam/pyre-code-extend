`N` players (`2 <= N <= 1e5`) are each dealt a three-card poker hand, and every hand must be ranked and a winner (or a tied set of winners) reported. A card is written as two characters, a rank followed by a suit: ranks are ordered `2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < T < J < Q < K < A` (`T` is ten) and suits are `C`, `D`, `H`, `S`. A hand is a list of exactly three such cards, e.g. `["9H", "TC", "9D"]`; at most `3e5` cards are in play in total.

### Part 1 — Hand types and comparisons (no suits yet)

Ignoring suits, every hand is one of three types, in priority order from strongest to weakest:

- **Three of a Kind**: all three cards share the same rank.
- **Pair**: exactly two of the three cards share a rank.
- **High Card**: the three ranks are all different.

A hand of a stronger type always beats a hand of a weaker type, regardless of the actual ranks involved. Between two hands of the *same* type, compare as follows:

- **Three of a Kind**: compare the shared rank.
- **Pair**: compare the pair's rank first; if that is equal, compare the remaining (unpaired) card's rank.
- **High Card**: sort each hand's three ranks from high to low and compare them in that order — highest first, then middle, then lowest.

If every comparison point above comes out equal, the two hands tie. Suit never breaks a tie.

```py
def hand_type(cards: list[str]) -> str:
    """cards has exactly 3 elements, each formatted <rank><suit>, e.g. "9H" or "TC". Returns one of
    "Three of a Kind", "Pair", "High Card"."""

def compare_hands(cards_a: list[str], cards_b: list[str]) -> int:
    """Compares two 3-card hands under the rules above. Returns 1 if cards_a ranks strictly above
    cards_b, -1 if strictly below, 0 if the two hands tie."""

def rank_players(hands: list[list[str]]) -> list[int]:
    """hands[i] is player i's three cards. Returns, in increasing order, the 0-indexed players whose hand
    does not rank below any other player's hand -- more than one index means those players tie for the
    win."""
```

Example: player 0 has `7H 7D 4C`, player 1 has `7S 7C 4D`, player 2 has `KD QS JC`.

```text
hand_type(player 0) = Pair       (a pair of 7s, kicker 4)
hand_type(player 1) = Pair       (a pair of 7s, kicker 4)
hand_type(player 2) = High Card  (K, Q, J)
rank_players(...)   = [0, 1]     # identical pairs, so players 0 and 1 tie for the win
```

### Part 2 — Adding flush

Suits now matter: a **Flush** is a hand whose three cards all share the same suit. A hand that is already Three of a Kind is never also called a Flush. The priority order becomes:

Three of a Kind > Flush > Pair > High Card.

Two flushes are compared exactly like High Card: sort each hand's three ranks high to low and compare in that order.

```py
def hand_type(cards: list[str]) -> str:
    """Same input as Part 1. Now also returns "Flush" for a same-suited hand that is not Three of a
    Kind, ranked between Three of a Kind and Pair."""
```

`compare_hands` and `rank_players` keep the signatures from Part 1; they now follow the updated priority order and the flush tie-break rule above.

Example: player 0 has `9C 7C 3C`, player 1 has `JD JH 8S`, player 2 has `AS QD 6H`.

```text
hand_type(player 0) = Flush      (9, 7, 3, all clubs)
hand_type(player 1) = Pair       (a pair of Js, kicker 8)
hand_type(player 2) = High Card  (A, Q, 6)
rank_players(...)   = [0]        # the flush beats the pair despite its lower top card
```

### Part 3 — Folding

Now the players are dealt from a shared `deck`: a list of at least `3 * N` distinct cards, in the exact order they will be drawn. Dealing proceeds in three passes; each pass visits players `0, 1, ..., N-1` in that order, once each:

- **Passes 1 and 2** always deal that player their next card, drawn from the front of the remaining `deck`.
- **Pass 3** first asks that player's *fold rule* — a function of that player's own first two cards, in the order they were dealt, and nothing else (not any other player's cards or decisions). If the rule says fold, the player is marked *folded* and receives no third card; no `deck` card is consumed for them, so whichever later player draws next still gets the next undrawn card. Otherwise, the player is dealt a third card as usual.

There are exactly `3 * N` deals-or-folds in total, `N` per pass. Once pass 3 has visited every player the round is over, and the result is decided:

- If every player folded, there is no winner.
- If exactly one player never folded, that player wins outright.
- Otherwise, compare the hands of every player who never folded using the Part 2 rules (Flush included); ties among them follow the same rule as in Parts 1 and 2.

Three fold rules:

- **Never fold.**
- **Fold unless the first two cards are a pair** (same rank).
- **Fold unless the first two cards are a flush draw** (same suit).

```py
def never_fold(first_two: list[str]) -> bool: ...
def fold_unless_pair(first_two: list[str]) -> bool: ...
def fold_unless_flush_draw(first_two: list[str]) -> bool: ...

class PokerRound:
    def __init__(self, num_players: int, deck: list[str], strategies: list):
        """strategies[i] is player i's fold rule, one of the three functions above."""

    def deal_card(self) -> str:
        """Called only while is_over() is False. Advances the round by exactly one deal-or-fold and
        returns a line describing it: either "Player {i} is dealt {card}." or "Player {i} folds."."""

    def is_over(self) -> bool: ...

    def check_result(self) -> str:
        """Valid once is_over() is True. Returns "Winner: Player {i}", or
        "Winner: Tie (Players {i}, {j}, ...)" with the indices in increasing order, or
        "No winner: all players folded."."""
```

Example: 3 players share the deck `3S TD 8H QC TS 4H 2D 6H 7C`; player 0 folds unless holding a pair after two cards, player 1 never folds, player 2 folds unless holding a flush draw after two cards.

```text
Player 0 is dealt 3S.
Player 1 is dealt TD.
Player 2 is dealt 8H.
Player 0 is dealt QC.
Player 1 is dealt TS.
Player 2 is dealt 4H.
Player 0 folds.
Player 1 is dealt 2D.
Player 2 is dealt 6H.
Winner: Player 2
```

Player 0's first two cards (`3S`, `QC`) are not a pair, so player 0 folds and no card is drawn for them: player 1's third card is `2D`, the next undrawn card, and `6H` goes to player 2. Player 2's first two (`8H`, `4H`) share a suit, so player 2 stays in and ends with a flush, which beats player 1's pair of tens; `7C` is never dealt.
