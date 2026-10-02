"""Rank three-card poker hands, add flushes, then deal a round where players may fold."""

from ._interview import interview

# An independent ranking written from the statement, plus deck and hand generators.
_CARDS = r"""
import random, time
from collections import Counter

RANK = {r: i for i, r in enumerate("23456789TJQKA")}
DECK = [r + s for r in "23456789TJQKA" for s in "CDHS"]

def strength(hand, flush=True):
    ranks = sorted((RANK[c[0]] for c in hand), reverse=True)
    counts = Counter(ranks)
    if len(counts) == 1:
        return (3, ranks[0])
    if flush and len({c[1] for c in hand}) == 1:
        return (2, *ranks)
    if len(counts) == 2:
        pair = max(counts, key=counts.get)
        return (1, pair, min(counts, key=counts.get))
    return (0, *ranks)

NAMES = {3: "Three of a Kind", 2: "Flush", 1: "Pair", 0: "High Card"}

def random_hand(rng, allow_flush):
    while True:
        hand = rng.sample(DECK, 3)
        if allow_flush or len({c[1] for c in hand}) > 1:
            return hand

def biased_hand(rng, allow_flush):
    # Mostly pairs and shared ranks, so ties and kickers come up often.
    while True:
        ranks = rng.sample("23456789TJQKA", rng.choice([1, 2, 2, 3]))
        picks = [rng.choice(ranks) for _ in range(3)]
        hand, used = [], set()
        for r in picks:
            suits = [s for s in "CDHS" if r + s not in used]
            card = r + rng.choice(suits)
            used.add(card)
            hand.append(card)
        if allow_flush or len({c[1] for c in hand}) > 1:
            return hand

def winners(hands, flush=True):
    keys = [strength(h, flush) for h in hands]
    best = max(keys)
    return [i for i, k in enumerate(keys) if k == best]
"""

# A straight simulation of a round, kept apart from the learner's class.
_ROUND = _CARDS + r"""
def never(first_two):
    return False

def unless_pair(first_two):
    return first_two[0][0] != first_two[1][0]

def unless_flush_draw(first_two):
    return first_two[0][1] != first_two[1][1]

def simulate(deck, strategies):
    n = len(strategies)
    hands, folded, lines, nxt = [[] for _ in range(n)], [False] * n, [], 0
    for step in range(3):
        for p in range(n):
            if step == 2 and strategies[p](list(hands[p])):
                folded[p] = True
                lines.append(f"Player {p} folds.")
                continue
            hands[p].append(deck[nxt])
            lines.append(f"Player {p} is dealt {deck[nxt]}.")
            nxt += 1
    alive = [p for p in range(n) if not folded[p]]
    if not alive:
        return lines, "No winner: all players folded."
    best = winners([hands[p] for p in alive])
    if len(best) == 1:
        return lines, f"Winner: Player {alive[best[0]]}"
    return lines, "Winner: Tie (Players " + ", ".join(str(alive[i]) for i in best) + ")"

def play(table, deck, strategies):
    table.start_round(deck, strategies)
    lines = []
    while not table.is_over():
        lines.append(table.deal_card())
        assert len(lines) <= 3 * len(strategies), "is_over() never became True"
    return lines, table.check_result()
"""

TASK = {
    "title": "Three-Card Poker",
    "difficulty": "Easy",
    "version": 1,
    "function_name": "PokerTable",
    "description_en": r"""Build `PokerTable`, which ranks three-card poker hands and decides who wins.

The requirement arrives in parts. Each part keeps every earlier behavior, so one `PokerTable` class passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A card is two characters, a rank then a suit. Ranks go `2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < T < J < Q < K < A`, where `T` is ten. Suits are `C`, `D`, `H`, `S`.
- A hand is a list of three distinct cards, such as `["5S", "JD", "5C"]`.
- Suit never breaks a tie.
- There may be up to 100,000 players; the work must grow linearly with the number of cards.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the rules are short, and the clean answer turns each hand into one comparable key so that each later part adds one requirement without new comparison code.

**Where it is used:** card games, and any ranking with tie-breaks, such as sorting search results by score, then recency, then id.

Adapted from the three-card poker question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one class.""",
    "parts": [
        {
            "title": "Hand types",
            "description_en": r"""**Signature:** `PokerTable()`, `hand_type(cards) -> str`, `compare_hands(a, b) -> int`, `rank_players(hands) -> list[int]`

Each hand has one type, strongest first:
- `"Three of a Kind"`: all three ranks equal. Ties compare that rank.
- `"Pair"`: exactly two ranks equal. Ties compare the pair's rank, then the third card's rank.
- `"High Card"`: three different ranks. Ties compare the ranks from highest to lowest: the highest first, then the middle, then the lowest.

- A stronger type always beats a weaker one, whatever the ranks.
- `compare_hands(a, b)` returns `1` if `a` wins, `-1` if `b` wins, and `0` for a tie.
- `rank_players(hands)` returns the indices of every hand that no other hand beats, in increasing order.
- In this part no hand has all three cards of one suit.

**Example:**
- `hand_type(["8D", "8C", "8S"])` is `"Three of a Kind"`, `hand_type(["KH", "KS", "2C"])` is `"Pair"`, `hand_type(["AS", "KD", "QC"])` is `"High Card"`
- `compare_hands(["4H", "4D", "9C"], ["4S", "4C", "TD"])` is `-1`: equal pairs, and the ten beats the nine
- `compare_hands(["AH", "7D", "3C"], ["AS", "7C", "3H"])` is `0`
- `rank_players([["KH", "KS", "2C"], ["AS", "KD", "QC"], ["KD", "KC", "2H"]])` is `[0, 2]`""",
        },
        {
            "title": "Flushes",
            "description_en": r"""Keep Part 1. Hands may now have all three cards of one suit.

- A `"Flush"` is a hand whose three cards share a suit and is not Three of a Kind.
- The order becomes Three of a Kind, then Flush, then Pair, then High Card.
- Two flushes compare like High Card: ranks from highest to lowest.
- `hand_type`, `compare_hands` and `rank_players` all follow the new order.

**Example:**
- `hand_type(["2D", "5D", "9D"])` is `"Flush"`, and it beats `["AS", "AH", "KC"]`
- `["QH", "QD", "QC"]` beats any flush
- `compare_hands(["KS", "7S", "4S"], ["KD", "7D", "3D"])` is `1`""",
        },
        {
            "title": "Deal and fold",
            "description_en": r"""Keep Parts 1–2 and play a round.

**Signature:** `start_round(deck, strategies) -> None`, `deal_card() -> str`, `is_over() -> bool`, `check_result() -> str`

- `strategies[i]` belongs to player `i`, so there are `N = len(strategies)` players. `deck` holds at least `3 * N` cards in the order they are drawn; it may repeat a card.
- The round is three passes over players `0` to `N - 1`. Each `deal_card()` call handles the next player of the current pass and returns one line.
- In passes 1 and 2 the player gets the next undrawn card: `"Player {i} is dealt {card}."`.
- In pass 3, call `strategies[i]` once with the player's two cards in the order dealt. If it returns `True`, the player folds and gets no card: `"Player {i} folds."`. The next player then draws the card this player would have had. Otherwise the player gets the next card as usual.
- `is_over()` is `True` after all `3 * N` calls. `check_result()` then returns `"No winner: all players folded."`, `"Winner: Player {i}"`, or for a tie among the best hands `"Winner: Tie (Players {i}, {j}, ...)"` in increasing order. Only players who did not fold count, and the Part 2 order applies.
- `start_round` may be called again to play a new round.

**Example:** four players; player 1 folds unless their first two cards share a rank, player 2 folds unless they share a suit, players 0 and 3 never fold. The deck is `9C 2H KS 4D 9S 7H QS 4C JD AS 3H 8S`:
- passes 1 and 2 give player 0 `9C 9S`, player 1 `2H 7H`, player 2 `KS QS`, player 3 `4D 4C`
- in pass 3 player 0 gets `JD`, player 1 folds, player 2 gets `AS`, player 3 gets `3H`
- `check_result()` is `"Winner: Player 2"`: a spade flush beats the two pairs""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "Could each hand become one value such that comparing two values gives the same answer as the rules? What would that value need first, so a stronger type always wins, and what after it, so ties within a type break correctly?"},
        {"level": 2, "kind": "analysis", "content": "Map each hand to a tuple (type, tie-break): (2, rank) for three of a kind, (1, pair rank, other rank) for a pair, (0, ranks high to low) for high card, with ranks as indexes into \"23456789TJQKA\". Python compares tuples left to right, so compare_hands compares keys, and rank_players keeps every index whose key equals the max."},
    ],
    "model_connections": [
        "Leaderboards and model evaluation reports rank entries by a primary metric with explicit tie-breaks, which is the same tuple key.",
        "Self-play environments for card games, used in reinforcement learning research, need exactly this kind of deterministic dealer and hand evaluator.",
    ],
    "pro_con_analysis": {
        "pros": [
            "One tuple key per hand turns every comparison into Python's tuple comparison.",
            "Adding a hand type only changes the key function, not compare_hands or rank_players.",
            "A single pointer into the deck that moves only on a deal makes folds cost nothing.",
        ],
        "cons": [
            "Type numbers baked into the key must be renumbered when a type is inserted between others.",
            "Five-card poker needs many more tie-break shapes, such as two pair and full house.",
            "Fold rules here see only the player's own cards; real strategy needs probabilities over unseen cards.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "state.invariant", "code": r"""
table = {fn}()
assert table.hand_type(["8D", "8C", "8S"]) == "Three of a Kind"
assert table.hand_type(["KH", "KS", "2C"]) == "Pair"
assert table.hand_type(["AS", "KD", "QC"]) == "High Card"
assert table.compare_hands(["4H", "4D", "9C"], ["4S", "4C", "TD"]) == -1
assert table.compare_hands(["AH", "7D", "3C"], ["AS", "7C", "3H"]) == 0
assert table.rank_players([["KH", "KS", "2C"], ["AS", "KD", "QC"], ["KD", "KC", "2H"]]) == [0, 2]
"""},
        {"name": "Part 1: random hands", "part": 1, "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "On random hands without a flush, a type, comparison or set of winners differed from the rules: type first, then the tie-break of that type, and suit never matters.",
         "code": _CARDS + r"""
table = {fn}()
for seed in range(3000):
    rng = random.Random(seed)
    make = biased_hand if seed % 2 else random_hand
    a, b = make(rng, False), make(rng, False)
    assert table.hand_type(a) == NAMES[strength(a)[0]], a
    expected = (strength(a) > strength(b)) - (strength(a) < strength(b))
    assert table.compare_hands(a, b) == expected, (a, b)
    assert table.compare_hands(b, a) == -expected, (b, a)
for seed in range(400):
    rng = random.Random(seed)
    hands = [biased_hand(rng, False) for _ in range(rng.randint(2, 8))]
    assert table.rank_players(hands) == winners(hands), hands
"""},
        {"name": "Part 1: type beats rank, kickers and ties", "part": 1, "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "A weaker type with higher cards must still lose; a pair compares its rank before the third card; high cards compare every rank from the top; and identical ranks tie.",
         "code": r"""
table = {fn}()
assert table.compare_hands(["2C", "2D", "2H"], ["AS", "AH", "KD"]) == 1
assert table.compare_hands(["3C", "3D", "2H"], ["AS", "KH", "QD"]) == 1
assert table.compare_hands(["5C", "5D", "AH"], ["6S", "6H", "2D"]) == -1, "the pair's rank comes before the third card"
assert table.compare_hands(["KC", "2D", "KH"], ["KS", "3H", "KD"]) == -1, "the pair may be anywhere in the list"
assert table.compare_hands(["AC", "9D", "3H"], ["AS", "9H", "2D"]) == 1, "the lowest card decides last"
assert table.compare_hands(["AC", "TD", "2H"], ["AS", "9H", "8D"]) == 1, "the middle card comes before the lowest"
assert table.compare_hands(["7C", "7D", "7H"], ["7S", "7D", "7C"]) == 0
assert table.rank_players([["9C", "9D", "4H"], ["9S", "9H", "4D"], ["9C", "9S", "4C"]]) == [0, 1, 2]
assert table.rank_players([["2C", "3D", "5H"], ["2S", "3H", "4D"]]) == [0]
"""},
        {"name": "Part 1: many players", "part": 1, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Ranking 40,000 hands took far more than ten times as long as 4,000; compute one key per hand and take the max instead of comparing hands in pairs.",
         "code": _CARDS + r"""
def timed(n):
    rng = random.Random(n)
    hands = [random_hand(rng, False) for _ in range(n)]
    table = {fn}()
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        result = table.rank_players(hands)
        best = min(best, time.perf_counter() - start)
    assert result == winners(hands)
    return best

ratio = timed(40000) / timed(4000)
assert ratio < 30, f"10x the players took {ratio:.0f}x as long"
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "state.invariant", "code": r"""
table = {fn}()
assert table.hand_type(["2D", "5D", "9D"]) == "Flush"
assert table.compare_hands(["2D", "5D", "9D"], ["AS", "AH", "KC"]) == 1
assert table.compare_hands(["QH", "QD", "QC"], ["2D", "5D", "9D"]) == 1
assert table.compare_hands(["KS", "7S", "4S"], ["KD", "7D", "3D"]) == 1
assert table.rank_players([["AS", "AH", "KC"], ["2D", "5D", "9D"], ["3C", "4C", "6C"]]) == [1]
"""},
        {"name": "Part 2: random hands with flushes", "part": 2, "visibility": "unshown", "behavior": "metrics.ties",
         "failure_message": "With flushes allowed, a type, comparison or set of winners differed from the order Three of a Kind, Flush, Pair, High Card, with flushes compared like high cards.",
         "code": _CARDS + r"""
table = {fn}()
for seed in range(3000):
    rng = random.Random(seed)
    if seed % 3 == 0:
        suit = rng.choice("CDHS")
        a = [r + suit for r in rng.sample("23456789TJQKA", 3)]
    else:
        a = (biased_hand if seed % 2 else random_hand)(rng, True)
    b = random_hand(rng, True)
    assert table.hand_type(a) == NAMES[strength(a)[0]], a
    expected = (strength(a) > strength(b)) - (strength(a) < strength(b))
    assert table.compare_hands(a, b) == expected, (a, b)
for seed in range(400):
    rng = random.Random(seed)
    hands = [random_hand(rng, True) for _ in range(rng.randint(2, 8))]
    assert table.rank_players(hands) == winners(hands), hands
assert table.compare_hands(["AH", "KH", "QH"], ["AD", "KD", "QD"]) == 0, "suit never breaks a tie"
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "events.ordering", "code": r"""
never = lambda first_two: False
unless_pair = lambda first_two: first_two[0][0] != first_two[1][0]
unless_flush_draw = lambda first_two: first_two[0][1] != first_two[1][1]
table = {fn}()
table.start_round("9C 2H KS 4D 9S 7H QS 4C JD AS 3H 8S".split(), [never, unless_pair, unless_flush_draw, never])
lines = []
while not table.is_over():
    lines.append(table.deal_card())
assert lines == [
    "Player 0 is dealt 9C.", "Player 1 is dealt 2H.", "Player 2 is dealt KS.", "Player 3 is dealt 4D.",
    "Player 0 is dealt 9S.", "Player 1 is dealt 7H.", "Player 2 is dealt QS.", "Player 3 is dealt 4C.",
    "Player 0 is dealt JD.", "Player 1 folds.", "Player 2 is dealt AS.", "Player 3 is dealt 3H.",
], lines
assert table.check_result() == "Winner: Player 2"
"""},
        {"name": "Part 3: random rounds", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On a random round, a dealt line or the result differed from dealing in three passes, where a fold takes no card and the next player draws it, and only players who stayed are compared.",
         "code": _ROUND + r"""
table = {fn}()
for seed in range(500):
    rng = random.Random(seed)
    n = rng.randint(1, 15)
    deck = rng.sample(DECK, rng.randint(3 * n, min(52, 3 * n + 4)))
    strategies = [rng.choice([never, unless_pair, unless_flush_draw]) for _ in range(n)]
    assert play(table, deck, strategies) == simulate(deck, strategies), seed
"""},
        {"name": "Part 3: folds, ties and what a fold rule sees", "part": 3, "visibility": "unshown", "behavior": "edge.empty_or_boundary",
         "failure_message": "A fold rule must be called once per player in pass 3 with that player's two cards in deal order; all folding gives no winner, one survivor wins outright, and ties list the players in increasing order.",
         "code": _ROUND + r"""
table = {fn}()
seen = []
def spy(first_two):
    seen.append(list(first_two))
    return True
lines, result = play(table, ["2C", "3D", "4H", "5S", "6C", "7D"], [spy, spy])
assert seen == [["2C", "4H"], ["3D", "5S"]], seen
assert result == "No winner: all players folded."
assert lines[-2:] == ["Player 0 folds.", "Player 1 folds."]
always = lambda first_two: True
never_ = lambda first_two: False
assert play(table, "2C 3D 4H 5S 6C 7D 8H 9S TC".split(), [always, never_, always])[1] == "Winner: Player 1"
deck = "AH AD KC KH 5S 5C 2D 2S 9C 9D".split()
assert play(table, deck, [never_, never_, never_]) == simulate(deck, [never_, never_, never_])
tie_deck = "AH AD QS KC KH QD 4S 4C".split()
assert play(table, tie_deck, [never_, never_])[1] == "Winner: Tie (Players 0, 1)"
three = "2H 2D 2C 3H 3D 3C 9S 9H 9D 4S 4H 4D".split()
assert play(table, three, [never_, always, never_, never_])[1] == simulate(three, [never_, always, never_, never_])[1]
"""},
        {"name": "Part 3: a round with many players", "part": 3, "visibility": "unshown", "behavior": "performance.complexity",
         "failure_message": "Dealing a round for 30,000 players took far more than ten times as long as for 3,000; each deal_card call must do constant work.",
         "code": _ROUND + r"""
def timed(n):
    rng = random.Random(n)
    deck = [DECK[i % 52] for i in range(3 * n)]
    strategies = [rng.choice([never, unless_pair]) for _ in range(n)]
    table = {fn}()
    start = time.perf_counter()
    table.start_round(deck, strategies)
    count = 0
    while not table.is_over():
        table.deal_card()
        count += 1
    table.check_result()
    assert count == 3 * n
    return time.perf_counter() - start

ratio = min(timed(30000) for _ in range(2)) / min(timed(3000) for _ in range(2))
assert ratio < 30, f"10x the players took {ratio:.0f}x as long"
"""},
    ],
    "solution": r'''# Adapted from Schuture/OpenAI-Interview-Notes (code under the MIT License).
RANKS = "23456789TJQKA"  # the index is the rank's value
TYPE_NAMES = ["High Card", "Pair", "Flush", "Three of a Kind"]


def _key(cards):
    """One comparable tuple per hand: (type, tie-break), so tuple order is hand order."""
    ranks = sorted((RANKS.index(card[0]) for card in cards), reverse=True)
    if ranks[0] == ranks[2]:
        return (3, ranks[0])
    if len({card[1] for card in cards}) == 1:
        return (2, *ranks)
    if ranks[0] == ranks[1]:  # after sorting, a pair is at [0:2] or [1:3]
        return (1, ranks[0], ranks[2])
    if ranks[1] == ranks[2]:
        return (1, ranks[1], ranks[0])
    return (0, *ranks)


class PokerTable:
    def hand_type(self, cards):
        return TYPE_NAMES[_key(cards)[0]]

    def compare_hands(self, a, b):
        ka, kb = _key(a), _key(b)
        return (ka > kb) - (ka < kb)

    def rank_players(self, hands):
        keys = [_key(hand) for hand in hands]
        best = max(keys)
        return [i for i, k in enumerate(keys) if k == best]

    def start_round(self, deck, strategies):
        self._deck, self._strategies = deck, strategies
        self._hands = [[] for _ in strategies]
        self._folded = [False] * len(strategies)
        self._next_card = 0  # moves only when a card is dealt, so a fold costs no card
        self._calls = 0

    def is_over(self):
        return self._calls >= 3 * len(self._strategies)

    def deal_card(self):
        round_no, player = divmod(self._calls, len(self._strategies))
        self._calls += 1
        if round_no == 2 and self._strategies[player](list(self._hands[player])):
            self._folded[player] = True
            return f"Player {player} folds."
        card = self._deck[self._next_card]
        self._next_card += 1
        self._hands[player].append(card)
        return f"Player {player} is dealt {card}."

    def check_result(self):
        alive = [p for p, folded in enumerate(self._folded) if not folded]
        if not alive:
            return "No winner: all players folded."
        best = self.rank_players([self._hands[p] for p in alive])
        if len(best) == 1:
            return f"Winner: Player {alive[best[0]]}"
        return "Winner: Tie (Players " + ", ".join(str(alive[i]) for i in best) + ")"
''',
    "interview_questions": interview(
        concept=[
            "How do you turn a hand into one value whose order matches the ranking rules?",
            "Why must a pair's tie-break compare the pair's rank before the third card?",
        ],
        deep_dive=[
            "How do you find every tied winner among 100,000 hands in one pass?",
        ],
        tradeoffs=[
            "What changes in your key when a new hand type is inserted between two existing ones?",
            "Why does a fold not consume a card, and how do you make sure the next player gets it?",
            "How would the key change for five-card hands with two pair, straights and full houses?",
            "How would you choose a fold rule that does better than the fixed ones?",
        ],
    ),
}
