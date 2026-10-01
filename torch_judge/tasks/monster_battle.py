"""A turn-based monster battle whose log grows rules part by part."""

from ._interview import interview

# A second implementation written from the statement, sharing no code with the reference.
_HELPERS = r"""
import random

BEATS = [("Ember", "Bramble"), ("Bramble", "Tide"), ("Tide", "Spark"), ("Spark", "Ember")]

def slow_battle(team_a, team_b, smart=False):
    sides = [[team_a[0], [list(m) + [None] * (4 - len(m)) for m in team_a[1]]],
             [team_b[0], [list(m) + [None] * (4 - len(m)) for m in team_b[1]]]]
    log = [f"Match start: {team_a[0]} takes on {team_b[0]}"]
    turn = 0
    while True:
        attackers, defenders = sides[turn][1], sides[1 - turn][1]
        defender = [m for m in defenders if m[1] > 0][0]
        def power(m):
            if m[3] is None or defender[3] is None:
                return 1, ""
            if (m[3], defender[3]) in BEATS:
                return 2, " (2x)"
            if (defender[3], m[3]) in BEATS:
                return 0.5, " (0.5x)"
            return 1, " (1x)"
        def damage(m):
            return max(1, int(m[2] * power(m)[0]))
        alive = [m for m in attackers if m[1] > 0]
        attacker = alive[0]
        if smart:
            for m in alive:
                if damage(m) > damage(attacker):
                    attacker = m
        hit = damage(attacker)
        defender[1] -= hit
        line = f"{attacker[0]} hits {defender[0]} for {hit} damage{power(attacker)[1]}. "
        log.append(line + (f"{defender[0]} has {defender[1]} HP left." if defender[1] > 0 else f"{defender[0]} is defeated!"))
        if all(m[1] <= 0 for m in defenders):
            log.append(f"Match over: {sides[turn][0]} wins!")
            return log
        turn = 1 - turn

def random_team(rng, name, elements):
    monsters = []
    for i in range(rng.randint(1, 4)):
        monster = (f"{name}{i}", rng.randint(1, 30), rng.randint(1, 12))
        if elements:
            monster += (rng.choice(["Ember", "Tide", "Bramble", "Spark"]),)
        monsters.append(monster)
    return (name, monsters)

def raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False
"""

TASK = {
    "title": "Monster Battle",
    "difficulty": "Medium",
    "version": 1,
    "function_name": "run_battle",
    "description_en": r"""Write `run_battle(team_a, team_b)`, which plays out a battle between two teams of monsters and returns the event log.

The requirement arrives in parts. Each part keeps every earlier behavior, so one function passes all parts at the end. Pass every test of the current part to reveal the next one.

**Rules for every part:**
- A team is a tuple `(name, monsters)`, where `monsters` is a non-empty list in a fixed order. A monster is a tuple `(name, hp, attack)` with positive integers.
- A monster with HP of 0 or less is out of the fight. A team is beaten once all of its monsters are out.
- Team A attacks first, then the teams alternate, one attack each, whatever happens.
- On a turn, the attacker comes from the attacking team and the defender is the first alive monster of the other team. The defender loses HP equal to the damage; the attacker loses nothing.
- The battle stops as soon as a team is beaten. Never modify the arguments.

────────────────────────────────

**Background — context only. Everything above this line is the requirement.**

**Why this shows up in interviews:** the simulation is easy, the exercise is about clean objects and an exact output format, and each later part adds one requirement.

**Where it is used:** turn-based games and any rules engine whose output is checked line by line.

Adapted from the monster battle question in Schuture/OpenAI-Interview-Notes (CC BY-NC 4.0), reworded, on one function with plain tuples, with later rules made opt-in so the parts build on each other.""",
    "parts": [
        {
            "title": "Plain attacks",
            "description_en": r"""**Signature:** `run_battle(team_a, team_b) -> list[str]`

- The attacker is the first alive monster of the attacking team, and the damage is its `attack`.
- The log starts with `Match start: {A} takes on {B}`, using the team names.
- Each attack adds `{attacker} hits {defender} for {damage} damage. {defender} has {hp} HP left.`, or, when the defender's HP reaches 0 or below, `{attacker} hits {defender} for {damage} damage. {defender} is defeated!`
- The log ends with `Match over: {winner} wins!`

**Example:** `run_battle(("Harbor", [("Kestrel", 22, 9)]), ("Ridge", [("Pebble", 8, 4), ("Boulder", 15, 5)]))` returns:
- `Match start: Harbor takes on Ridge`
- `Kestrel hits Pebble for 9 damage. Pebble is defeated!`
- `Boulder hits Kestrel for 5 damage. Kestrel has 17 HP left.`
- `Kestrel hits Boulder for 9 damage. Boulder has 6 HP left.`
- `Boulder hits Kestrel for 5 damage. Kestrel has 12 HP left.`
- `Kestrel hits Boulder for 9 damage. Boulder is defeated!`
- `Match over: Harbor wins!`""",
        },
        {
            "title": "Elements",
            "description_en": r"""Keep Part 1. A monster may have a fourth field, its element: `"Ember"`, `"Tide"`, `"Bramble"` or `"Spark"`. If any monster has another value, `run_battle` raises `ValueError` before the first attack.

- Each element is strong against one other: Ember beats Bramble, Bramble beats Tide, Tide beats Spark, Spark beats Ember.
- When the attacker and the defender both have an element, the multiplier is `2` if the attacker's element beats the defender's, `0.5` if the defender's beats the attacker's, and `1` otherwise, including the same element. Without both elements it is `1`.
- Damage is `attack` times the multiplier, rounded down, and never less than `1`.
- When both have an element, the hit line names the multiplier after the damage: `for 14 damage (2x).`, `(0.5x)` or `(1x)`. Otherwise the line is exactly as in Part 1.

**Example:** `run_battle(("Kiln", [("Flint", 28, 7, "Ember")]), ("Glade", [("Fern", 20, 5, "Bramble")]))` returns:
- `Match start: Kiln takes on Glade`
- `Flint hits Fern for 14 damage (2x). Fern has 6 HP left.`
- `Fern hits Flint for 2 damage (0.5x). Flint has 26 HP left.`
- `Flint hits Fern for 14 damage (2x). Fern is defeated!`
- `Match over: Kiln wins!`""",
        },
        {
            "title": "Smart attackers",
            "description_en": r"""**Signature:** `run_battle(team_a, team_b, smart=False) -> list[str]`

Keep Parts 1–2. With `smart=True`, both teams choose their attackers better:

- The attacker is the alive monster that deals the most damage to the current defender. On a tie, the one listed first attacks.
- The defender is still the first alive monster of the other team.
- With `smart=False` nothing changes.

**Example:** with `smart=True`, team `Sky` has `Volt` (20 HP, attack 10, Spark) then `Drake` (25 HP, attack 12, Ember), and team `Deep` has `Coral` (22 HP, attack 8, Tide). `Drake` attacks first: it deals 12 to `Coral`, while `Volt` would deal only 5.""",
        },
    ],
    "hints": [
        {"level": 1, "kind": "questions", "content": "What state changes during the battle, and how do you avoid changing the tuples you were given? How do you find the first alive monster of a team? Which check ends the battle, and when exactly does it run?"},
        {"level": 2, "kind": "analysis", "content": "Copy each monster into a small mutable record. Loop with a turn index: pick the first alive attacker and defender, subtract the attack, append the right line, and stop with the winner line as soon as the defending team has no alive monster left; otherwise switch sides."},
    ],
    "model_connections": [
        "Turn-based games such as Pokémon resolve type advantages with a lookup table and log every action in a fixed format.",
        "Rules engines in billing or games are tested with golden logs, so an exact output format is part of the contract.",
    ],
    "pro_con_analysis": {
        "pros": [
            "Small Monster and Team objects keep turn logic, damage and formatting in separate places.",
            "A table of which element beats which makes the multiplier one lookup and easy to extend.",
            "Making new rules opt-in keeps every earlier output unchanged.",
        ],
        "cons": [
            "An exact log format makes any wording change a breaking change.",
            "Choosing the best attacker scans the whole team every turn.",
            "Opt-in flags multiply the combinations a test suite has to cover.",
        ],
    },
    "tests": [
        {"name": "Part 1: the worked example", "part": 1, "behavior": "events.ordering", "code": r"""
assert {fn}(("Harbor", [("Kestrel", 22, 9)]), ("Ridge", [("Pebble", 8, 4), ("Boulder", 15, 5)])) == [
    "Match start: Harbor takes on Ridge",
    "Kestrel hits Pebble for 9 damage. Pebble is defeated!",
    "Boulder hits Kestrel for 5 damage. Kestrel has 17 HP left.",
    "Kestrel hits Boulder for 9 damage. Boulder has 6 HP left.",
    "Boulder hits Kestrel for 5 damage. Kestrel has 12 HP left.",
    "Kestrel hits Boulder for 9 damage. Boulder is defeated!",
    "Match over: Harbor wins!",
]
"""},
        {"name": "Part 1: exact kills, B wins, and arguments untouched", "part": 1, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "A monster at exactly 0 HP is defeated, team B can win, and the teams passed in must not change.",
         "code": r"""
team_a = ("Left", [("Ant", 3, 2), ("Bee", 4, 1)])
team_b = ("Right", [("Cat", 4, 3)])
assert {fn}(team_a, team_b) == [
    "Match start: Left takes on Right",
    "Ant hits Cat for 2 damage. Cat has 2 HP left.",
    "Cat hits Ant for 3 damage. Ant is defeated!",
    "Bee hits Cat for 1 damage. Cat has 1 HP left.",
    "Cat hits Bee for 3 damage. Bee has 1 HP left.",
    "Bee hits Cat for 1 damage. Cat is defeated!",
    "Match over: Left wins!",
]
assert team_a == ("Left", [("Ant", 3, 2), ("Bee", 4, 1)]) and team_b == ("Right", [("Cat", 4, 3)])
assert {fn}(("Weak", [("Gnat", 1, 1)]), ("Strong", [("Ogre", 5, 7)]))[-1] == "Match over: Strong wins!"
"""},
        {"name": "Part 1: random battles match a second implementation", "part": 1, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On random teams, the log differed from an independent simulation of the rules.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(seed)
    a, b = random_team(rng, "A", False), random_team(rng, "B", False)
    assert {fn}(a, b) == slow_battle(a, b), (seed, a, b)
"""},
        {"name": "Part 2: the worked example", "part": 2, "behavior": "events.ordering", "code": r"""
assert {fn}(("Kiln", [("Flint", 28, 7, "Ember")]), ("Glade", [("Fern", 20, 5, "Bramble")])) == [
    "Match start: Kiln takes on Glade",
    "Flint hits Fern for 14 damage (2x). Fern has 6 HP left.",
    "Fern hits Flint for 2 damage (0.5x). Flint has 26 HP left.",
    "Flint hits Fern for 14 damage (2x). Fern is defeated!",
    "Match over: Kiln wins!",
]
"""},
        {"name": "Part 2: the whole element table", "part": 2, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "One pair of elements gave the wrong multiplier, rounding or label; check the cycle Ember > Bramble > Tide > Spark > Ember and the minimum damage of 1.",
         "code": _HELPERS + r"""
elements = ["Ember", "Tide", "Bramble", "Spark"]
for attack_element in elements:
    for defend_element in elements:
        for attack in (1, 5):
            log = {fn}(("A", [("X", 50, attack, attack_element)]), ("B", [("Y", 99, 1, defend_element)]))
            expected = slow_battle(("A", [("X", 50, attack, attack_element)]), ("B", [("Y", 99, 1, defend_element)]))
            assert log[1] == expected[1], (attack_element, defend_element, attack, log[1])
"""},
        {"name": "Part 2: mixed teams and bad elements", "part": 2, "visibility": "unshown", "behavior": "protocol.validation",
         "failure_message": "A hit where one side has no element is plain Part 1 output, and an unknown element must raise ValueError.",
         "code": _HELPERS + r"""
log = {fn}(("A", [("X", 9, 4, "Tide")]), ("B", [("Y", 3, 2)]))
assert log[1] == "X hits Y for 4 damage. Y is defeated!", log
assert raises_value_error({fn}, ("A", [("X", 9, 4, "Water")]), ("B", [("Y", 3, 2)]))
assert raises_value_error({fn}, ("A", [("X", 9, 4)]), ("B", [("Y", 3, 2, "ember")]))
"""},
        {"name": "Part 2: random battles with elements", "part": 2, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On random teams with elements, the log differed from an independent simulation of the rules.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(1000 + seed)
    a, b = random_team(rng, "A", True), random_team(rng, "B", rng.random() < 0.8)
    assert {fn}(a, b) == slow_battle(a, b), (seed, a, b)
"""},
        {"name": "Part 3: the worked example", "part": 3, "behavior": "events.ordering", "code": r"""
sky = ("Sky", [("Volt", 20, 10, "Spark"), ("Drake", 25, 12, "Ember")])
deep = ("Deep", [("Coral", 22, 8, "Tide")])
assert {fn}(sky, deep, smart=True) == [
    "Match start: Sky takes on Deep",
    "Drake hits Coral for 12 damage (1x). Coral has 10 HP left.",
    "Coral hits Volt for 16 damage (2x). Volt has 4 HP left.",
    "Drake hits Coral for 12 damage (1x). Coral is defeated!",
    "Match over: Sky wins!",
]
assert {fn}(sky, deep)[1] == "Volt hits Coral for 5 damage (0.5x). Coral has 17 HP left."
"""},
        {"name": "Part 3: ties go to the first listed", "part": 3, "visibility": "unshown", "behavior": "state.invariant",
         "failure_message": "With smart=True the strongest attacker against the current defender attacks, the earliest listed on a tie, and dead monsters never attack.",
         "code": r"""
log = {fn}(("A", [("P", 9, 4), ("Q", 9, 6), ("R", 9, 6)]), ("B", [("S", 12, 5)]), smart=True)
assert log[1] == "Q hits S for 6 damage. S has 6 HP left.", log
assert log[2] == "S hits P for 5 damage. P has 4 HP left.", "the defender is still the first alive"
log = {fn}(("A", [("P", 1, 1), ("Q", 10, 2)]), ("B", [("S", 30, 9), ("T", 1, 20)]), smart=True)
assert log[2] == "T hits P for 20 damage. P is defeated!" and log[3].startswith("Q hits S"), log
"""},
        {"name": "Part 3: random smart battles", "part": 3, "visibility": "unshown", "behavior": "events.ordering",
         "failure_message": "On random teams with smart=True, the log differed from an independent simulation of the rules.",
         "code": _HELPERS + r"""
for seed in range(300):
    rng = random.Random(2000 + seed)
    a, b = random_team(rng, "A", rng.random() < 0.8), random_team(rng, "B", True)
    assert {fn}(a, b, smart=True) == slow_battle(a, b, smart=True), (seed, a, b)
    assert {fn}(a, b) == slow_battle(a, b), seed
"""},
    ],
    "solution": r'''ELEMENTS = ("Ember", "Tide", "Bramble", "Spark")
BEATS = {"Ember": "Bramble", "Bramble": "Tide", "Tide": "Spark", "Spark": "Ember"}


class Monster:
    def __init__(self, name, hp, attack, element=None):
        if element is not None and element not in ELEMENTS:
            raise ValueError(f"unknown element: {element!r}")
        self.name, self.hp, self.attack, self.element = name, hp, attack, element

    def is_alive(self):
        return self.hp > 0

    def multiplier(self, defender):
        """The multiplier against defender, or None when one of the two has no element."""
        if self.element is None or defender.element is None:
            return None
        if BEATS[self.element] == defender.element:
            return 2
        if BEATS[defender.element] == self.element:
            return 0.5
        return 1

    def damage_to(self, defender):
        return max(1, int(self.attack * (self.multiplier(defender) or 1)))


class Team:
    def __init__(self, name, monsters):
        self.name = name
        self.monsters = [Monster(*spec) for spec in monsters]

    def alive(self):
        return [m for m in self.monsters if m.is_alive()]

    def is_beaten(self):
        return not self.alive()

    def attacker_against(self, defender, smart):
        alive = self.alive()
        if not smart:
            return alive[0]
        return max(alive, key=lambda m: m.damage_to(defender))  # max keeps the first on a tie


def run_battle(team_a, team_b, smart=False):
    attacking, defending = Team(*team_a), Team(*team_b)
    log = [f"Match start: {attacking.name} takes on {defending.name}"]
    while True:
        defender = defending.alive()[0]
        attacker = attacking.attacker_against(defender, smart)
        damage = attacker.damage_to(defender)
        defender.hp -= damage
        multiplier = attacker.multiplier(defender)
        label = "" if multiplier is None else f" ({multiplier:g}x)"
        outcome = f"{defender.name} has {defender.hp} HP left." if defender.is_alive() else f"{defender.name} is defeated!"
        log.append(f"{attacker.name} hits {defender.name} for {damage} damage{label}. {outcome}")
        if defending.is_beaten():
            log.append(f"Match over: {attacking.name} wins!")
            return log
        attacking, defending = defending, attacking
''',
    "interview_questions": interview(
        concept=[
            "Which objects would you model, and which of them owns the HP that changes during the battle?",
            "How do you make sure the battle stops on the attack that beats a team, and not one turn later?",
        ],
        deep_dive=[
            "How do you keep the log format exact, and how would you test it?",
        ],
        tradeoffs=[
            "How do you represent which element beats which so that adding a fifth element is a one-line change?",
            "How do you add smart targeting without changing the output of battles that do not ask for it?",
            "Why does picking the attacker by maximum damage need an explicit rule for ties?",
        ],
    ),
}
