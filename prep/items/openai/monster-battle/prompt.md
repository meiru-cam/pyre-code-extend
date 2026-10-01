Team `A` and Team `B` each have a non-empty, ordered list of monsters. Every monster starts with a positive integer HP and a positive integer attack power. Complete the following three parts.

### Part 1 — Plain attacks

A monster is *alive* while its HP is above 0, and *eliminated* once its HP drops to 0 or below. A team is *defeated* once every monster in its list is eliminated.

The battle is a fixed sequence of individual attacks:

- Team `A` attacks first; the two teams alternate after every single attack, whether or not that attack eliminated anyone.
- On a team's turn, the *attacker* is the first alive monster in its list (index 0 first), and the *defender* is the first alive monster in the other team's list.
- The defender's HP decreases by exactly the attacker's attack power; the attacker takes no damage back.
- The battle ends the moment one team is defeated. (Only the side being hit ever loses HP, so the two teams cannot become defeated on the same attack.)

Record one line for every event, in exactly this format (the line after `damage.` depends on whether the defender is still alive):

```text
Match start: {A.name} takes on {B.name}
{attacker.name} hits {defender.name} for {damage} damage. {defender.name} has {defender.hp} HP left.
{attacker.name} hits {defender.name} for {damage} damage. {defender.name} is defeated!
Match over: {winner.name} wins!
```

```py
class Monster:
    def __init__(self, name: str, hp: int, attack: int):
        """hp and attack are positive integers."""

    def is_alive(self) -> bool: ...
    def take_damage(self, damage: int) -> None: ...


class Team:
    def __init__(self, name: str, monsters: list[Monster]):
        """monsters is the attack/defense order; it is non-empty."""

    def first_alive(self) -> "Monster | None": ...
    def is_defeated(self) -> bool: ...


def run_battle(team_a: Team, team_b: Team) -> list[str]:
    """Simulates the battle and returns the event log, one string per line."""
```

Example: team `Vanguard` has a single monster `Talonis` (hp 25, attack 14); team `Marsh` has `Sable` (hp 10, attack 5) followed by `Thornback` (hp 20, attack 6). `run_battle` returns:

```text
Match start: Vanguard takes on Marsh
Talonis hits Sable for 14 damage. Sable is defeated!
Thornback hits Talonis for 6 damage. Talonis has 19 HP left.
Talonis hits Thornback for 14 damage. Thornback has 6 HP left.
Thornback hits Talonis for 6 damage. Talonis has 13 HP left.
Talonis hits Thornback for 14 damage. Thornback is defeated!
Match over: Vanguard wins!
```

### Part 2 — Elemental types

Every monster also has an *element*: one of `Ember`, `Tide`, `Bramble` or `Spark`. The elements form a cycle of advantage — `Ember` is strong against `Bramble`, `Bramble` is strong against `Tide`, `Tide` is strong against `Spark`, and `Spark` is strong against `Ember`. Being strong against an element deals double damage to it (`2x`); the reverse direction deals half damage (`0.5x`). Every other ordered pair, including a monster's element against itself, is neutral (`1x`). The full table, attacker's element in the row, defender's element in the column:

- : Ember · Ember: 1x · Tide: 1x · Bramble: 2x · Spark: 0.5x
- : Tide · Ember: 1x · Tide: 1x · Bramble: 0.5x · Spark: 2x
- : Bramble · Ember: 0.5x · Tide: 2x · Bramble: 1x · Spark: 1x
- : Spark · Ember: 2x · Tide: 0.5x · Bramble: 1x · Spark: 1x

Damage is $\lfloor \text{attack} \times \text{multiplier} \rfloor$; if that value is below 1, use 1 instead. The attacker and defender are still chosen exactly as in Part 1. Each hit line now also names the multiplier that applied, written with no trailing `.0` (`2x`, `0.5x`, `1x`):

```text
{attacker.name} hits {defender.name} for {damage} damage ({multiplier}). {defender.name} has {defender.hp} HP left.
{attacker.name} hits {defender.name} for {damage} damage ({multiplier}). {defender.name} is defeated!
```

```py
class ElementType(Enum):
    EMBER = "Ember"
    TIDE = "Tide"
    BRAMBLE = "Bramble"
    SPARK = "Spark"


class Monster:
    def __init__(self, name: str, hp: int, attack: int, element: ElementType): ...
    def calculate_damage(self, defender: "Monster") -> int:
        """Damage this monster deals to defender, after the type chart and the rounding rule above."""
```

Example: team `Wildfire` has `Cinder` (hp 30, attack 9, `Ember`); team `Thicket` has `Mossken` (hp 26, attack 7, `Bramble`).

```text
Match start: Wildfire takes on Thicket
Cinder hits Mossken for 18 damage (2x). Mossken has 8 HP left.
Mossken hits Cinder for 3 damage (0.5x). Cinder has 27 HP left.
Cinder hits Mossken for 18 damage (2x). Mossken is defeated!
Match over: Wildfire wins!
```

### Part 3 — Smart targeting

The attacking team no longer always leads with its first alive monster. Instead, among all of its alive monsters, it picks the one whose `calculate_damage` against the current defender is highest; if several tie for the highest, the one that appears earliest in the list attacks. The defender is unchanged from Parts 1 and 2: whichever monster is alive and comes first in the defending team's list.

```py
class Team:
    def best_attacker_against(self, defender: Monster) -> "Monster | None":
        """The living monster on this team with the highest calculate_damage(defender);
        ties go to whichever of them appears first in the list."""
```

Example: team `Skyfleet` has `Talos` (hp 20, attack 10, `Spark`) followed by `Draketh` (hp 25, attack 12, `Ember`); team `Depths` has `Coraline` (hp 22, attack 8, `Tide`).

```text
Match start: Skyfleet takes on Depths
Draketh hits Coraline for 12 damage (1x). Coraline has 10 HP left.
Coraline hits Talos for 16 damage (2x). Talos has 4 HP left.
Draketh hits Coraline for 12 damage (1x). Coraline is defeated!
Match over: Skyfleet wins!
```
