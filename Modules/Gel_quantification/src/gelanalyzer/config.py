"""What the user tells us about a gel before analysis (US-4).

Can be written as a plain dict, e.g. read from a JSON/YAML file or a form:

    {"n_lanes": 15, "ladder_lanes": [4], "ladder": "Precision Plus Unstained",
     "gel_type": "4-20% gradient"}

Lanes are numbered like on the bench: 1 = leftmost.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ladder:
    """A molecular-weight marker: band sizes from top (largest) to bottom."""

    name: str
    kda: tuple[float, ...]
    thick: tuple[float, ...] = ()        # bands that are deliberately more intense (reference bands)

    def __len__(self) -> int:
        return len(self.kda)


LADDERS: dict[str, Ladder] = {
    "Precision Plus Unstained": Ladder(
        "Precision Plus Unstained", (250, 150, 100, 75, 50, 37, 25, 20, 15, 10), thick=(75, 50, 25)),
}


@dataclass(frozen=True)
class GelConfig:
    n_lanes: int
    ladder_lanes: tuple[int, ...] = ()   # 1-based lane numbers that hold a ladder
    ladder: Ladder | None = None
    gel_type: str = ""                   # free text, e.g. "4-20% gradient"; affects kDa fitting later

    def __post_init__(self):
        if self.n_lanes < 1:
            raise ValueError(f"n_lanes must be at least 1, got {self.n_lanes}")
        bad = [lane for lane in self.ladder_lanes if not 1 <= lane <= self.n_lanes]
        if bad:
            raise ValueError(f"ladder lane(s) {bad} outside 1..{self.n_lanes}")

    @classmethod
    def from_dict(cls, data: dict) -> GelConfig:
        ladder = data.get("ladder")
        if isinstance(ladder, str):
            if ladder not in LADDERS:
                raise ValueError(f"unknown ladder {ladder!r}; known: {', '.join(LADDERS)}")
            ladder = LADDERS[ladder]
        return cls(n_lanes=int(data["n_lanes"]),
                   ladder_lanes=tuple(int(x) for x in data.get("ladder_lanes", ())),
                   ladder=ladder,
                   gel_type=data.get("gel_type", ""))
