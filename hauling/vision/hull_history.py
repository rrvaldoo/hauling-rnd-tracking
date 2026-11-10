"""Bounded, causal OCR evidence; nothing after finish can affect its event."""

from collections import defaultdict
from dataclasses import replace

from hauling.vision.plate import HullObservation, PlateCandidate


def quality(candidate: PlateCandidate) -> tuple:
    characters = candidate.characters
    known = [c.confidence for c in characters if c.text != "?"]
    if not characters:
        return (0, 0., candidate.confidence, candidate.timestamp_s)
    # Complete, strong reads beat small clear fragments and weak whole strings.
    score = sum(known) / 7 + .3 * min(known, default=0)
    return (int(len(characters) == 7), score, candidate.confidence, candidate.timestamp_s)


def consensus_from_observations(anchor, observations):
    """Vote among comparably clear whole reads, excluding a mass of weak crops.

    A weak slot can indicate shifted/misaligned boxes for the entire reading.
    The weakest glyph therefore defines its whole-read reliability. Scores
    remain conservative, not calibrated word probabilities.
    """
    eligible = [p for p in observations if len(p.text) == 7 and p.confidence is not None]
    if len(anchor) != 7 or not eligible:
        return tuple(anchor), "unread" if not anchor else "partial"
    maximum_strength = max(p.confidence for p in eligible)
    comparable = [p for p in eligible if p.confidence >= maximum_strength * .9]
    characters = []
    for index, original in enumerate(anchor):
        votes, maximum = defaultdict(float), defaultdict(float)
        for reading in comparable:
            text = reading.text[index]
            if text != "?":
                votes[text] += reading.confidence ** 2
                maximum[text] = max(maximum[text], reading.confidence)
        if not votes or sum(votes.values()) <= 0:
            characters.append(replace(original, text="?", confidence=0.))
            continue
        text = max(votes, key=votes.get)
        support = votes[text] / sum(votes.values())
        characters.append(replace(original, text=text if support >= .65 else "?",
                                  confidence=min(maximum[text], support)))
    status = "complete" if all(c.text != "?" for c in characters) else "partial"
    return tuple(characters), status


class HullReadHistory:
    def __init__(self, maximum_samples: int = 32):
        self.samples: list[PlateCandidate] = []
        self.maximum_samples = maximum_samples

    @property
    def best(self) -> PlateCandidate | None:
        return max(self.samples, key=quality, default=None)

    def add(self, candidate: PlateCandidate) -> None:
        if not candidate.valid:
            return
        self.samples.append(candidate)
        if len(self.samples) > self.maximum_samples:
            # Keep the strongest evidence rather than aging out a clear early read.
            self.samples.remove(min(self.samples, key=quality))

    def resolve(self) -> PlateCandidate | None:
        best = self.best
        if best is None:
            return None
        observations = tuple(HullObservation(p.timestamp_s, p.text, p.text_status, p.text_confidence)
                             for p in sorted(self.samples, key=lambda p: p.timestamp_s))
        if len(best.characters) != 7:
            return replace(best, observations=observations)
        characters, status = consensus_from_observations(best.characters, observations)
        return replace(best, characters=characters, text_status=status, observations=observations)
