"""Threat detection logic module."""

from __future__ import annotations

from typing import Iterable


class ThreatAnalyzer:
    """Convert detected objects and behavior patterns into threat levels."""

    def __init__(self) -> None:
        self.levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def analyze(self, detections: Iterable[object], behavior_reasons: Iterable[str]) -> tuple[str, list[str]]:
        """Return the threat level and the matched reasons."""
        reasons: list[str] = []
        score = 0
        names = [d.name.lower() for d in detections]

        person_count = sum(1 for name in names if name == "person")
        helmet_count = sum(1 for name in names if name in {"helmet"})
        mask_count = sum(1 for name in names if name in {"mask", "face mask"})
        backpack_count = sum(1 for name in names if name == "backpack")
        motorcycle_count = sum(1 for name in names if name == "motorcycle")
        phone_count = sum(1 for name in names if name in {"mobile phone", "cell phone", "phone"})

        if person_count == 0:
            return "LOW", []

        if person_count > 1:
            reasons.append("Multiple suspicious persons")
            score += 2
        if helmet_count:
            reasons.append("Helmet detected")
            score += 2
        if mask_count:
            reasons.append("Face mask detected")
            score += 2
        if backpack_count:
            reasons.append("Backpack detected")
            score += 1
        if motorcycle_count:
            reasons.append("Motorcycle detected")
            score += 1
        if phone_count:
            reasons.append("Mobile phone detected")
            score += 1

        for behavior_reason in behavior_reasons:
            reasons.append(behavior_reason)
            if behavior_reason in {"Person loitering near ATM", "Sudden running after entering"}:
                score += 3
            elif behavior_reason in {"Person covering camera", "ATM vandalism", "Aggressive behaviour"}:
                score += 4
            elif behavior_reason == "Object left unattended":
                score += 2

        if helmet_count and mask_count:
            reasons.append("Helmet + mask combination")
            score += 3

        if score >= 10:
            threat_level = "CRITICAL"
        elif score >= 6:
            threat_level = "HIGH"
        elif score >= 3:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        return threat_level, reasons
