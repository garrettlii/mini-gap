"""Skill registry and prompt catalog. OWNER: done"""

from __future__ import annotations

from gap.skills.base import PORT_TYPES, Skill

_REGISTRY: dict[str, type[Skill]] = {}


def register(cls: type[Skill]) -> type[Skill]:
    for port_map in (cls.inputs, cls.outputs):
        bad = set(port_map.values()) - PORT_TYPES
        if bad:
            raise TypeError(f"{cls.name}: unknown port types {bad}")
    if cls.name in _REGISTRY:
        raise KeyError(f"duplicate skill {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Skill]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown skill {name!r}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_skills() -> dict[str, type[Skill]]:
    return dict(_REGISTRY)


def catalog_text() -> str:
    """The skill catalog agents see. Deterministic order so prompts cache well."""
    return "\n".join(_REGISTRY[n].catalog_entry() for n in names())
