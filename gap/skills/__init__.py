"""Importing this package registers every skill."""

from gap.skills import grasp, motion, perception, planning  # noqa: F401
from gap.skills.registry import all_skills, catalog_text, get, names  # noqa: F401
