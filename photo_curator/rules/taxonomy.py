"""Canonical zero-shot taxonomy and prompt text for OpenCLIP.

``CategoryType`` is the closed set of routing buckets. ``TAXONOMY_PROMPTS``
maps each class to the natural-language prompt(s) fed to the CLIP text tower.
Each category also exposes its target sub-folder for the FolderOrganizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CategoryType(str, Enum):
    """Closed set of routing buckets (mirrors the spec decision matrix)."""

    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    GRAPHIC = "graphic"            # meme / poster / digital graphic
    FOOD = "food"
    ANIMAL = "animal"
    SCENERY = "scenery"
    PEOPLE = "people"
    ACCIDENTAL = "accidental"      # dark / blurry junk (quality-driven)
    UNCERTAIN = "uncertain"
    DUPLICATE = "duplicate"        # exact sha256 match
    UNKNOWN = "unknown"

    @property
    def folder(self) -> str:
        """Relative target folder under the organize root (Option 1)."""
        return {
            CategoryType.SCREENSHOT: "Junk_Review/Screenshots",
            CategoryType.DOCUMENT: "Junk_Review/Documents",
            CategoryType.GRAPHIC: "Junk_Review/Graphics",
            CategoryType.FOOD: "Non-Human/Food",
            CategoryType.ANIMAL: "Non-Human/Animals",
            CategoryType.SCENERY: "Non-Human/Scenery",
            CategoryType.PEOPLE: "People",
            CategoryType.ACCIDENTAL: "Junk_Review/Accidental",
            CategoryType.UNCERTAIN: "Uncertain_Review",
            CategoryType.DUPLICATE: "Junk_Review/Duplicates",
            CategoryType.UNKNOWN: "Unknown_Date",
        }[self]

    @property
    def is_junk(self) -> bool:
        return self in (
            CategoryType.SCREENSHOT, CategoryType.DOCUMENT,
            CategoryType.GRAPHIC, CategoryType.ACCIDENTAL,
            CategoryType.DUPLICATE,
        )

    @property
    def is_date_grouped(self) -> bool:
        """People / Food / Scenery / Animals get YYYY/MM/DD date subfolders."""
        return self in (
            CategoryType.PEOPLE, CategoryType.FOOD,
            CategoryType.SCENERY, CategoryType.ANIMAL,
        )


@dataclass(frozen=True)
class TaxonomyEntry:
    category: CategoryType
    prompts: tuple[str, ...]
    # Optional explicit folder override (e.g. for a custom taxonomy).
    folder_override: str | None = None


# ---------------------------------------------------------------------------
# Canonical taxonomy (per spec section 4.A).
# ---------------------------------------------------------------------------
TAXONOMY: tuple[TaxonomyEntry, ...] = (
    TaxonomyEntry(
        CategoryType.SCREENSHOT,
        (
            "a screenshot of a mobile phone screen",
            "a screenshot of a computer desktop or window",
            "a UI screen capture with text and buttons",
        ),
    ),
    TaxonomyEntry(
        CategoryType.DOCUMENT,
        (
            "a document, receipt, contract, or invoice with text",
            "a photograph of a paper with printed text",
        ),
    ),
    TaxonomyEntry(
        CategoryType.GRAPHIC,
        (
            "a meme, cartoon, poster, or digital graphic design",
            "a flat colored illustration or logo",
        ),
    ),
    TaxonomyEntry(
        CategoryType.FOOD,
        (
            "a close-up photo of food, a meal, or a restaurant dish",
            "a plate of prepared food",
        ),
    ),
    TaxonomyEntry(
        CategoryType.ANIMAL,
        (
            "a domestic pet, dog, or cat",
            "a wild animal or bird",
        ),
    ),
    TaxonomyEntry(
        CategoryType.SCENERY,
        (
            "a landscape, mountain, beach, or nature scenery",
            "an architectural or cityscape photo",
        ),
    ),
    TaxonomyEntry(
        CategoryType.PEOPLE,
        (
            "a portrait or selfie of a person",
            "a group photo of people together",
        ),
    ),
)

# Prompt lists keyed by category for quick lookup / the CLIP text tower.
TAXONOMY_PROMPTS: dict[CategoryType, tuple[str, ...]] = {
    e.category: e.prompts for e in TAXONOMY
}

# Categories that participate in CLIP zero-shot scoring (junk "accidental" and
# "uncertain"/"duplicate" are assigned by rules, not by the model).
CLIP_CATEGORIES: tuple[CategoryType, ...] = tuple(
    e.category for e in TAXONOMY
)

# Flat, ordered prompt list (one per category) for vectorized CLIP encoding.
def flat_prompts() -> list[str]:
    """One representative prompt per CLIP category, aligned to CLIP_CATEGORIES."""
    return [TAXONOMY_PROMPTS[c][0] for c in CLIP_CATEGORIES]


def category_to_folder(cat: CategoryType, folder_override: str | None = None) -> str:
    if folder_override:
        return folder_override
    return cat.folder
