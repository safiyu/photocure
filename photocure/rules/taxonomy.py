"""Canonical zero-shot taxonomy and prompt text for OpenCLIP.

``CategoryType`` is the closed set of routing buckets. ``TAXONOMY_PROMPTS``
maps each class to the natural-language prompt(s) fed to the CLIP text tower.
Each category also exposes its target sub-folder for the FolderOrganizer.

The CLIP class list can be replaced by a user-editable YAML (or JSON) file,
see :func:`load_taxonomy`. Custom class names that are not in ``CategoryType``
route through ``CategoryType.CUSTOM`` with their own folder.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CategoryType(str, Enum):
    """Closed set of routing buckets (mirrors the spec decision matrix)."""

    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    GRAPHIC = "graphic"            # meme / poster / digital graphic
    FOOD = "food"
    ANIMAL = "animal"
    SCENERY = "scenery"
    PEOPLE = "people"
    ILLUSTRATION = "illustration"  # diagrams, charts, infographics, artwork
    SLIDE = "slide"                # photographed slides, whiteboards, notes
    EVENT = "event"                # parties, concerts, fireworks, conference rooms (no faces found)
    VEHICLE = "vehicle"
    ARCHITECTURE = "architecture"  # buildings, streets, skylines
    PLANT = "plant"                # flowers, trees, gardens
    PRODUCT = "product"            # objects and items for sale
    ACCIDENTAL = "accidental"      # dark / blurry junk (quality-driven)
    UNCERTAIN = "uncertain"
    DUPLICATE = "duplicate"        # exact sha256 match
    NEAR_DUPLICATE = "near_duplicate"  # perceptual-hash match
    UNREADABLE = "unreadable"      # decode failed
    BURST_ALTERNATE = "burst_alternate"
    JUNK_BLINKS = "junk_blinks"    # burst alternates with closed eyes, set by "Reject All Blinks"
    UTILITY = "utility"            # labels, serial numbers, parking pillars, medicine packaging
    CUSTOM = "custom"              # user-defined YAML class
    UNKNOWN = "unknown"

    @property
    def folder(self) -> str:
        """Relative target folder under the organize root (Option 1)."""
        return {
            CategoryType.SCREENSHOT: "Utility_Reference/Screenshots",
            CategoryType.DOCUMENT: "Utility_Reference/Documents",
            CategoryType.GRAPHIC: "Junk_Review/Graphics",
            CategoryType.FOOD: "Non-Human/Food",
            CategoryType.ANIMAL: "Non-Human/Animals",
            CategoryType.SCENERY: "Non-Human/Scenery",
            CategoryType.PEOPLE: "People",
            CategoryType.EVENT: "Events",
            CategoryType.ILLUSTRATION: "Non-Human/Art_and_Diagrams",
            CategoryType.SLIDE: "Utility_Reference/Notes_and_Whiteboards",
            CategoryType.UTILITY: "Utility_Reference/Labels_and_Signs",
            CategoryType.VEHICLE: "Non-Human/Vehicles",
            CategoryType.ARCHITECTURE: "Non-Human/Architecture",
            CategoryType.PLANT: "Non-Human/Plants",
            CategoryType.PRODUCT: "Non-Human/Products",
            CategoryType.ACCIDENTAL: "Junk_Review/Accidental",
            CategoryType.UNCERTAIN: "Uncertain_Review",
            CategoryType.DUPLICATE: "Junk_Review/Duplicates",
            CategoryType.NEAR_DUPLICATE: "Junk_Review/Near_Duplicates",
            CategoryType.UNREADABLE: "Junk_Review/Unreadable",
            CategoryType.BURST_ALTERNATE: "Bursts",
            CategoryType.JUNK_BLINKS: "Junk_Review/Blinks",
            CategoryType.CUSTOM: "Custom",
            CategoryType.UNKNOWN: "Unknown_Date",
        }[self]

    @property
    def is_junk(self) -> bool:
        return self in (
            CategoryType.GRAPHIC, CategoryType.ACCIDENTAL,
            CategoryType.DUPLICATE, CategoryType.NEAR_DUPLICATE,
            CategoryType.UNREADABLE, CategoryType.JUNK_BLINKS,
        )

    @property
    def is_utility(self) -> bool:
        """Reference shots kept in the Utility vault instead of Junk or the photo categories."""
        return self in (CategoryType.UTILITY, CategoryType.SCREENSHOT,
                        CategoryType.DOCUMENT, CategoryType.SLIDE)

    @property
    def is_date_grouped(self) -> bool:
        """Photographic classes no longer use YYYY/MM/DD date subfolders."""
        return False


@dataclass(frozen=True)
class TaxonomyEntry:
    category: CategoryType
    prompts: tuple[str, ...]
    # Optional explicit folder override (e.g. for a custom taxonomy).
    folder_override: str | None = None
    # Class name as stored in the DB; defaults to the category value.
    name: str = ""
    # Custom classes opt in to date subfolders explicitly.
    date_grouped: bool | None = None
    # Look-alike classes share a group; confidence is judged per group.
    group: str = ""

    @property
    def key(self) -> str:
        return self.name or self.category.value

    @property
    def group_key(self) -> str:
        return self.group or self.key


@dataclass(frozen=True)
class Taxonomy:
    """An ordered set of zero-shot classes with their prompts and folders."""

    entries: tuple[TaxonomyEntry, ...]
    source: str = "builtin"
    _by_key: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_key", {e.key: e for e in self.entries})

    @property
    def names(self) -> list[str]:
        return [e.key for e in self.entries]

    @property
    def categories(self) -> list[CategoryType]:
        return [e.category for e in self.entries]

    @property
    def prompts(self) -> list[tuple[str, ...]]:
        return [e.prompts for e in self.entries]

    def entry(self, name: str) -> TaxonomyEntry | None:
        return self._by_key.get(name)

    def category_of(self, name: str) -> CategoryType | None:
        e = self._by_key.get(name)
        if e is not None:
            return e.category
        try:
            return CategoryType(name)
        except ValueError:
            return None

    def folder_override_of(self, name: str) -> str | None:
        e = self._by_key.get(name)
        return e.folder_override if e is not None else None

    def group_of(self, name: str) -> str:
        """Group a class belongs to (its own name when ungrouped or unknown)."""
        e = self._by_key.get(name)
        return e.group_key if e is not None else name

    def group_members(self, group: str) -> list[str]:
        return [e.key for e in self.entries if e.group_key == group]

    def is_date_grouped(self, name: str) -> bool:
        e = self._by_key.get(name)
        if e is not None and e.date_grouped is not None:
            return e.date_grouped
        cat = self.category_of(name)
        return bool(cat and cat.is_date_grouped)


# ---------------------------------------------------------------------------
# Canonical taxonomy (per spec section 4.A).
# ---------------------------------------------------------------------------
def _entry(cat: CategoryType, group: str, *prompts: str) -> TaxonomyEntry:
    return TaxonomyEntry(cat, tuple(prompts), group=group)


TAXONOMY: tuple[TaxonomyEntry, ...] = (
    _entry(
        CategoryType.SCREENSHOT, "screen",
        "a screenshot of a mobile phone screen",
        "a screenshot of a computer desktop or application window",
        "a screenshot of a dark mode app interface with text and buttons",
        "a screenshot of a code editor or terminal window",
        "a screenshot of a chat conversation or the phone settings screen",
        "a screenshot of a web page or an analytics dashboard",
    ),
    _entry(
        CategoryType.DOCUMENT, "screen",
        "a scanned document page with printed text",
        "a photo of a paper receipt or invoice",
        "a photo of a contract, letter or form on a desk",
        "a photo of a passport, ID card or ticket",
        "a page of a book or a printed article",
        "a poster, flyer or printed sign",
    ),
    _entry(
        CategoryType.GRAPHIC, "screen",
        "a meme with text over an image",
        "a logo or icon on a plain background",
        "a cartoon or comic",
        "a flat colored digital graphic or banner",
    ),
    _entry(
        CategoryType.ILLUSTRATION, "screen",
        "a diagram with boxes and arrows",
        "a chart, graph or infographic",
        "a digital illustration or concept art",
        "a drawing, painting or artwork",
        "a hand drawn sketch",
        "a technical flowchart or architecture diagram",
    ),
    _entry(
        CategoryType.SLIDE, "screen",
        "a presentation slide with a title and bullet points",
        "a photo of a whiteboard with handwriting",
        "a photo of handwritten notes in a notebook",
        "a close-up photo of a projector screen showing a slide",
        "a slide deck page with text and a chart",
    ),
    _entry(
        CategoryType.UTILITY, "screen",
        "a close-up photo of a sticker label with a serial number or barcode",
        "a photo of a wifi router or modem with the label showing its password",
        "a photo of a parking garage pillar with a level and bay number painted on it",
        "a photo of a parking space sign, street sign or location marker",
        "a photo of a medicine bottle, pill box or prescription label",
        "a photo of a product rating plate or model number sticker",
    ),
    _entry(
        CategoryType.FOOD, "objects",
        "a close-up photo of food on a plate",
        "a photo of a meal at a restaurant",
        "a photo of a drink, coffee or a dessert",
        "a photo of home cooked food in a kitchen",
        "a photo of fruit, vegetables or groceries",
    ),
    _entry(
        CategoryType.ANIMAL, "nature",
        "a photo of a pet dog",
        "a photo of a pet cat",
        "a photo of a wild animal in nature",
        "a photo of a bird",
        "a photo of a horse, cow or farm animal",
        "a close-up photo of an insect, a butterfly or a fish",
    ),
    _entry(
        CategoryType.SCENERY, "nature",
        "a landscape photo of mountains or hills",
        "a photo of a beach, sea or lake",
        "a photo of a sunset or sky with clouds",
        "a photo of a forest, field or countryside",
        "a photo of a waterfall, river or snowy mountains",
        "a photo of a night sky with stars",
    ),
    _entry(
        CategoryType.PLANT, "nature",
        "a close-up photo of flowers",
        "a photo of a plant or tree leaves",
        "a photo of a garden",
        "a photo of a houseplant in a pot",
    ),
    _entry(
        CategoryType.PEOPLE, "people",
        "a portrait photo of a person",
        "a selfie of a person smiling",
        "a group photo of people posing together",
        "a photo of friends or family at a party or celebration",
        "a photo of a person giving a talk or presentation to an audience",
        "a photo of a child or a baby",
    ),
    _entry(
        CategoryType.EVENT, "people",
        "a birthday cake with candles",
        "fireworks in the night sky",
        "a concert with a stage and bright lights",
        "a conference room or auditorium with rows of seats",
        "party decorations, balloons and streamers",
        "a wedding or festival with decorations",
    ),
    _entry(
        CategoryType.VEHICLE, "objects",
        "a photo of a car on a road",
        "a photo of a motorcycle or bicycle",
        "a photo of a train, bus or airplane",
        "a photo of a boat or ship",
        "a close-up photo of a car interior or dashboard",
    ),
    _entry(
        CategoryType.ARCHITECTURE, "objects",
        "a photo of a building exterior",
        "a photo of a city street with buildings",
        "a photo of a city skyline",
        "a photo of a church, castle or monument",
        "a photo of a building interior or room",
    ),
    _entry(
        CategoryType.PRODUCT, "objects",
        "a product photo of an object on a plain background",
        "a photo of an item for sale such as electronics or clothing",
        "a close-up photo of a gadget or device",
        "a photo of a package or box",
        "a photo of a tool or household object",
    ),
)

DEFAULT_TAXONOMY = Taxonomy(TAXONOMY, source="builtin")

# Shots kept for reference: shown together in the Utility tab, never mixed into Junk.
UTILITY_CATEGORIES: tuple[CategoryType, ...] = tuple(c for c in CategoryType if c.is_utility)

# Prompt lists keyed by category for quick lookup / the CLIP text tower.
TAXONOMY_PROMPTS: dict[CategoryType, tuple[str, ...]] = {
    e.category: e.prompts for e in TAXONOMY
}

# Categories that participate in CLIP zero-shot scoring (junk "accidental" and
# "uncertain"/"duplicate" are assigned by rules, not by the model).
CLIP_CATEGORIES: tuple[CategoryType, ...] = tuple(
    e.category for e in TAXONOMY
)


def flat_prompts() -> list[str]:
    """First prompt per built-in category (kept for callers that want one)."""
    return [TAXONOMY_PROMPTS[c][0] for c in CLIP_CATEGORIES]


def all_prompts(taxonomy: Taxonomy | None = None) -> list[list[str]]:
    """Every prompt per class, aligned to the taxonomy order (for ensembling)."""
    tax = taxonomy or DEFAULT_TAXONOMY
    return [list(e.prompts) for e in tax.entries]


def category_to_folder(cat: CategoryType, folder_override: str | None = None) -> str:
    if folder_override:
        return folder_override
    return cat.folder


# ---------------------------------------------------------------------------
# User-editable taxonomy file (YAML or JSON)
# ---------------------------------------------------------------------------
# Buckets only rules may assign; a model class must never route into them.
_RULE_ONLY = frozenset({
    CategoryType.CUSTOM, CategoryType.UNKNOWN, CategoryType.ACCIDENTAL,
    CategoryType.UNCERTAIN, CategoryType.DUPLICATE, CategoryType.NEAR_DUPLICATE,
    CategoryType.UNREADABLE, CategoryType.BURST_ALTERNATE, CategoryType.JUNK_BLINKS,
})
_RESERVED = re.compile(r'[\\:*?"<>|\x00-\x1f]+')


def _safe_folder(value: str) -> str:
    """Normalize a user folder into safe relative path components."""
    parts: list[str] = []
    for raw in str(value).replace("\\", "/").split("/"):
        part = _RESERVED.sub("_", unicodedata.normalize("NFC", raw)).strip(" .")
        if part in ("", ".", ".."):
            continue
        parts.append(part)
    if not parts:
        raise ValueError(f"taxonomy folder {value!r} has no usable path component")
    return "/".join(parts)


def _read_mapping(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading a YAML taxonomy needs PyYAML (pip install pyyaml); "
            "alternatively provide the taxonomy as a .json file.") from e
    return yaml.safe_load(text)


def parse_taxonomy(data: object, source: str = "inline") -> Taxonomy:
    """Build a :class:`Taxonomy` from parsed YAML/JSON data.

    Accepted shapes: a list of entries, or ``{"classes": [...]}``. Each entry is
    ``{"name": str, "prompts": [str, ...], "folder": str?, "date_grouped": bool?,
    "group": str?}``
    where ``name`` is either a built-in category value (food, scenery, ...) or a
    new custom class that requires ``folder``.
    """
    if isinstance(data, dict):
        data = data.get("classes", data.get("taxonomy"))
    if not isinstance(data, list) or not data:
        raise ValueError("taxonomy must be a non-empty list of classes")
    entries: list[TaxonomyEntry] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"taxonomy entry must be a mapping, got {item!r}")
        name = str(item.get("name") or item.get("category") or "").strip()
        if not name:
            raise ValueError("taxonomy entry is missing 'name'")
        if name in seen:
            raise ValueError(f"duplicate taxonomy class {name!r}")
        seen.add(name)
        prompts = item.get("prompts") or item.get("prompt")
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError(f"class {name!r} needs at least one non-empty prompt")
        folder = item.get("folder")
        try:
            category = CategoryType(name)
            if category in _RULE_ONLY:
                raise ValueError(f"{name!r} is reserved")
            custom = False
        except ValueError:
            if name in {c.value for c in CategoryType}:
                raise
            category, custom = CategoryType.CUSTOM, True
        if custom and not folder:
            raise ValueError(f"custom class {name!r} needs a 'folder'")
        date_grouped = item.get("date_grouped")
        group = str(item.get("group") or "").strip()
        entries.append(TaxonomyEntry(
            category=category,
            prompts=tuple(p.strip() for p in prompts),
            folder_override=_safe_folder(folder) if folder else None,
            name=name if custom else "",
            date_grouped=None if date_grouped is None else bool(date_grouped),
            group=group,
        ))
    return Taxonomy(tuple(entries), source=source)


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load the taxonomy from ``path`` or ``PCUR_TAXONOMY_FILE``; else built-in."""
    chosen = path or os.environ.get("PCUR_TAXONOMY_FILE") or None
    if not chosen:
        return DEFAULT_TAXONOMY
    p = Path(chosen).expanduser()
    return parse_taxonomy(_read_mapping(p), source=str(p))


def get_taxonomy(config=None) -> Taxonomy:
    """Resolve the active taxonomy for ``config`` (file setting, env, built-in)."""
    tax_cfg = getattr(config, "taxonomy", None)
    path = getattr(tax_cfg, "taxonomy_file", None) if tax_cfg is not None else None
    return load_taxonomy(path)
