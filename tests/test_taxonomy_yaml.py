"""User-editable taxonomy: YAML/JSON loading, validation and routing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from photocure.config import CuratorConfig
from photocure.db.models import ImageRecord, SemanticScore
from photocure.rules.decision_matrix import DecisionMatrix
from photocure.rules.organizer import FolderOrganizer
from photocure.rules.taxonomy import (
    CLIP_CATEGORIES,
    DEFAULT_TAXONOMY,
    CategoryType,
    all_prompts,
    get_taxonomy,
    load_taxonomy,
    parse_taxonomy,
)


def test_default_taxonomy_matches_builtin():
    tax = load_taxonomy(None)
    assert tax is DEFAULT_TAXONOMY
    assert tax.categories == list(CLIP_CATEGORIES)
    assert len(all_prompts()) == len(CLIP_CATEGORIES)
    assert all(len(p) >= 1 for p in all_prompts())


def test_json_taxonomy_with_custom_class(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"classes": [
        {"name": "food", "prompts": ["a meal"]},
        {"name": "kids", "prompts": ["children playing", "a toddler"],
         "folder": "Family/Kids", "date_grouped": True},
    ]}))
    tax = load_taxonomy(f)
    assert tax.names == ["food", "kids"]
    assert tax.category_of("kids") == CategoryType.CUSTOM
    assert tax.folder_override_of("kids") == "Family/Kids"
    assert tax.is_date_grouped("kids") is True
    assert tax.is_date_grouped("food") is True  # built-in default


def test_yaml_taxonomy(tmp_path):
    pytest.importorskip("yaml")
    f = tmp_path / "t.yaml"
    f.write_text(
        "- name: scenery\n"
        "  prompts: [a mountain, a beach]\n"
        "- name: whiteboard\n"
        "  prompts: [a whiteboard with notes]\n"
        "  folder: Work/Whiteboards\n")
    tax = load_taxonomy(f)
    assert tax.names == ["scenery", "whiteboard"]
    assert tax.entry("whiteboard").prompts == ("a whiteboard with notes",)


def test_env_var_selects_file(tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"name": "food", "prompts": ["x"]}]))
    monkeypatch.setenv("PCUR_TAXONOMY_FILE", str(f))
    assert get_taxonomy(CuratorConfig()).names == ["food"]


def test_config_setting_wins_over_builtin(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"name": "animal", "prompts": ["a dog"]}]))
    cfg = CuratorConfig()
    object.__setattr__(cfg.taxonomy, "taxonomy_file", str(f))
    assert get_taxonomy(cfg).names == ["animal"]


@pytest.mark.parametrize("data,msg", [
    ([], "non-empty"),
    ([{"name": "food"}], "prompt"),
    ([{"prompts": ["a"]}], "name"),
    ([{"name": "kids", "prompts": ["a"]}], "folder"),
    ([{"name": "food", "prompts": ["a"]}, {"name": "food", "prompts": ["b"]}], "duplicate"),
    ([{"name": "custom", "prompts": ["a"], "folder": "X"}], "reserved"),
    (["nope"], "mapping"),
])
def test_invalid_taxonomies_are_rejected(data, msg):
    with pytest.raises(ValueError, match=msg):
        parse_taxonomy(data)


def test_custom_folder_is_sanitized():
    tax = parse_taxonomy([{"name": "x", "prompts": ["a"], "folder": "../../etc/Evil:Dir"}])
    folder = tax.folder_override_of("x")
    assert ".." not in folder.split("/")
    assert ":" not in folder
    assert folder == "etc/Evil_Dir"
    with pytest.raises(ValueError):
        parse_taxonomy([{"name": "y", "prompts": ["a"], "folder": "../.."}])


def test_yaml_missing_dependency_message(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "yaml", None)
    f = tmp_path / "t.yaml"
    f.write_text("- name: food\n  prompts: [a]\n")
    with pytest.raises(ImportError, match="PyYAML"):
        load_taxonomy(f)


def _rec(top1: str, score: float = 0.8) -> ImageRecord:
    from datetime import datetime
    return ImageRecord(
        path="/lib/k.jpg", filename="k.jpg",
        taken_at=datetime(2024, 5, 12, 10, 0, 0), taken_at_source="exif:dtoriginal",
        infer_status="inferred",
        semantic=SemanticScore(top1_category=top1, top1_score=score,
                               top2_category="food", top2_score=0.1,
                               score_margin=score - 0.1))


def test_custom_class_routes_to_its_folder(tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([
        {"name": "food", "prompts": ["a meal"]},
        {"name": "kids", "prompts": ["children"], "folder": "Family/Kids",
         "date_grouped": True}]))
    monkeypatch.setenv("PCUR_TAXONOMY_FILE", str(f))
    cfg = CuratorConfig()
    dec = DecisionMatrix(cfg).evaluate_record(_rec("kids"))
    assert dec.category == CategoryType.CUSTOM
    assert dec.folder == "Family/Kids" and dec.needs_date is True
    target = FolderOrganizer(cfg, organize_root=Path("/org")).full_target(_rec("kids"), dec)
    assert target == Path("/org/Family/Kids/2024/05/12/k.jpg")


def test_class_removed_from_taxonomy_routes_to_uncertain(tmp_path, monkeypatch):
    f = tmp_path / "t.json"
    f.write_text(json.dumps([{"name": "food", "prompts": ["a meal"]}]))
    monkeypatch.setenv("PCUR_TAXONOMY_FILE", str(f))
    dec = DecisionMatrix(CuratorConfig()).evaluate_record(_rec("scenery"))
    assert dec.category == CategoryType.UNCERTAIN
