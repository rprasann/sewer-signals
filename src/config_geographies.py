"""
Geography configuration system.

Decouples all region-specific constants from model logic.  A new geographic
region only needs a YAML file in ``config/geographies/`` — zero code changes.

Usage
-----
    # Load and activate a geography at pipeline startup (main.py):
    from src.config_geographies import load_geography, apply_geography
    geo = load_geography("upper_socal")
    apply_geography(geo)   # overwrites BAY_AREA_FIPS etc. in src.config

    # Then run the pipeline normally:
    uv run main.py --geography upper_socal --skip-cv --no-dash --two-stage

Geography YAML schema (config/geographies/<name>.yaml)
-------------------------------------------------------
name            : Human-readable label (e.g. "Upper Southern California")
shortname       : Machine-readable key used by --geography flag
counties        : dict  county_name → 5-digit FIPS string
populations     : dict  FIPS → 2020 Census population (int)
validation_counties : list of FIPS for the --counties validation shorthand
exclude_fips    : list of FIPS to exclude from training (sparse WW data)
data_start_date : ISO date
data_end_date   : ISO date
train_end_date  : First CV cutoff — must be ≥ INPUT_SIZE+H weeks into the series
val_end_date    : Last CV cutoff
map_center_lat  : Dashboard map latitude centre
map_center_lon  : Dashboard map longitude centre
map_zoom        : Dashboard map default zoom level
centroids       : dict  FIPS → [lat, lon]
outbreak_validation_windows : list of dicts (name, eval_start, eval_end, counties)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_GEO_DIR = Path(__file__).resolve().parents[1] / "config" / "geographies"


# ---------------------------------------------------------------------------
# GeographyConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeographyConfig:
    """All region-specific constants for one geographic area."""

    name:                       str
    shortname:                  str
    county_fips:                dict[str, str]           # county_name → FIPS
    fips_to_county:             dict[str, str]           # FIPS → county_name
    populations:                dict[str, int]           # FIPS → population
    validation_counties:        list[str]                # FIPS
    exclude_fips:               list[str]                # FIPS
    data_start_date:            str
    data_end_date:              str
    train_end_date:             str
    val_end_date:               str
    map_center_lat:             float = 37.7
    map_center_lon:             float = -122.2
    map_zoom:                   float = 7.8
    centroids:                  dict[str, tuple[float, float]] = field(default_factory=dict)
    outbreak_validation_windows: list[dict] = field(default_factory=list)

    # Derived convenience properties
    @property
    def county_names(self) -> list[str]:
        return list(self.county_fips.keys())

    @property
    def fips_list(self) -> list[str]:
        return list(self.county_fips.values())

    @property
    def fips_int_map(self) -> dict[str, int]:
        """Stable integer encoding of FIPS codes (sorted order).

        Used by tft_model._build_static_df() for the county_fips_encoded
        static covariate.  Called at model-fit time (after apply_geography()),
        NOT at module import time, so it always reflects the active geography.
        """
        return {
            fips: idx
            for idx, fips in enumerate(sorted(self.fips_list))
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_geography(name: str) -> GeographyConfig:
    """Load a geography configuration from ``config/geographies/<name>.yaml``.

    Parameters
    ----------
    name : Shortname of the geography (e.g. "bay_area", "upper_socal").
           Matches the YAML filename (without .yaml) and the ``shortname`` key.

    Raises
    ------
    FileNotFoundError  if the YAML file does not exist.
    KeyError           if required fields are missing from the YAML.
    """
    yaml_path = _GEO_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        available = [p.stem for p in _GEO_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Geography '{name}' not found at {yaml_path}.\n"
            f"Available geographies: {sorted(available)}\n"
            f"To add a new region, create config/geographies/{name}.yaml "
            "(see bay_area.yaml for the schema)."
        )

    with yaml_path.open() as fh:
        raw = yaml.safe_load(fh)

    county_fips = {str(k): str(v) for k, v in raw["counties"].items()}
    fips_to_county = {v: k for k, v in county_fips.items()}

    centroids: dict[str, tuple[float, float]] = {}
    for fips, coords in (raw.get("centroids") or {}).items():
        centroids[str(fips)] = (float(coords[0]), float(coords[1]))

    windows: list[dict] = []
    for w in (raw.get("outbreak_validation_windows") or []):
        windows.append({
            "name":       w["name"],
            "eval_start": w["eval_start"],
            "eval_end":   w["eval_end"],
            "counties":   w.get("counties"),   # None = all counties
        })

    return GeographyConfig(
        name=raw["name"],
        shortname=raw["shortname"],
        county_fips=county_fips,
        fips_to_county=fips_to_county,
        populations={str(k): int(v) for k, v in raw["populations"].items()},
        validation_counties=[str(f) for f in raw.get("validation_counties", [])],
        exclude_fips=[str(f) for f in raw.get("exclude_fips", [])],
        data_start_date=str(raw["data_start_date"]),
        data_end_date=str(raw["data_end_date"]),
        train_end_date=str(raw["train_end_date"]),
        val_end_date=str(raw["val_end_date"]),
        map_center_lat=float(raw.get("map_center_lat", 37.7)),
        map_center_lon=float(raw.get("map_center_lon", -122.2)),
        map_zoom=float(raw.get("map_zoom", 7.8)),
        centroids=centroids,
        outbreak_validation_windows=windows,
    )


def list_geographies() -> list[str]:
    """Return the shortnames of all available geography configs."""
    return sorted(p.stem for p in _GEO_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# apply_geography  — overwrites src.config module-level geography variables
# ---------------------------------------------------------------------------

def apply_geography(geo: GeographyConfig) -> None:
    """Overwrite all geography-specific constants in ``src.config`` in-place.

    Must be called at pipeline startup (before any module reads these values
    from config).  main.py calls this immediately after argument parsing
    when ``--geography`` is specified.

    After calling this, the following are all consistent with the new geography:
      cfg.BAY_AREA_FIPS, cfg.FIPS_TO_COUNTY, cfg.BAY_AREA_POPULATION,
      cfg.BAY_AREA_COUNTIES, cfg.THREE_COUNTY_FIPS, cfg.EXCLUDE_FIPS,
      cfg.DATA_START_DATE, cfg.DATA_END_DATE, cfg.TRAIN_END_DATE,
      cfg.VAL_END_DATE, cfg.OUTBREAK_VALIDATION_WINDOWS,
      cfg.ACTIVE_GEOGRAPHY  (the full GeographyConfig object)

    Note: ``_FIPS_INT`` in ``tft_model.py`` is a module-level constant computed
    at import time.  It is bypassed: ``_build_static_df`` now calls
    ``geo.fips_int_map`` directly so it always reflects the active geography.
    """
    import src.config as cfg  # local import to avoid circular dependency

    # Mutate dicts and lists IN-PLACE so every locally-bound ``from src.config import X``
    # reference (in processor.py, dashboard.py, attention_plots.py, etc.) automatically
    # sees the new geography without requiring those modules to be re-imported or patched.
    # Strings are immutable and must be rebound; callers that need dynamic string values
    # must reference cfg.TRAIN_END_DATE / cfg.VAL_END_DATE (not locally-bound imports).

    # ── Dicts: in-place mutation ────────────────────────────────────────────────
    cfg.BAY_AREA_FIPS.clear()
    cfg.BAY_AREA_FIPS.update(geo.county_fips)

    cfg.FIPS_TO_COUNTY.clear()
    cfg.FIPS_TO_COUNTY.update(geo.fips_to_county)

    cfg.BAY_AREA_POPULATION.clear()
    cfg.BAY_AREA_POPULATION.update(geo.populations)

    # ── Lists: in-place mutation ────────────────────────────────────────────────
    cfg.BAY_AREA_COUNTIES.clear()
    cfg.BAY_AREA_COUNTIES.extend(geo.county_names)

    cfg.THREE_COUNTY_FIPS.clear()
    cfg.THREE_COUNTY_FIPS.extend(geo.validation_counties)

    cfg.EXCLUDE_FIPS.clear()
    cfg.EXCLUDE_FIPS.extend(geo.exclude_fips)

    cfg.OUTBREAK_VALIDATION_WINDOWS.clear()
    cfg.OUTBREAK_VALIDATION_WINDOWS.extend(geo.outbreak_validation_windows or [])

    # ── Strings: must rebind (immutable) ───────────────────────────────────────
    cfg.DATA_START_DATE = geo.data_start_date
    cfg.DATA_END_DATE   = geo.data_end_date
    cfg.TRAIN_END_DATE  = geo.train_end_date
    cfg.VAL_END_DATE    = geo.val_end_date

    # ── Object references ───────────────────────────────────────────────────────
    cfg.ACTIVE_GEOGRAPHY  = geo
    cfg.PRIORITY_COUNTIES = geo.validation_counties

    from loguru import logger
    logger.info(
        "Geography loaded: {}  ({} counties, train_end={}, val_end={})",
        geo.name,
        len(geo.county_fips),
        geo.train_end_date,
        geo.val_end_date,
    )
