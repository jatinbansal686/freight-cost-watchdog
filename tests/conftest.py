import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from watchdog.config import load_config  # noqa: E402
from watchdog.io_utils import load_shipments, load_notes  # noqa: E402
from watchdog.weekly import aggregate_weekly  # noqa: E402
from watchdog.baselines import add_baselines  # noqa: E402
from watchdog.detect import flag_candidates  # noqa: E402
from watchdog.report import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def shipments_df(cfg):
    return load_shipments(cfg.paths.shipments)


@pytest.fixture(scope="session")
def notes_df(cfg):
    return load_notes(cfg.paths.notes)


@pytest.fixture(scope="session")
def weekly_df(shipments_df):
    return aggregate_weekly(shipments_df)


@pytest.fixture(scope="session")
def baselined_df(weekly_df, cfg):
    return add_baselines(weekly_df, cfg.baselines)


@pytest.fixture(scope="session")
def flagged_df(baselined_df, cfg):
    return flag_candidates(baselined_df, cfg.detector, cfg.baselines.own_history_weeks)


@pytest.fixture(scope="session")
def pipeline_result(cfg):
    """Full pipeline in template explain-mode against the committed note-enrichment cache: zero
    LLM calls, fast, deterministic, no API key needed. Verdicts and numeric columns are identical
    to LLM mode by design -- only `reason` wording differs (see test_reproducibility intent)."""
    outputs_dir = REPO_ROOT / "outputs"
    return run_pipeline(cfg, explain_mode="template", use_cache=True, outputs_dir=outputs_dir)


@pytest.fixture(scope="session")
def output_df(pipeline_result):
    return pipeline_result["output_df"]


@pytest.fixture(scope="session")
def weekly_metrics_df(pipeline_result):
    return pipeline_result["weekly_metrics_df"]


def make_shipment_row(shipment_id, origin, destination, route_type, date, qty, dist, cost, material="X", transporter="T"):
    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "route": f"{origin}-{destination}",  # normally added by io_utils.load_shipments
        "route_type": route_type,
        "material": material,
        "quantity_tonnes": qty,
        "distance_km": dist,
        "freight_cost_inr": cost,
        "shipment_date": pd.Timestamp(date),
        "transporter": transporter,
    }
