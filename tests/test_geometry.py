from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from hecras_tools.geometry_operations import GeometryHdf
from hecras_tools.cross_section import CrossSectionData, CrossSectionRecord


@pytest.fixture
def dummy_hdf() -> str:
    return str((Path(__file__).parent / "dummy.hdf").resolve())


def test_geometry_hdf_init(dummy_hdf: str) -> None:
    ops = GeometryHdf(dummy_hdf)
    assert ops.hdf_file == dummy_hdf


def test_cross_section_record_copy(dummy_hdf: str) -> None:
    data = CrossSectionData(dummy_hdf)
    station_key = next(iter(data.available_stations()))
    original = data._records[station_key]
    copied = original.copy()

    for field_def in fields(CrossSectionRecord):
        original_value = getattr(original, field_def.name)
        copied_value = getattr(copied, field_def.name)

        if isinstance(original_value, pd.DataFrame):
            pd.testing.assert_frame_equal(copied_value, original_value)
            assert copied_value is not original_value
        elif isinstance(original_value, np.ndarray):
            assert copied_value is not original_value
            assert np.array_equal(copied_value, original_value)
        elif isinstance(original_value, (LineString, MultiLineString)):
            assert copied_value is not original_value
            assert copied_value.equals(original_value)
        else:
            assert copied_value == original_value

    if copied.geometry_points.size:
        copied.geometry_points[0, 0] = 999
        assert original.geometry_points[0, 0] != 999

    if isinstance(copied.manning_n, np.ndarray) and copied.manning_n.size:
        copied.manning_n[0, 1] = -1
        assert original.manning_n[0, 1] != -1
