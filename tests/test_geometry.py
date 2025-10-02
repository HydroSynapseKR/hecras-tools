from __future__ import annotations

from dataclasses import fields

import h5py
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString

from hecras_tools.geometry_operations import GeometryHdf
from hecras_tools.cross_section import CrossSectionData, CrossSectionRecord


@pytest.fixture
def dummy_hdf(tmp_path) -> str:
    hdf_path = tmp_path / "dummy.hdf"
    with h5py.File(hdf_path, "w") as hdf:
        hdf.attrs["Projection"] = np.bytes_("EPSG:4326")

        geometry = hdf.create_group("Geometry")
        cross_sections = geometry.create_group("Cross Sections")

        str_type = h5py.string_dtype("utf-8")
        attrs_dtype = np.dtype([
            ("River", str_type),
            ("Reach", str_type),
            ("RS", str_type),
            ("Description", str_type),
            ("Len Left", "<f8"),
            ("Len Channel", "<f8"),
            ("Len Right", "<f8"),
            ("Left Bank", "<f8"),
            ("Right Bank", "<f8"),
            ("Friction Mode", str_type),
            ("Contr", "<f8"),
            ("Expan", "<f8"),
            ("HP Count", "<i4"),
            ("HP Start Elev", "<f8"),
            ("HP Vert Incr", "<f8"),
            ("HP LOB Slices", "<i4"),
            ("HP Chan Slices", "<i4"),
            ("HP ROB Slices", "<i4"),
            ("Default Centerline", "<i4"),
            ("Skew", "<f8"),
            ("PC Invert", "<f8"),
            ("PC Width", "<f8"),
            ("PC Mann", "<f8"),
            ("Deck Preissman Slot", "<i4"),
            ("Contr (USF)", "<f8"),
            ("Expan (USF)", "<f8"),
        ])

        attrs_data = np.array([
            (
                "Test River",
                "Reach A",
                "100.0",
                "XS 1",
                120.0,
                130.0,
                140.0,
                10.0,
                12.0,
                "Normal",
                0.3,
                0.4,
                2,
                5.0,
                0.5,
                1,
                2,
                3,
                1,
                15.0,
                50.0,
                60.0,
                0.02,
                0,
                0.1,
                0.2,
            )
        ], dtype=attrs_dtype)

        cross_sections.create_dataset("Attributes", data=attrs_data)

        cross_sections.create_dataset(
            "Polyline Info", data=np.array([[0, 2, 2, 1]], dtype=np.int32)
        )
        cross_sections.create_dataset(
            "Polyline Points", data=np.array([[0, 0], [10, 5], [0, 2]], dtype=np.int32)
        )
        cross_sections.create_dataset(
            "Manning's n Info", data=np.array([[0, 2]], dtype=np.int32)
        )
        cross_sections.create_dataset(
            "Manning's n Values",
            data=np.array([[0.0, 0.025], [10.0, 0.03]], dtype=np.float64),
        )
        cross_sections.create_dataset(
            "Station Elevation Info", data=np.array([[0, 2]], dtype=np.int32)
        )
        cross_sections.create_dataset(
            "Station Elevation Values",
            data=np.array([[0.0, 100.0], [10.0, 101.0]], dtype=np.float64),
        )

    return str(hdf_path)


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
