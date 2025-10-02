"""Utilities for working with HEC-RAS structure data."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely import affinity, wkb
from shapely.geometry import LineString

from hecras_tools.cross_section import CrossSectionData
from hecras_tools.utils import safe_literal_eval


STRUCTURE_RENAME_MAP = {
    "Structure ID": "structure_id",
    "River": "river",
    "Reach": "reach",
    "RS": "station",
    "Description": "description",
    "Type": "structure_type",
    "US River": "us_river",
    "US Reach": "us_reach",
    "US RS": "us_station",
    "DS River": "ds_river",
    "DS Reach": "ds_reach",
    "DS RS": "ds_station",
}

TABLE_INFO_RENAME_MAP = {
    "Centerline Profile (Index)": "centerline_profile_index",
    "Centerline Profile (Count)": "centerline_profile_count",
    "US BR Weir Profile (Index)": "us_deck_high_index",
    "US BR Weir Profile (Count)": "us_deck_high_count",
    "US BR Lid Profile (Index)": "us_deck_low_index",
    "US BR Lid Profile (Count)": "us_deck_low_count",
    "DS BR Weir Profile (Index)": "ds_deck_high_index",
    "DS BR Weir Profile (Count)": "ds_deck_high_count",
    "DS BR Lid Profile (Index)": "ds_deck_low_index",
    "DS BR Lid Profile (Count)": "ds_deck_low_count",
}

EMPTY_PROFILE = pd.DataFrame(columns=["Station", "Elevation"])


def _decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte strings in a dataframe into native strings."""

    return df.map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)


def _profile_dataframe(data: list, start: float | int | None, count: float | int | None) -> pd.DataFrame:
    """Return a dataframe slice from profile data using start/count metadata."""

    if pd.isna(start) or pd.isna(count):
        return EMPTY_PROFILE.copy()
    start_idx = int(start)
    count_val = int(count)
    if count_val <= 0:
        return EMPTY_PROFILE.copy()
    rows = data[start_idx:start_idx + count_val]
    if not rows:
        return EMPTY_PROFILE.copy()
    return pd.DataFrame(rows, columns=["Station", "Elevation"])  # type: ignore[arg-type]


def _station_adjustment(profile: pd.DataFrame | np.ndarray | None) -> float:
    """Return the minimum station value from a profile for alignment purposes."""

    if isinstance(profile, pd.DataFrame):
        if "Station" in profile and not profile.empty:
            return float(profile["Station"].min())
        if not profile.empty:
            return float(profile.iloc[:, 0].min())
        return 0.0
    if isinstance(profile, np.ndarray) and profile.size:
        return float(np.nanmin(profile[:, 0]))
    return 0.0


@dataclass(slots=True)
class StructureRecord:
    """Container object for an individual HEC-RAS structure."""

    structure_id: int
    river: str
    reach: str
    station: float | int | str
    structure_type: str
    description: str | None = None
    us_river: str | None = None
    us_reach: str | None = None
    us_station: float | int | str | None = None
    ds_river: str | None = None
    ds_reach: str | None = None
    ds_station: float | int | str | None = None
    geometry: LineString | None = None
    us_xs_geometry: LineString | None = None
    ds_xs_geometry: LineString | None = None
    us_adjust: float | None = None
    ds_adjust: float | None = None
    centerline_adjust: float | None = None
    centerline_profile: pd.DataFrame = field(default_factory=lambda: EMPTY_PROFILE.copy())
    us_deck_high_profile: pd.DataFrame = field(default_factory=lambda: EMPTY_PROFILE.copy())
    us_deck_low_profile: pd.DataFrame = field(default_factory=lambda: EMPTY_PROFILE.copy())
    ds_deck_high_profile: pd.DataFrame = field(default_factory=lambda: EMPTY_PROFILE.copy())
    ds_deck_low_profile: pd.DataFrame = field(default_factory=lambda: EMPTY_PROFILE.copy())
    culvert_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    pier_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    multiple_opening_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    abutment_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> "StructureRecord":
        """Return a deep copy of the record for defensive consumers."""

        copied_fields: dict[str, object] = {}
        for field_def in fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, pd.DataFrame):
                copied_fields[field_def.name] = value.copy(deep=True)
            elif isinstance(value, LineString):
                copied_fields[field_def.name] = wkb.loads(wkb.dumps(value))
            else:
                copied_fields[field_def.name] = value
        return StructureRecord(**copied_fields)  # type: ignore[arg-type]


class StructureData:
    """Load and provide access to structure data from a HEC-RAS geometry HDF file."""

    def __init__(self, hdf_file: str):
        self.hdf_file = hdf_file
        self._dataframe = self._load_dataframe()
        self._records = self._build_record_lookup()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def available_structures(self) -> Iterable[int]:
        """Return identifiers for the structures available in the file."""

        return self._records.keys()

    def full_dataframe(self) -> pd.DataFrame:
        """Return a copy of the consolidated structure dataframe."""

        return self._dataframe.copy(deep=True)

    def centerline_profile(self, structure_id: int) -> pd.DataFrame:
        """Return the centerline station-elevation profile for the structure."""

        return self._get_record(structure_id).centerline_profile.copy(deep=True)

    def deck_profile(self, structure_id: int, side: str, position: str) -> pd.DataFrame:
        """Return deck profiles for the specified structure.

        Parameters
        ----------
        structure_id:
            Identifier for the structure to retrieve.
        side:
            ``"us"`` for upstream profiles, ``"ds"`` for downstream profiles.
        position:
            ``"high"`` for weir profiles and ``"low"`` for lid profiles.
        """

        record = self._get_record(structure_id)
        side = side.lower()
        position = position.lower()
        if side not in {"us", "ds"}:
            raise ValueError("side must be 'us' or 'ds'")
        if position not in {"high", "low"}:
            raise ValueError("position must be 'high' or 'low'")
        attr_name = f"{side}_deck_{position}_profile"
        return getattr(record, attr_name).copy(deep=True)

    def culvert_dataframe(self, structure_id: int) -> pd.DataFrame:
        """Return culvert barrel/group data for the structure."""

        return self._get_record(structure_id).culvert_data.copy(deep=True)

    def pier_dataframe(self, structure_id: int) -> pd.DataFrame:
        """Return pier attribute data for the structure."""

        return self._get_record(structure_id).pier_data.copy(deep=True)

    def multiple_opening_dataframe(self, structure_id: int) -> pd.DataFrame:
        """Return multiple opening attribute data for the structure."""

        return self._get_record(structure_id).multiple_opening_data.copy(deep=True)

    def abutment_dataframe(self, structure_id: int) -> pd.DataFrame:
        """Return abutment attribute data for the structure."""

        return self._get_record(structure_id).abutment_data.copy(deep=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_record(self, key: int) -> StructureRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No structure found for {key!r}") from exc

    def _build_record_lookup(self) -> dict[int, StructureRecord]:
        valid_fields = {f.name for f in fields(StructureRecord)}
        lookup: dict[int, StructureRecord] = {}
        for _, row in self._dataframe.iterrows():
            record_kwargs: dict[str, object] = {}
            for field_name in valid_fields:
                if field_name in row:
                    record_kwargs[field_name] = row[field_name]
            if "structure_id" not in record_kwargs:
                continue
            structure_id = int(record_kwargs["structure_id"])
            lookup[structure_id] = StructureRecord(**record_kwargs)  # type: ignore[arg-type]
        return lookup

    def _load_dataframe(self) -> pd.DataFrame:
        xs_df = self._cross_section_dataframe()
        with h5py.File(self.hdf_file, "r") as hdf:
            try:
                struct_attrs = pd.DataFrame(hdf["Geometry"]["Structures"]["Attributes"][:])
            except KeyError:
                return pd.DataFrame(columns=list(STRUCTURE_RENAME_MAP.values()))
            struct_attrs = _decode_dataframe(struct_attrs)
            for col in ("RS", "US RS", "DS RS"):
                if col in struct_attrs:
                    struct_attrs[col] = struct_attrs[col].apply(safe_literal_eval)
            struct_attrs.rename(columns=STRUCTURE_RENAME_MAP, inplace=True)
            if "structure_id" not in struct_attrs:
                struct_attrs["structure_id"] = struct_attrs.index

            # Merge upstream/downstream cross sections if available
            if not xs_df.empty:
                us_merge = xs_df[["river", "reach", "station", "geometry", "station_adjust"]].copy()
                us_merge.rename(
                    columns={
                        "river": "us_river",
                        "reach": "us_reach",
                        "station": "us_station",
                        "geometry": "us_xs_geometry",
                        "station_adjust": "us_adjust",
                    },
                    inplace=True,
                )
                ds_merge = xs_df[["river", "reach", "station", "geometry", "station_adjust"]].copy()
                ds_merge.rename(
                    columns={
                        "river": "ds_river",
                        "reach": "ds_reach",
                        "station": "ds_station",
                        "geometry": "ds_xs_geometry",
                        "station_adjust": "ds_adjust",
                    },
                    inplace=True,
                )
                if {"us_river", "us_reach", "us_station"}.issubset(struct_attrs.columns):
                    struct_attrs = struct_attrs.merge(us_merge, how="left", on=["us_river", "us_reach", "us_station"])
                else:
                    struct_attrs["us_xs_geometry"] = None
                    struct_attrs["us_adjust"] = None
                if {"ds_river", "ds_reach", "ds_station"}.issubset(struct_attrs.columns):
                    struct_attrs = struct_attrs.merge(ds_merge, how="left", on=["ds_river", "ds_reach", "ds_station"], suffixes=("", "_ds"))
                else:
                    struct_attrs["ds_xs_geometry"] = None
                    struct_attrs["ds_adjust"] = None

            structures_group = hdf["Geometry"]["Structures"]
            centerline_info = structures_group["Centerline Info"][:]
            centerline_counts = [info[1] for info in centerline_info.tolist()]
            centerline_points = pd.DataFrame(structures_group["Centerline Points"][:], columns=["x", "y"])
            point_index = 0
            line_strings: list[LineString] = []
            for count in centerline_counts:
                if count <= 0:
                    line_strings.append(LineString())
                    continue
                pts = centerline_points.iloc[point_index: point_index + count]
                line_strings.append(LineString(pts[["x", "y"]].to_numpy()))
                point_index += count
            struct_attrs["geometry"] = line_strings

            # Profiles (HEC-RAS 6.1+)
            try:
                table_info = pd.DataFrame(structures_group["Table Info"][:])
                table_info = _decode_dataframe(table_info)
                table_info.rename(columns=TABLE_INFO_RENAME_MAP, inplace=True)
                profile_data = structures_group["Profile Data"][:].tolist()
                for column in TABLE_INFO_RENAME_MAP.values():
                    if column not in table_info:
                        table_info[column] = np.nan
                struct_attrs = pd.concat([struct_attrs, table_info], axis=1)
                struct_attrs["centerline_profile"] = struct_attrs.apply(
                    lambda row: _profile_dataframe(profile_data, row.get("centerline_profile_index"), row.get("centerline_profile_count")),
                    axis=1,
                )
                struct_attrs["us_deck_high_profile"] = struct_attrs.apply(
                    lambda row: _profile_dataframe(profile_data, row.get("us_deck_high_index"), row.get("us_deck_high_count")),
                    axis=1,
                )
                struct_attrs["us_deck_low_profile"] = struct_attrs.apply(
                    lambda row: _profile_dataframe(profile_data, row.get("us_deck_low_index"), row.get("us_deck_low_count")),
                    axis=1,
                )
                struct_attrs["ds_deck_high_profile"] = struct_attrs.apply(
                    lambda row: _profile_dataframe(profile_data, row.get("ds_deck_high_index"), row.get("ds_deck_high_count")),
                    axis=1,
                )
                struct_attrs["ds_deck_low_profile"] = struct_attrs.apply(
                    lambda row: _profile_dataframe(profile_data, row.get("ds_deck_low_index"), row.get("ds_deck_low_count")),
                    axis=1,
                )
            except KeyError:
                # Older versions store profiles in a single table
                try:
                    profiles = pd.DataFrame(structures_group["Profiles"][:])
                except KeyError:
                    profiles = pd.DataFrame(columns=["SID", "Type", "Station", "Elevation"])
                profiles = _decode_dataframe(profiles)
                grouped = profiles.groupby("SID")

                def _profile_by_type(sid: str, profile_type: str) -> pd.DataFrame:
                    if sid not in grouped.groups:
                        return EMPTY_PROFILE.copy()
                    group = grouped.get_group(sid)
                    filtered = group[group["Type"] == profile_type][["Station", "Elevation"]]
                    if filtered.empty:
                        return EMPTY_PROFILE.copy()
                    return filtered.reset_index(drop=True)

                struct_attrs["centerline_profile"] = struct_attrs["structure_id"].astype(str).map(
                    lambda sid: _profile_by_type(sid, "Centerline")
                )
                struct_attrs["us_deck_high_profile"] = struct_attrs["structure_id"].astype(str).map(
                    lambda sid: _profile_by_type(sid, "US")
                )
                struct_attrs["ds_deck_high_profile"] = struct_attrs["structure_id"].astype(str).map(
                    lambda sid: _profile_by_type(sid, "DS")
                )
                struct_attrs["us_deck_low_profile"] = [EMPTY_PROFILE.copy() for _ in range(len(struct_attrs))]
                struct_attrs["ds_deck_low_profile"] = [EMPTY_PROFILE.copy() for _ in range(len(struct_attrs))]

            struct_attrs["centerline_adjust"] = struct_attrs["centerline_profile"].apply(_station_adjustment)

            # Culvert groups
            try:
                culvert_attrib = pd.DataFrame(structures_group["Culvert Groups"]["Attributes"][:])
                culvert_attrib = _decode_dataframe(culvert_attrib)
                barrel_attrib = pd.DataFrame(structures_group["Culvert Groups"]["Barrels"]["Attributes"][:])
                barrel_attrib = _decode_dataframe(barrel_attrib)
                try:
                    barrel_info = structures_group["Culvert Groups"]["Barrels"]["Centerline Info"][:]
                    barrel_counts = [info[1] for info in barrel_info.tolist()]
                    barrel_points = pd.DataFrame(structures_group["Culvert Groups"]["Barrels"]["Centerline Points"][:], columns=["x", "y"])
                    point_index = 0
                    barrel_geoms: list[LineString] = []
                    for count in barrel_counts:
                        if count <= 0:
                            barrel_geoms.append(LineString())
                            continue
                        pts = barrel_points.iloc[point_index: point_index + count]
                        barrel_geoms.append(LineString(pts[["x", "y"]].to_numpy()))
                        point_index += count
                except KeyError:
                    barrel_geoms = [LineString() for _ in range(len(barrel_attrib))]
                barrel_attrib["geometry"] = barrel_geoms
                barrel_attrib["merge_id"] = (
                    barrel_attrib["Structure ID"].astype(str) + "_" + barrel_attrib["Culvert Group ID"].astype(str)
                )
                culvert_attrib["merge_id"] = (
                    culvert_attrib["Structure ID"].astype(str) + "_" + culvert_attrib.index.astype(str)
                )
                if "Name" in culvert_attrib:
                    culvert_attrib.rename(columns={"Name": "Group_Name"}, inplace=True)
                struc_attrib = struct_attrs[
                    ["structure_id", "geometry", "us_xs_geometry", "ds_xs_geometry", "us_adjust", "ds_adjust", "centerline_adjust", "structure_type"]
                ].copy()
                struc_attrib.rename(columns={"geometry": "sa2d_geom"}, inplace=True)
                culvert_df = barrel_attrib.merge(culvert_attrib, how="left", on="merge_id")
                culvert_df = culvert_df.merge(struc_attrib, how="left", left_on="Structure ID", right_on="structure_id")

                def _barrel_geometry(row: pd.Series) -> pd.Series:
                    length = row.get("Length", 0.0)
                    line: LineString | None = row.get("sa2d_geom")
                    if line is None or line.is_empty:
                        return pd.Series([row.get("geometry", LineString()), 0.0], index=["geometry", "connection_station"])
                    structure_type = row.get("structure_type", "")
                    if structure_type in {"Lateral", "Connection"}:
                        barrel_geom: LineString = row.get("geometry", LineString())
                        if not barrel_geom.is_empty:
                            station = line.project(line.intersection(barrel_geom))
                            return pd.Series([barrel_geom, station], index=["geometry", "connection_station"])
                        distance = float(row.get("US Station", 0.0)) - float(row.get("centerline_adjust", 0.0))
                        pt_on_line = line.interpolate(distance)
                        coords = list(line.coords)
                        for index in range(1, len(coords)):
                            segment_line = LineString([coords[index - 1], coords[index]])
                            if segment_line.length + LineString(coords[:index]).length >= distance:
                                break
                        else:
                            return pd.Series([LineString(), distance], index=["geometry", "connection_station"])
                        segment = [coords[index - 1], coords[index]]
                        segment_ls = LineString(segment)
                        segment_length_left = LineString([segment[0], pt_on_line.coords[0]]).length
                        segment_length_right = LineString([pt_on_line.coords[0], segment[-1]]).length
                        segment_length = segment_ls.length
                        if segment_length == 0:
                            return pd.Series([LineString(), distance], index=["geometry", "connection_station"])
                        extend_left = length * 0.5 - segment_length_left
                        extend_right = length * 0.5 - segment_length_right
                        unit_dx = (segment[-1][0] - segment[0][0]) / segment_length
                        unit_dy = (segment[-1][1] - segment[0][1]) / segment_length
                        extended_segment = [
                            (segment[0][0] - unit_dx * extend_left, segment[0][1] - unit_dy * extend_left),
                            (segment[-1][0] + unit_dx * extend_right, segment[-1][1] + unit_dy * extend_right),
                        ]
                        extended_line = LineString(extended_segment)
                        return pd.Series([affinity.rotate(extended_line, 90, origin=pt_on_line), distance], index=["geometry", "connection_station"])
                    us_geom: LineString | None = row.get("us_xs_geometry")
                    ds_geom: LineString | None = row.get("ds_xs_geometry")
                    if us_geom is None or ds_geom is None or us_geom.is_empty or ds_geom.is_empty:
                        return pd.Series([LineString(), 0.0], index=["geometry", "connection_station"])
                    us_station = float(row.get("US Station", 0.0)) - float(row.get("us_adjust", 0.0))
                    ds_station = float(row.get("DS Station", 0.0)) - float(row.get("ds_adjust", 0.0))
                    us_point = us_geom.interpolate(us_station)
                    ds_point = ds_geom.interpolate(ds_station)
                    barrel_line = LineString([us_point, ds_point])
                    us_distance = float(row.get("US Distance", 0.0))
                    start_pt = barrel_line.interpolate(us_distance)
                    end_pt = barrel_line.interpolate(us_distance + float(length))
                    barrel_ls = LineString([start_pt, end_pt])
                    station = line.project(barrel_ls.intersection(line))
                    return pd.Series([barrel_ls, station], index=["geometry", "connection_station"])

                culvert_df[["geometry", "connection_station"]] = culvert_df.apply(_barrel_geometry, axis=1)
            except KeyError:
                culvert_df = pd.DataFrame(columns=["Structure ID"])
            culvert_group_dict = {str_id: group.reset_index(drop=True) for str_id, group in culvert_df.groupby("Structure ID")}
            struct_attrs["culvert_data"] = struct_attrs["structure_id"].map(
                lambda sid: culvert_group_dict.get(sid, pd.DataFrame())
            )

            # Pier data
            try:
                pier_df = pd.DataFrame(structures_group["Pier Attributes"][:])
                pier_df = _decode_dataframe(pier_df)
                pier_prof = pd.DataFrame(structures_group["Pier Data"][:], columns=["Pier Width", "Elevation"])

                def _pier_profile(row: pd.Series, profile: str) -> pd.DataFrame:
                    start = int(row[f"{profile} Profile (Index)"])
                    count = int(row[f"{profile} Profile (Count)"])
                    if count <= 0:
                        return pd.DataFrame(columns=["Pier Width", "Elevation"])
                    return pier_prof.iloc[start: start + count].reset_index(drop=True)

                pier_df["pier_us_shape"] = pier_df.apply(lambda row: _pier_profile(row, "US"), axis=1)
                pier_df["pier_ds_shape"] = pier_df.apply(lambda row: _pier_profile(row, "DS"), axis=1)
            except KeyError:
                pier_df = pd.DataFrame(columns=["Structure ID"])
            pier_group_dict = {str_id: group.reset_index(drop=True) for str_id, group in pier_df.groupby("Structure ID")}
            struct_attrs["pier_data"] = struct_attrs["structure_id"].map(
                lambda sid: pier_group_dict.get(sid, pd.DataFrame())
            )

            # Multiple openings
            try:
                mult_open_df = pd.DataFrame(structures_group["Multiple Opening Attributes"][:])
                mult_open_df = _decode_dataframe(mult_open_df)
                struc_attrib = struct_attrs[[
                    "structure_id",
                    "geometry",
                    "us_xs_geometry",
                    "ds_xs_geometry",
                    "us_adjust",
                    "ds_adjust",
                ]].copy()
                struc_attrib.rename(columns={"geometry": "sa2d_geom"}, inplace=True)
                mult_open_df = mult_open_df.merge(
                    struc_attrib,
                    how="left",
                    left_on="Structure ID",
                    right_on="structure_id",
                )

                def _cl_stations(row: pd.Series) -> pd.Series:
                    line: LineString | None = row.get("sa2d_geom")
                    if line is None or line.is_empty:
                        return pd.Series([0.0, 0.0])
                    us_geom: LineString | None = row.get("us_xs_geometry")
                    ds_geom: LineString | None = row.get("ds_xs_geometry")
                    if us_geom is None or ds_geom is None or us_geom.is_empty or ds_geom.is_empty:
                        return pd.Series([0.0, line.length])
                    us_sta_l = float(row.get("US Sta L", 0.0)) - float(row.get("us_adjust", 0.0))
                    us_sta_r = float(row.get("US Sta R", 0.0)) - float(row.get("us_adjust", 0.0))
                    ds_sta_l = float(row.get("DS Sta L", 0.0)) - float(row.get("ds_adjust", 0.0))
                    ds_sta_r = float(row.get("DS Sta R", 0.0)) - float(row.get("ds_adjust", 0.0))
                    us_point_l = us_geom.interpolate(us_sta_l)
                    us_point_r = us_geom.interpolate(us_sta_r)
                    ds_point_l = ds_geom.interpolate(ds_sta_l)
                    ds_point_r = ds_geom.interpolate(ds_sta_r)

                    left_line = LineString([us_point_l, ds_point_l])
                    right_line = LineString([us_point_r, ds_point_r])

                    cl_sta_l = line.project(line.intersection(left_line))
                    cl_sta_r = line.project(line.intersection(right_line))

                    return pd.Series([cl_sta_l, cl_sta_r])

                results = Parallel(n_jobs=-1)(delayed(_cl_stations)(row) for _, row in mult_open_df.iterrows())
                mult_open_df[["CL Sta L", "CL Sta R"]] = pd.DataFrame(results, index=mult_open_df.index)
                mult_open_df["CL Sta L"].fillna(0, inplace=True)
                mult_open_df["CL Sta R"] = mult_open_df.apply(
                    lambda row: row["CL Sta R"] if pd.notna(row["CL Sta R"]) else row.get("sa2d_geom").length if row.get("sa2d_geom") is not None else 0,
                    axis=1,
                )
                mult_open_df = mult_open_df[[
                    "Structure ID",
                    "Opening ID",
                    "CL Sta L",
                    "CL Sta R",
                ] + [col for col in mult_open_df.columns if col not in {"Structure ID", "Opening ID", "CL Sta L", "CL Sta R", "structure_id", "sa2d_geom", "us_xs_geometry", "ds_xs_geometry", "us_adjust", "ds_adjust"}]]
            except KeyError:
                mult_open_df = pd.DataFrame(columns=["Structure ID"])
            mult_open_group_dict = {str_id: group.reset_index(drop=True) for str_id, group in mult_open_df.groupby("Structure ID")}
            struct_attrs["multiple_opening_data"] = struct_attrs["structure_id"].map(
                lambda sid: mult_open_group_dict.get(sid, pd.DataFrame())
            )

            # Abutments
            try:
                abutment_df = pd.DataFrame(structures_group["Abutment Attributes"][:])
                abutment_df = _decode_dataframe(abutment_df)
                abutment_prof = pd.DataFrame(structures_group["Abutment Data"][:], columns=["Station", "Elevation"])

                def _abutment_profile(row: pd.Series, side: str) -> pd.DataFrame:
                    start = int(row[f"{side} Profile (Index)"])
                    count = int(row[f"{side} Profile (Count)"])
                    if count <= 0:
                        return EMPTY_PROFILE.copy()
                    return abutment_prof.iloc[start: start + count].reset_index(drop=True)

                abutment_df["abutment_us_shape"] = abutment_df.apply(lambda row: _abutment_profile(row, "US"), axis=1)
                abutment_df["abutment_ds_shape"] = abutment_df.apply(lambda row: _abutment_profile(row, "DS"), axis=1)
            except KeyError:
                abutment_df = pd.DataFrame(columns=["Structure ID"])
            abutment_group_dict = {str_id: group.reset_index(drop=True) for str_id, group in abutment_df.groupby("Structure ID")}
            struct_attrs["abutment_data"] = struct_attrs["structure_id"].map(
                lambda sid: abutment_group_dict.get(sid, pd.DataFrame())
            )

        return struct_attrs

    def _cross_section_dataframe(self) -> pd.DataFrame:
        try:
            xs_data = CrossSectionData(self.hdf_file).full_dataframe()
        except (KeyError, OSError):
            return pd.DataFrame()
        if xs_data.empty:
            return xs_data
        xs_data = xs_data.copy()
        if "station_elevation" in xs_data:
            xs_data["station_elevation"] = xs_data["station_elevation"].apply(
                lambda values: pd.DataFrame(values, columns=["Station", "Elevation"]) if isinstance(values, np.ndarray) and values.size else (values if isinstance(values, pd.DataFrame) else EMPTY_PROFILE.copy())
            )
            xs_data["station_adjust"] = xs_data["station_elevation"].apply(_station_adjustment)
        else:
            xs_data["station_adjust"] = 0.0
        return xs_data
