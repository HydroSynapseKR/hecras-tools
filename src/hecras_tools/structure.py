"""Utilities for working with HEC-RAS structure data."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable, Any

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely import affinity, wkb
from shapely.geometry import LineString, MultiLineString

from hecras_tools.cross_section import CrossSectionData
from hecras_tools.utils import safe_literal_eval, STRUCTURE_RENAME_MAP, TABLE_INFO_RENAME_MAP


EMPTY_PROFILE = pd.DataFrame(columns=["Station", "Elevation"])
EMPTY_MANNING = pd.DataFrame(columns=["Station", "Value"])


def _empty_profile_array() -> np.ndarray:
    return np.zeros((0, 2))


def _empty_manning_array() -> np.ndarray:
    return np.zeros((0, 2))


def _decode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte strings in a dataframe into native strings."""

    return df.map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)


@dataclass(slots=True)
class StructureRecord:
    """Container object for an individual HEC-RAS structure."""

    structure_key: int | str
    structure_type: str
    river: str
    reach: str
    station: float | int | str
    structure_mode: str | None = None
    description: str | None = None
    connection: str | None = None
    last_edited: str | None = None
    upstream_distance: float | int | None = None
    weir_width: float | int | None = None
    weir_max_submergence: float | int | None = None
    weir_min_elevation: float | int | None = None
    weir_coefficient: float | int | None = None
    weir_shape: str | None = None
    weir_design_eg_head: float | int | None = None
    weir_design_spillway_ht: float | int | None = None
    weir_us_slope: float | int | None = None
    weir_ds_slope: float | int | None = None
    linear_routing_positive_coef: float | int | None = None
    linear_routing_negative_coef: float | int | None = None
    linear_routing_elevation: float | int | None = None
    lw_hw_position: float | int | None = None
    lw_tw_position: float | int | None = None
    lw_hw_distance: float | int | None = None
    lw_tw_distance: float | int | None = None
    lw_span_multiple: float | int | None = None
    use_2d_for_overflow: int | None = None
    use_velocity_into_2d: int | None = None
    hagers_weir_coefficient: float | int | None = None
    hagers_height: float | int | None = None
    hagers_slope: float | int | None = None
    hagers_angle: float | int | None = None
    hagers_radius: float | int | None = None
    use_ws_for_weir_reference: int | None = None
    pilot_flow: float | int | None = None
    culvert_groups: int | None = None
    culverts_flap_gates: int | None = None
    gate_groups: int | None = None
    htab_ff_points: int | None = None
    htab_rc_count: int | None = None
    htab_rc_points: int | None = None
    htab_hw_max: float | int | None = None
    htab_tw_max: float | int | None = None
    htab_max_flow: float | int | None = None
    cell_spacing_near: float | int | None = None
    cell_spacing_far: float | int | None = None
    near_repeats: int | None = None
    protection_radius: float | int | None = None
    use_friction_in_momentum: int | None = None
    use_weight_in_momentum: int | None = None
    use_critical_us: int | None = None
    use_eg_for_pressure_criteria: int | None = None
    ice_option: int | None = None
    weir_skew: float | int | None = None
    pier_skew: float | int | None = None
    bridge_us_left_bank: float | int | None = None
    bridge_us_right_bank: float | int | None = None
    bridge_ds_left_bank: float | int | None = None
    bridge_ds_right_bank: float | int | None = None
    xs_us_left_bank: float | int | None = None
    xs_us_right_bank: float | int | None = None
    xs_ds_left_bank: float | int | None = None
    xs_ds_right_bank: float | int | None = None
    us_ineff_left_station: float | int | None = None
    us_ineff_left_elevation: float | int | None = None
    us_ineff_right_station: float | int | None = None
    us_ineff_right_elevation: float | int | None = None
    ds_ineff_left_station: float | int | None = None
    ds_ineff_left_elevation: float | int | None = None
    ds_ineff_right_station: float | int | None = None
    ds_ineff_right_elevation: float | int | None = None
    use_override_hw_connectivity: int | None = None
    use_override_tw_connectivity: int | None = None
    use_override_htab_ib_curves: int | None = None
    snn_id: int | None = None
    default_centerline: int | None = None
    us_river: str | None = None
    us_reach: str | None = None
    us_station: float | int | str | None = None
    ds_river: str | None = None
    ds_reach: str | None = None
    ds_station: float | int | str | None = None
    us_sa2d: str | None = None
    ds_sa2d: str | None = None
    us_type: str | None = None
    ds_type: str | None = None
    node_name: str | None = None
    structure_id: int | None = None
    geometry: LineString | None = None
    us_xs_geometry: LineString | None = None
    ds_xs_geometry: LineString | None = None
    us_adjust: float | None = None
    ds_adjust: float | None = None
    centerline_adjust: float | None = None
    geometry_points: Any | None = None
    geometry_parts: Any | None = None
    profile_info: list[int] | None = None
    centerline_profile: np.ndarray = field(default_factory=_empty_profile_array)
    us_cross_section_profile: np.ndarray = field(default_factory=_empty_profile_array)
    us_bridge_profile: np.ndarray = field(default_factory=_empty_profile_array)
    us_deck_high_profile: np.ndarray = field(default_factory=_empty_profile_array)
    us_deck_low_profile: np.ndarray = field(default_factory=_empty_profile_array)
    ds_cross_section_profile: np.ndarray = field(default_factory=_empty_profile_array)
    ds_bridge_profile: np.ndarray = field(default_factory=_empty_profile_array)
    ds_deck_high_profile: np.ndarray = field(default_factory=_empty_profile_array)
    ds_deck_low_profile: np.ndarray = field(default_factory=_empty_profile_array)
    us_cross_section_mannings: np.ndarray = field(default_factory=_empty_manning_array)
    us_bridge_mannings: np.ndarray = field(default_factory=_empty_manning_array)
    ds_cross_section_mannings: np.ndarray = field(default_factory=_empty_manning_array)
    ds_bridge_mannings: np.ndarray = field(default_factory=_empty_manning_array)
    culvert_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    pier_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    multiple_opening_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    abutment_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    bridge_coefficient: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def copy(self) -> "StructureRecord":
        """Return a deep copy of the record for defensive consumers."""

        copied_fields: dict[str, object] = {}
        for field_def in fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, pd.DataFrame):
                copied_fields[field_def.name] = value.copy(deep=True)
            elif isinstance(value, (LineString, MultiLineString)):
                copied_fields[field_def.name] = wkb.loads(wkb.dumps(value))
            elif isinstance(value, np.ndarray):
                copied_fields[field_def.name] = value.copy()
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
    def available_structures(self) -> Iterable[str]:
        """Return identifiers for the structures available in the file."""

        return self._records.keys()

    def full_dataframe(self) -> pd.DataFrame:
        """Return a copy of the consolidated structure dataframe."""

        return self._dataframe.copy(deep=True)

    def centerline_profile(self, structure_key: str) -> pd.DataFrame:
        """Return the centerline station-elevation profile for the structure."""

        profile = self._get_record(structure_key).centerline_profile
        if isinstance(profile, np.ndarray) and profile.size:
            return pd.DataFrame(profile, columns=["Station", "Elevation"])
        return EMPTY_PROFILE.copy()

    def deck_profile(self, structure_key: str, side: str, position: str) -> np.ndarray:
        """Return deck profiles for the specified structure.

        Parameters
        ----------
        structure_key:
            Identifier for the structure to retrieve.
        side:
            ``"us"`` for upstream profiles, ``"ds"`` for downstream profiles.
        position:
            ``"high"`` for weir deck profiles and ``"low"`` for low deck (lid) profiles.
        """

        record = self._get_record(structure_key)
        side = side.lower()
        position = position.lower()
        if side not in {"us", "ds"}:
            raise ValueError("side must be 'us' or 'ds'")
        if position not in {"high", "low"}:
            raise ValueError("position must be 'high' or 'low'")
        attr_map = {
            ("us", "high"): "us_deck_high_profile",
            ("us", "low"): "us_deck_low_profile",
            ("ds", "high"): "ds_deck_high_profile",
            ("ds", "low"): "ds_deck_low_profile",
        }
        attr_name = attr_map[(side, position)]
        profile = getattr(record, attr_name)
        if isinstance(profile, np.ndarray):
            return profile.copy()
        return np.asarray(profile)

    def culvert_dataframe(self, structure_key: str) -> pd.DataFrame:
        """Return culvert barrel/group data for the structure."""

        return self._get_record(structure_key).culvert_data.copy(deep=True)

    def pier_dataframe(self, structure_key: str) -> pd.DataFrame:
        """Return pier attribute data for the structure."""

        return self._get_record(structure_key).pier_data.copy(deep=True)

    def multiple_opening_dataframe(self, structure_key: str) -> pd.DataFrame:
        """Return multiple opening attribute data for the structure."""

        return self._get_record(structure_key).multiple_opening_data.copy(deep=True)

    def abutment_dataframe(self, structure_key: str) -> pd.DataFrame:
        """Return abutment attribute data for the structure."""

        return self._get_record(structure_key).abutment_data.copy(deep=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_record(self, key: str) -> StructureRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No structure found for {key!r}") from exc

    def _build_record_lookup(self) -> dict[str, StructureRecord]:
        valid_fields = {f.name for f in fields(StructureRecord)}
        lookup: dict[str, StructureRecord] = {}
        for _, row in self._dataframe.iterrows():
            record_kwargs: dict[str, object] = {}
            for field_name in valid_fields:
                if field_name in row:
                    record_kwargs[field_name] = row[field_name]
            if "structure_key" not in record_kwargs:
                continue
            lookup_key = str(record_kwargs["structure_key"])  # type: ignore[arg-type]
            lookup[lookup_key] = StructureRecord(**record_kwargs)  # type: ignore[arg-type]
        return lookup

    def _load_dataframe(self) -> pd.DataFrame:
        xs_df = CrossSectionData(self.hdf_file).full_dataframe()
        xs_df['station_adjust'] = [arr[:, 0].min() for arr in xs_df['station_elevation'].values]
        with h5py.File(self.hdf_file, "r") as hdf:
            structures_group = hdf["Geometry"]["Structures"]
            try:
                struct_attrs = pd.DataFrame(structures_group["Attributes"][:])
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
            else:
                struct_attrs['us_xs_geometry'] = None
                struct_attrs['us_adjust'] = None
                struct_attrs['ds_xs_geometry'] = None
                struct_attrs['ds_adjust'] = None

            # ------------------------------------------------------------------
            # Geometry points
            # ------------------------------------------------------------------
            geometry_info = structures_group["Centerline Info"][:]
            points = structures_group["Centerline Points"][:]
            parts = structures_group["Centerline Parts"][:]
            struct_attrs['geometry_points'] = struct_attrs.index.map(
                lambda i: points[geometry_info[i, 0]: geometry_info[i, 0] + geometry_info[i, 1]]
            )
            struct_attrs['geometry_parts'] = struct_attrs.index.map(
                lambda i: parts[geometry_info[i, 2]: geometry_info[i, 2] + geometry_info[i, 3]]
            )

            def build_cl_geom(row):
                geoms = [LineString(row['geometry_points'][i[0]:i[0] + i[1]]) for i in row['geometry_parts']]
                if len(geoms) == 1:
                    return geoms[0]
                return MultiLineString(geoms)

            struct_attrs["geometry"] = struct_attrs.apply(build_cl_geom, axis=1)

            # Profiles (HEC-RAS 6.1+)
            try:
                profile_info = structures_group["Table Info"]
                # converting structured array to normal 2d array
                profile_info = profile_info[:].view(np.int32).reshape(len(profile_info[:]), -1)
                struct_attrs['profile_info'] = profile_info.tolist()
                profile_data = structures_group['Profile Data'][:]
                mann_data = structures_group['Mannings Data'][:]
                struct_attrs['centerline_profile'] = [profile_data[start:start + count]
                                                      for start, count in profile_info[:, :2]]
                struct_attrs['us_xs_profile'] = [profile_data[start:start + count]
                                                 for start, count in profile_info[:, 2:4]]
                struct_attrs['us_br_profile'] = [profile_data[start:start + count]
                                                 for start, count in profile_info[:, 4:6]]
                struct_attrs['us_br_weir_profile'] = [profile_data[start:start + count]
                                                      for start, count in profile_info[:, 6:8]]
                struct_attrs['us_br_lid_profile'] = [profile_data[start:start + count]
                                                     for start, count in profile_info[:, 8:10]]
                struct_attrs['ds_xs_profile'] = [profile_data[start:start + count]
                                                 for start, count in profile_info[:, 10:12]]
                struct_attrs['ds_br_profile'] = [profile_data[start:start + count]
                                                 for start, count in profile_info[:, 12:14]]
                struct_attrs['ds_br_weir_profile'] = [profile_data[start:start + count]
                                                      for start, count in profile_info[:, 14:16]]
                struct_attrs['ds_br_lid_profile'] = [profile_data[start:start + count]
                                                     for start, count in profile_info[:, 16:18]]
                struct_attrs['us_xs_mann'] = [mann_data[start:start + count]
                                              for start, count in profile_info[:, 18:20]]
                struct_attrs['us_br_mann'] = [mann_data[start:start + count]
                                              for start, count in profile_info[:, 20:22]]
                struct_attrs['ds_xs_mann'] = [mann_data[start:start + count]
                                              for start, count in profile_info[:, 22:24]]
                struct_attrs['ds_br_mann'] = [mann_data[start:start + count]
                                              for start, count in profile_info[:, 24:26]]

                struct_attrs.rename(
                    columns={
                        "us_br_weir_profile": "us_deck_high_profile",
                        "us_br_lid_profile": "us_deck_low_profile",
                        "ds_br_weir_profile": "ds_deck_high_profile",
                        "ds_br_lid_profile": "ds_deck_low_profile",
                        "us_br_profile": "us_bridge_profile",
                        "ds_br_profile": "ds_bridge_profile",
                        "us_xs_profile": "us_cross_section_profile",
                        "ds_xs_profile": "ds_cross_section_profile",
                        "us_xs_mann": "us_cross_section_mannings",
                        "us_br_mann": "us_bridge_mannings",
                        "ds_xs_mann": "ds_cross_section_mannings",
                        "ds_br_mann": "ds_bridge_mannings",
                    },
                    inplace=True,
                )
            except KeyError:
                # Older versions store profiles in a single table
                raise NotImplementedError('Only HEC-RAS version 6.1+ is supported')

            def _centerline_min_station(array: Any) -> float | None:
                if isinstance(array, np.ndarray) and array.size:
                    return float(array[:, 0].min())
                return None

            struct_attrs["centerline_adjust"] = struct_attrs['centerline_profile'].map(_centerline_min_station)

            # Culvert groups and barrels
            try:
                culvert_grp_attrib = pd.DataFrame(structures_group["Culvert Groups"]["Attributes"][:])
                culvert_grp_attrib = _decode_dataframe(culvert_grp_attrib)
                barrel_attrib = pd.DataFrame(structures_group["Culvert Groups"]["Barrels"]["Attributes"][:])
                barrel_attrib = _decode_dataframe(barrel_attrib)
                # Extracting the barrel geometry if it exists
                try:
                    barrel_geom_info = structures_group["Culvert Groups"]["Barrels"]["Centerline Info"][:]
                    barrel_geom_points = structures_group["Culvert Groups"]["Barrels"]["Centerline Points"][:]

                    # Barrel centerline parts are not implemented as it is unlikely that barrel centerlines will be a
                    # multipart geometry
                    # barrel_geom_parts = structures_group["Culvert Groups"]["Barrels"]["Centerline Parts"][:]
                    # barrel_attrib['geometry_points'] = struct_attrs.index.map(
                    #     lambda i: points[geometry_info[i, 0]: geometry_info[i, 0] + geometry_info[i, 1]]
                    # )
                    # barrel_attrib['geometry_parts'] = struct_attrs.index.map(
                    #     lambda i: parts[geometry_info[i, 2]: geometry_info[i, 2] + geometry_info[i, 3]]
                    # )

                    barrel_geoms = [LineString(barrel_geom_points[start: start + count])
                                    for start, count, _, _ in barrel_geom_info]
                except KeyError:
                    barrel_geoms = [LineString() for _ in range(len(barrel_attrib))]
                barrel_attrib["geometry"] = barrel_geoms
                barrel_attrib["merge_id"] = (
                    barrel_attrib["Structure ID"].astype(str) + "_" + barrel_attrib["Culvert Group ID"].astype(str)
                )
                barrel_attrib.drop(columns=['Structure ID', 'Culvert Group ID'], inplace=True)
                culvert_grp_attrib["merge_id"] = (
                    culvert_grp_attrib["Structure ID"].astype(str) + "_" + culvert_grp_attrib.index.astype(str)
                )
                if "Name" in culvert_grp_attrib:
                    culvert_grp_attrib.rename(columns={"Name": "Group_Name"}, inplace=True)
                struc_attrib = struct_attrs[
                    ["structure_id", "geometry", "us_xs_geometry", "ds_xs_geometry", "us_adjust", "ds_adjust",
                     "centerline_adjust", "structure_type"]
                ].copy()
                struc_attrib.rename(columns={"geometry": "sa2d_geom", 'structure_id': 'Structure ID'}, inplace=True)
                culvert_df = barrel_attrib.merge(culvert_grp_attrib, how="left", on="merge_id")
                culvert_df = culvert_df.merge(struc_attrib, how="left", on="Structure ID")

                # For those barrels using a default geometry, calculate actual geometry
                def _barrel_geometry(row: pd.Series) -> pd.Series:
                    length = row.get("Length", 0.0)
                    line: LineString | None = row.get("sa2d_geom")
                    structure_type = row.get("structure_type", "")
                    barrel_geom: LineString = row.get("geometry", LineString())
                    # For cases where barrel geometry already exists
                    if not barrel_geom.is_empty:
                        station = line.project(line.intersection(barrel_geom))
                        return pd.Series([barrel_geom, station], index=["geometry", "connection_station"])
                    # For structures without US Distance
                    if structure_type in {"Lateral", "Connection"}:
                        distance = float(row.get("US Station", 0.0)) - float(row.get("centerline_adjust", 0.0))
                        pt_on_line = line.interpolate(distance)
                        try:
                            segment = next([line.coords[i - 1], line.coords[i]] for i in range(1, len(line.coords))
                                           if LineString(line.coords[:i + 1]).length >= distance)
                        except StopIteration:
                            # barrel station is beyond connection length
                            segment = line.coords[-2:]
                        segment_length_left = LineString([segment[0], pt_on_line.coords[0]]).length
                        segment_length_right = LineString([pt_on_line.coords[0], segment[-1]]).length
                        segment_length = LineString(segment).length
                        extend_left = length * 0.5 - segment_length_left
                        extend_right = length * 0.5 - segment_length_right
                        unit_dx = (segment[-1][0] - segment[0][0]) / segment_length
                        unit_dy = (segment[-1][1] - segment[0][1]) / segment_length
                        extended_segment = [
                            (segment[0][0] - unit_dx * extend_left, segment[0][1] - unit_dy * extend_left),
                            (segment[-1][0] + unit_dx * extend_right, segment[-1][1] + unit_dy * extend_right),
                        ]
                        extended_line = LineString(extended_segment)
                        return pd.Series([affinity.rotate(extended_line, 90, origin=pt_on_line), distance],
                                         index=["geometry", "connection_station"])

                    # for cases that uses 1D bounding XS (bridges/culverts/inline)
                    us_geom = row.get("us_xs_geometry")
                    ds_geom = row.get("ds_xs_geometry")
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
            culvert_group_dict = {
                str_id: group.reset_index(drop=True) for str_id, group in culvert_df.groupby("Structure ID")
            }
            struct_attrs["culvert_data"] = struct_attrs["structure_id"].map(
                lambda sid: culvert_group_dict.get(sid, pd.DataFrame())
            )

            # Pier data
            try:
                pier_df = pd.DataFrame(structures_group["Pier Attributes"][:])
                pier_df = _decode_dataframe(pier_df)
                pier_data = structures_group["Pier Data"][:]
                pier_us_info = pier_df[["US Profile (Index)", "US Profile (Count)"]]
                pier_ds_info = pier_df[["DS Profile (Index)", "DS Profile (Count)"]]

                pier_df["pier_us_shape"] = [
                    pier_data[start:start + count] for start, count in pier_us_info.to_numpy()
                ]
                pier_df["pier_ds_shape"] = [
                    pier_data[start:start + count] for start, count in pier_ds_info.to_numpy()
                ]
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
                    lambda row: row["CL Sta R"] if pd.notna(row["CL Sta R"]) else
                    row.get("sa2d_geom").length if row.get("sa2d_geom") is not None else 0,
                    axis=1,
                )
                mult_open_df = mult_open_df[
                    ["structure_id"] + [x for x in mult_open_df.columns if x not in struc_attrib.columns]]
            except KeyError:
                mult_open_df = pd.DataFrame(columns=["structure_id"])
            mult_open_group_dict = {str_id: group.reset_index(drop=True) for str_id, group in mult_open_df.groupby(
                "structure_id")}
            struct_attrs["multiple_opening_data"] = struct_attrs["structure_id"].map(
                lambda sid: mult_open_group_dict.get(sid, pd.DataFrame())
            )

            # Abutments
            try:
                abutment_df = pd.DataFrame(structures_group["Abutment Attributes"][:])
                abutment_df = _decode_dataframe(abutment_df)
                abutment_data = structures_group["Abutment Data"][:]
                abut_us_info = abutment_df[["US Profile (Index)", "US Profile (Count)"]]
                abut_ds_info = abutment_df[["DS Profile (Index)", "DS Profile (Count)"]]

                abutment_df["abutment_us_shape"] = [
                    abutment_data[start:start + count] for start, count in abut_us_info.to_numpy()
                ]
                abutment_df["abutment_ds_shape"] = [
                    abutment_data[start:start + count] for start, count in abut_ds_info.to_numpy()
                ]

            except KeyError:
                abutment_df = pd.DataFrame(columns=["Structure ID"])
            abutment_group_dict = {
                str_id: group.reset_index(drop=True) for str_id, group in abutment_df.groupby("Structure ID")
            }
            struct_attrs["abutment_data"] = struct_attrs["structure_id"].map(
                lambda sid: abutment_group_dict.get(sid, pd.DataFrame())
            )

            # Bridge coefficient
            try:
                bridge_coeff = structures_group['Bridge Coefficient Attributes'][:]
                coeff_dict = {row['Structure ID']: row for row in bridge_coeff}
                struct_attrs['bridge_coefficient'] = struct_attrs['structure_id'].map(
                    lambda sid: coeff_dict.get(sid, np.zeros(0, dtype=bridge_coeff.dtype))
                )
            except KeyError:
                empty = np.zeros(0)
                struct_attrs['bridge_coefficient'] = struct_attrs['structure_id'].map(lambda sid: empty)
        return struct_attrs
