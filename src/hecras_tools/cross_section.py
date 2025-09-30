"""Utilities for working with HEC-RAS cross section data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import h5py
import pandas as pd
from shapely.geometry import LineString

from hecras_tools.utils import safe_literal_eval


@dataclass(slots=True)
class CrossSectionRecord:
    """Container for the attributes associated with a single cross section."""

    river: str | None
    reach: str | None
    station: float | str | None
    name: str | None
    skew: float | int | None
    geometry: LineString
    attributes: dict[str, object]
    mannings_n: pd.DataFrame
    station_elevation: pd.DataFrame
    ineffective: pd.DataFrame
    blocked_obstruction: pd.DataFrame
    lid_data: pd.DataFrame | None

    def copy(self) -> "CrossSectionRecord":
        """Return a deep copy of the record for defensive access."""

        return CrossSectionRecord(
            river=self.river,
            reach=self.reach,
            station=self.station,
            name=self.name,
            skew=self.skew,
            geometry=self.geometry,
            attributes=dict(self.attributes),
            mannings_n=self.mannings_n.copy(),
            station_elevation=self.station_elevation.copy(),
            ineffective=self.ineffective.copy(),
            blocked_obstruction=self.blocked_obstruction.copy(),
            lid_data=None if self.lid_data is None else self.lid_data.copy(),
        )


class CrossSectionData:
    """Load cross section data from a HEC-RAS geometry HDF file.

    The class mirrors the information produced by the legacy
    :func:`get_xs_data_full` helper but exposes convenience methods for
    retrieving the nested data frames for an individual river station.
    """

    def __init__(self, hdf_file: str):
        self.hdf_file = hdf_file
        self._dataframe = self._load_dataframe()
        self._records = self._build_record_lookup()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def available_stations(self) -> Iterable[str]:
        """Return iterable of available river station identifiers."""

        return self._records.keys()

    def full_dataframe(self) -> pd.DataFrame:
        """Return a copy of the consolidated cross section dataframe."""

        return self._dataframe.copy()

    def mann_n_df(self, rs: float | str) -> pd.DataFrame:
        """Return the Manning's n values for the requested station."""

        return self._get_record(rs).mannings_n.copy()

    def station_elevation_df(self, rs: float | str) -> pd.DataFrame:
        """Return the station-elevation profile for the requested station."""

        return self._get_record(rs).station_elevation.copy()

    def ineffective_df(self, rs: float | str) -> pd.DataFrame:
        """Return the ineffective flow area dataframe for the station."""

        return self._get_record(rs).ineffective.copy()

    def blocked_obstruction_df(self, rs: float | str) -> pd.DataFrame:
        """Return the blocked obstruction dataframe for the station."""

        return self._get_record(rs).blocked_obstruction.copy()

    def lid_data_df(self, rs: float | str) -> pd.DataFrame | None:
        """Return deck (lid) data for the station if it exists."""

        lid_df = self._get_record(rs).lid_data
        return None if lid_df is None else lid_df.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_station(self, rs: float | str) -> str:
        if isinstance(rs, str):
            normalized = safe_literal_eval(rs)
        else:
            normalized = rs
        return str(normalized)

    def _get_record(self, rs: float | str) -> CrossSectionRecord:
        key = self._normalize_station(rs)
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No cross section found for RS {rs!r}") from exc

    def _build_record_lookup(self) -> dict[str, CrossSectionRecord]:
        lookup: dict[str, CrossSectionRecord] = {}
        for _, row in self._dataframe.iterrows():
            lookup[str(row["RS_key"])] = CrossSectionRecord(
                river=row.get("River"),
                reach=row.get("Reach"),
                station=row.get("RS"),
                name=row.get("Name"),
                skew=row.get("Skew"),
                geometry=row.get("geometry"),
                attributes={
                    column: row[column]
                    for column in row.index
                    if column
                    not in {
                        "River",
                        "Reach",
                        "RS",
                        "RS_key",
                        "Name",
                        "Skew",
                        "geometry",
                        "Manning's n df",
                        "Station Elevation df",
                        "Ineffective df",
                        "Blocked obstruction df",
                        "Lid Data df",
                    }
                },
                mannings_n=row["Manning's n df"],
                station_elevation=row["Station Elevation df"],
                ineffective=row["Ineffective df"],
                blocked_obstruction=row["Blocked obstruction df"],
                lid_data=row["Lid Data df"],
            )
        return lookup

    def _load_dataframe(self) -> pd.DataFrame:
        with h5py.File(self.hdf_file, "r") as hdf:
            xs_attrib = pd.DataFrame(
                hdf["Geometry"]["Cross Sections"]["Attributes"][:]
            )
            xs_attrib = xs_attrib.map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            xs_attrib["RS"] = xs_attrib["RS"].apply(safe_literal_eval)
            if "Skew" not in xs_attrib.columns:
                xs_attrib["Skew"] = 0

            # ------------------------------------------------------------------
            # Geometry points
            # ------------------------------------------------------------------
            no_pts = [
                a[1]
                for a in hdf["Geometry"]["Cross Sections"]["Polyline Info"][:].tolist()
            ]
            xs_attrib["no_pts"] = no_pts
            pts_df = pd.DataFrame(
                hdf["Geometry"]["Cross Sections"]["Polyline Points"][:]
            )
            pts_df.columns = ["x", "y"]
            points_index = 0
            line_strings = []
            for _, row in xs_attrib.iterrows():
                num_points = row["no_pts"]
                line_points = pts_df.iloc[points_index : points_index + num_points]
                line_strings.append(
                    LineString(zip(line_points["x"], line_points["y"]))
                )
                points_index += num_points
            xs_attrib["geometry"] = line_strings

            # ------------------------------------------------------------------
            # Manning's n
            # ------------------------------------------------------------------
            xs_attrib["mann_pts"] = [
                a[1]
                for a in hdf["Geometry"]["Cross Sections"]["Manning's n Info"][:].tolist()
            ]
            mann_pts_df = pd.DataFrame(
                hdf["Geometry"]["Cross Sections"]["Manning's n Values"][:]
            )
            mann_pts_df.columns = ["x", "n"]
            points_index = 0
            mann_pts_dfs = []
            for _, row in xs_attrib.iterrows():
                num_points = row["mann_pts"]
                mann_df = mann_pts_df.iloc[points_index : points_index + num_points]
                mann_df.reset_index(drop=True, inplace=True)
                mann_pts_dfs.append(mann_df)
                points_index += num_points
            xs_attrib["Manning's n df"] = mann_pts_dfs

            # ------------------------------------------------------------------
            # Station-elevation
            # ------------------------------------------------------------------
            xs_attrib["prof_pts"] = [
                a[1]
                for a in hdf["Geometry"]["Cross Sections"]["Station Elevation Info"][:].tolist()
            ]
            sta_elev_df = pd.DataFrame(
                hdf["Geometry"]["Cross Sections"]["Station Elevation Values"][:]
            )
            sta_elev_df.columns = ["x", "y"]
            points_index = 0
            sta_elev_dfs = []
            for _, row in xs_attrib.iterrows():
                num_points = row["prof_pts"]
                prof_df = sta_elev_df.iloc[
                    points_index : points_index + num_points
                ]
                prof_df.reset_index(drop=True, inplace=True)
                sta_elev_dfs.append(prof_df)
                points_index += num_points
            xs_attrib["Station Elevation df"] = sta_elev_dfs

            # ------------------------------------------------------------------
            # Ineffective areas
            # ------------------------------------------------------------------
            try:
                xs_attrib["ineff_pts"] = [
                    a[1]
                    for a in hdf["Geometry"]["Cross Sections"]["Ineffective Info"][:].tolist()
                ]
                ineff_values_df = pd.DataFrame(
                    hdf["Geometry"]["Cross Sections"]["Ineffective Blocks"][:]
                )
                points_index = 0
                ineff_dfs = []
                for _, row in xs_attrib.iterrows():
                    num_points = row["ineff_pts"]
                    ineff_df = ineff_values_df.iloc[
                        points_index : points_index + num_points
                    ]
                    ineff_df.reset_index(drop=True, inplace=True)
                    ineff_dfs.append(ineff_df)
                    points_index += num_points
                xs_attrib["Ineffective df"] = ineff_dfs
            except KeyError:
                xs_attrib["Ineffective df"] = [
                    pd.DataFrame({"Permanent": [0]})
                ] * len(xs_attrib)

            # ------------------------------------------------------------------
            # Blocked obstructions
            # ------------------------------------------------------------------
            try:
                xs_attrib["obs_pts"] = [
                    a[1]
                    for a in hdf["Geometry"]["Cross Sections"]["Obstruction Info"][:].tolist()
                ]
                blocked_obs_values_df = pd.DataFrame(
                    hdf["Geometry"]["Cross Sections"]["Obstruction Blocks"][:]
                )
                points_index = 0
                blocked_obs_dfs = []
                for _, row in xs_attrib.iterrows():
                    num_points = row["obs_pts"]
                    blocked_obs_df = blocked_obs_values_df.iloc[
                        points_index : points_index + num_points
                    ]
                    blocked_obs_df.reset_index(drop=True, inplace=True)
                    blocked_obs_dfs.append(blocked_obs_df)
                    points_index += num_points
                xs_attrib["Blocked obstruction df"] = blocked_obs_dfs
            except KeyError:
                xs_attrib["Blocked obstruction df"] = [
                    pd.DataFrame({"Left Sta": [], "Right Sta": [], "Elevation": []})
                ] * len(xs_attrib)

            # ------------------------------------------------------------------
            # Lid data
            # ------------------------------------------------------------------
            try:
                xs_attrib["Lid_pts_high"] = [
                    a[1]
                    for a in hdf["Geometry"]["Cross Sections"]["Deck High Info"][:].tolist()
                ]
                high_deck_values_df = pd.DataFrame(
                    hdf["Geometry"]["Cross Sections"]["Deck High Values"][:]
                )
                high_deck_values_df.columns = ["Station", "High El."]
                xs_attrib["Lid_pts_low"] = [
                    a[1]
                    for a in hdf["Geometry"]["Cross Sections"]["Deck Low Info"][:].tolist()
                ]
                low_deck_values_df = pd.DataFrame(
                    hdf["Geometry"]["Cross Sections"]["Deck Low Values"][:]
                )
                low_deck_values_df.columns = ["Station", "Low El."]
                points_index_high = 0
                points_index_low = 0
                lid_data_dfs = []
                for _, row in xs_attrib.iterrows():
                    num_points_high = row["Lid_pts_high"]
                    lid_df_high = high_deck_values_df.iloc[
                        points_index_high : points_index_high + num_points_high
                    ]
                    num_points_low = row["Lid_pts_low"]
                    lid_df_low = low_deck_values_df.iloc[
                        points_index_low : points_index_low + num_points_low
                    ]
                    lid_df = lid_df_high.merge(
                        lid_df_low, on="Station", how="outer"
                    )
                    lid_df.reset_index(drop=True, inplace=True)
                    lid_data_dfs.append(lid_df)
                    points_index_high += num_points_high
                    points_index_low += num_points_low
                xs_attrib["Lid Data df"] = lid_data_dfs
            except KeyError:
                xs_attrib["Lid Data df"] = [pd.DataFrame()] * len(xs_attrib)

            xs_attrib.drop(
                columns=[
                    "no_pts",
                    "mann_pts",
                    "prof_pts",
                    "ineff_pts",
                    "obs_pts",
                    "Lid_pts_high",
                    "Lid_pts_low",
                ],
                errors="ignore",
                inplace=True,
            )

        xs_attrib["RS_key"] = xs_attrib["RS"].apply(str)
        return xs_attrib
