"""Utilities for working with HEC-RAS cross-section data."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import LineString, MultiLineString

from hecras_tools.utils import safe_literal_eval, CROSS_SECTION_RENAME_MAP


@dataclass(slots=True)
class CrossSectionRecord:
    """Container for the attributes associated with a single cross-section record."""

    ID_key: str
    river: str
    reach: str
    station: float | int
    description: str | None
    reach_len_left: float | None
    reach_len_chan: float | None
    reach_len_right: float | None
    bank_sta_left: float
    bank_sta_right: float
    n_mode: str
    contr_coeff: float | None
    expan_coeff: float | None
    hp_count: int
    hp_start_el: int | float
    hp_ver_incr: int | float
    hp_lob_slices: int
    hp_chan_slices: int
    hp_rob_slices: int
    default_centerline: int
    skew: float = 0.0
    pc_invert: int | float | None = None
    pc_width: int | float | None = None
    pc_mann: int | float | None = None
    deck_preissman_slot: int | None = None
    contr_coeff_usf: float | None = None
    expan_coeff_usf: float | None = None
    geometry_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    geometry_parts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    geometry: LineString | MultiLineString | None = None
    manning_n_start: int = 0
    manning_n_count: int = 0
    manning_n: pd.DataFrame = field(default_factory=pd.DataFrame)
    station_elevation_start: int = 0
    station_elevation_count: int = 0
    station_elevation: pd.DataFrame = field(default_factory=pd.DataFrame)
    ineffective_start: int = 0
    ineffective_count: int = 0
    ineffective: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocked_start: int = 0
    blocked_count: int = 0
    blocked_obstruction: pd.DataFrame = field(default_factory=pd.DataFrame)
    lid_deck_high_start: int = 0
    lid_deck_count: int = 0
    lid_low_start: int = 0
    lid_low_count: int = 0
    lid_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> "CrossSectionRecord":
        """Return a deep copy of the record for defensive access."""
        copied_fields: dict[str, object] = {}
        for field_def in fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, pd.DataFrame):
                copied_fields[field_def.name] = value.copy(deep=True)
            elif isinstance(value, np.ndarray):
                copied_fields[field_def.name] = value.copy()
            elif isinstance(value, (LineString, MultiLineString)):
                copied_fields[field_def.name] = wkb.loads(wkb.dumps(value))
            else:
                copied_fields[field_def.name] = value
        return CrossSectionRecord(**copied_fields)


class CrossSectionData:
    """
    Load cross-section data from a HEC-RAS geometry HDF file.
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
        """Return a copy of the consolidated cross-section dataframe with entire data."""

        return self._dataframe.copy()

    def mann_n_df(self, river: str, reach: str, rs: float | int) -> pd.DataFrame:
        """Return the Manning's n values for the requested station."""
        key = 'River: ' + river + ' Reach: ' + reach + ' RS: ' + str(safe_literal_eval(rs))

        return self._get_record(key).manning_n.copy()

    def station_elevation_df(self, river: str, reach: str, rs: float | int) -> pd.DataFrame:
        """Return the station-elevation profile for the requested station."""
        key = 'River: ' + river + ' Reach: ' + reach + ' RS: ' + str(safe_literal_eval(rs))

        return self._get_record(key).station_elevation.copy()

    def ineffective_df(self, river: str, reach: str, rs: float | int) -> pd.DataFrame:
        """Return the ineffective flow area dataframe for the station."""
        key = 'River: ' + river + ' Reach: ' + reach + ' RS: ' + str(safe_literal_eval(rs))

        return self._get_record(key).ineffective.copy()

    def blocked_obstruction_df(self, river: str, reach: str, rs: float | int) -> pd.DataFrame:
        """Return the blocked obstruction dataframe for the station."""
        key = 'River: ' + river + ' Reach: ' + reach + ' RS: ' + str(safe_literal_eval(rs))

        return self._get_record(key).blocked_obstruction.copy()

    def lid_data_df(self, river: str, reach: str, rs: float | int) -> pd.DataFrame | None:
        """Return deck (lid) data for the station if it exists."""
        key = 'River: ' + river + ' Reach: ' + reach + ' RS: ' + str(safe_literal_eval(rs))

        lid_df = self._get_record(key).lid_data
        return lid_df.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_record(self, key: str) -> CrossSectionRecord:
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"No cross section found for {key!r}") from exc

    def _build_record_lookup(self) -> dict[str, CrossSectionRecord]:
        valid_fields = {f.name for f in fields(CrossSectionRecord)}
        lookup = {
            str(row['ID_key']): CrossSectionRecord(**{k: v for k, v in row.items() if k in valid_fields})
            for _, row in self._dataframe.iterrows()
        }
        return lookup

    def _load_dataframe(self) -> pd.DataFrame:
        with (h5py.File(self.hdf_file, "r") as hdf):
            xs_attrib = pd.DataFrame(
                hdf["Geometry"]["Cross Sections"]["Attributes"][:]
            )
            xs_attrib = xs_attrib.map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            xs_attrib["RS"] = xs_attrib["RS"].apply(safe_literal_eval)
            xs_attrib["ID_key"] = ('River: ' + xs_attrib['River'] + ' Reach: ' + xs_attrib['Reach'] + ' RS: ' +
                                   xs_attrib['RS'].astype(str))

            # ------------------------------------------------------------------
            # Geometry points
            # ------------------------------------------------------------------
            geometry_info = hdf["Geometry"]["Cross Sections"]["Polyline Info"][:]
            points = hdf["Geometry"]["Cross Sections"]["Polyline Points"][:]
            parts = hdf["Geometry"]["Cross Sections"]["Polyline Parts"][:]
            xs_attrib['geometry_points'] = xs_attrib.index.map(
                lambda i: points[geometry_info[i, 0]: geometry_info[i, 0] + geometry_info[i, 1]]
            )
            xs_attrib['geometry_parts'] = xs_attrib.index.map(
                lambda i: parts[geometry_info[i, 2]: geometry_info[i, 2] + geometry_info[i, 3]]
            )
            geom_list = []
            for _, row in xs_attrib.iterrows():
                geom = MultiLineString(
                    [LineString(row['geometry_points'][i[0]:i[0] + i[1]]) for i in row['geometry_parts']]
                )
                geom = geom if len(geom.geoms) > 1 else geom.geoms[0]
                geom_list.append(geom)
            xs_attrib["geometry"] = geom_list

            # ------------------------------------------------------------------
            # Manning's n
            # ------------------------------------------------------------------
            mann_info = hdf["Geometry"]["Cross Sections"]["Manning's n Info"]
            xs_attrib[["manning_n_start", "manning_n_count"]] = mann_info[:]
            mann_pts = hdf["Geometry"]["Cross Sections"]["Manning's n Values"][:]
            xs_attrib["manning_n"] = [mann_pts[start:start + count] for start, count in mann_info[:]]

            # ------------------------------------------------------------------
            # Station-elevation
            # ------------------------------------------------------------------
            prof_info = hdf["Geometry"]["Cross Sections"]["Station Elevation Info"]
            xs_attrib[["station_elevation_start", "station_elevation_count"]] = mann_info[:]
            prof_values = hdf["Geometry"]["Cross Sections"]["Station Elevation Values"][:]
            xs_attrib["station_elevation"] = [prof_values[start:start + count] for start, count in prof_info[:]]

            # ------------------------------------------------------------------
            # Ineffective areas
            # ------------------------------------------------------------------
            try:
                ineff_info = hdf["Geometry"]["Cross Sections"]["Ineffective Info"]
                xs_attrib[["ineffective_start", "ineffective_count"]] = ineff_info[:]
                ineff_values = hdf["Geometry"]["Cross Sections"]["Ineffective Blocks"][:]
                xs_attrib["ineffective"] = [ineff_values[start:start + count] for start, count in ineff_info[:]]
            except KeyError:
                pass

            # ------------------------------------------------------------------
            # Blocked obstructions
            # ------------------------------------------------------------------
            try:
                obs_info = hdf["Geometry"]["Cross Sections"]["Obstruction Info"]
                xs_attrib[["blocked_start", "blocked_count"]] = ineff_info[:]
                obs_values = hdf["Geometry"]["Cross Sections"]["Obstruction Blocks"][:]
                xs_attrib["bocked_obstruction"] = [obs_values[start:start + count] for start, count in obs_info[:]]
            except KeyError:
                pass

            # ------------------------------------------------------------------
            # Lid data
            # ------------------------------------------------------------------
            try:
                deck_high_info = hdf["Geometry"]["Cross Sections"]["Deck High Info"]
                deck_low_info = hdf["Geometry"]["Cross Sections"]["Deck Low Info"]
                lid_info = np.concatenate((deck_high_info, deck_low_info), axis=1)
                xs_attrib[["lid_deck_high_start", "lid_deck_high_count"]] = deck_high_info[:]
                xs_attrib[["lid_deck_low_start", "lid_deck_low_count"]] = deck_low_info[:]
                deck_high_values = hdf["Geometry"]["Cross Sections"]["Deck High Values"][:]
                deck_low_values = hdf["Geometry"]["Cross Sections"]["Deck Low Values"][:]

                lid_data = []
                for high_start, high_count, low_start, low_count in lid_info[:]:
                    high = deck_high_values[high_start: high_start + high_count]
                    low = deck_low_values[low_start: low_start + low_count]

                    # lookup dictionaries
                    high_dict = dict(high)
                    low_dict = dict(low)

                    # Union of stations, sorted
                    all_stations = np.union1d(high[:, 0], low[:, 0])

                    # List comprehension to build consolidated array
                    consolidated = np.array([
                        [station, high_dict.get(station, np.nan), low_dict.get(station, np.nan)]
                        for station in all_stations
                    ])
                    lid_data.append(consolidated)

                xs_attrib["lid data"] = lid_data
            except KeyError:
                pass
        xs_attrib.rename(columns=CROSS_SECTION_RENAME_MAP, inplace=True)
        return xs_attrib
