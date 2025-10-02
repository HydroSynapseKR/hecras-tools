# HEC-RAS Tools

A Python package for working with HEC-RAS geometry, terrain, and hydrodynamic results.

## 🚀 Features
- Read boundary condition lines, breaklines, and mesh data
- Extract water surface elevation (WSE) profiles
- Work with 1D and 2D flow geometry
- Export rasters and shapefiles

## 📦 Installation
```bash
git clone https://github.com/your-username/hecras-tools.git
cd hecras-tools
pip install -e .
```

Or with Docker:
```bash
docker build -t hecras-tools .
docker run -it --rm hecras-tools
```

## 💻 VS Code Devcontainer
Open this repo in VS Code, and it will auto-load inside Docker with dependencies ready.

## 🧰 Available Classes & Methods

### `GeometryHdf`
Utilities for working with HEC-RAS geometry HDF files.

- `get_crs()` – Read the coordinate reference system stored in the file.
- `get_bc_lines()` – Return boundary condition lines as a GeoDataFrame.
- `get_2d_boundary()` – Build polygons for 2D flow area boundaries.
- `get_2d_breaklines()` – Extract 2D flow area breaklines as geometries.
- `get_2d_refinement_regions()` – Load refinement region polygons.
- `get_2d_flow_area_mesh()` – Construct GeoDataFrames for 2D mesh cells and faces.
- `get_structures()` – Retrieve hydraulic structure property tables.

### `CrossSectionData`
Helper for exploring 1D cross-section information stored in geometry HDF files.

- `available_stations()` – Enumerate river station identifiers present in the file.
- `full_dataframe()` – Return the full consolidated cross-section attribute table.
- `mann_n_df(river, reach, rs)` – Retrieve Manning's *n* values for a station.
- `station_elevation_df(river, reach, rs)` – Retrieve station–elevation profiles.
- `ineffective_df(river, reach, rs)` – Retrieve ineffective flow areas.
- `blocked_obstruction_df(river, reach, rs)` – Retrieve blocked obstruction data.
- `lid_data_df(river, reach, rs)` – Retrieve deck (lid) geometry where available.

### `CrossSectionRecord`
A dataclass representing a single cross-section. Use the `.copy()` method for a deep
copy of all associated attributes before mutating data in client code.

## 🛠️ Example
```python
from hecras_tools import GeometryHdf, CrossSectionData

ops = GeometryHdf("example.hdf")
print(ops.get_crs())

xs_data = CrossSectionData("example.hdf")
print(list(xs_data.available_stations())[:3])
```
