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

## 🛠️ Example
```python
from hecras_tools.geometry_operations import GeometryHdf

ops = GeometryHdf("example.hdf")
bc_lines = ops.get_bc_lines()
print(bc_lines.head())
```
