from hecras_tools.geometry_operations import GeometryHdf
from hecras_tools.cross_section import CrossSectionData


def test_init():
    ops = GeometryHdf("dummy.hdf")
    assert ops.hdf_file == "dummy.hdf"
