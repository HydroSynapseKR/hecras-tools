from hecras_tools.geometry_operations import GeometryHdf

def test_init():
    ops = GeometryHdf("dummy.hdf")
    assert ops.hdf_file == "dummy.hdf"
