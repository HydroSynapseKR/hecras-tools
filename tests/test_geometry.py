from hecras_tools.geometry_operations import GeometryOperations

def test_init():
    ops = GeometryOperations("dummy.hdf")
    assert ops.hdf_file == "dummy.hdf"
