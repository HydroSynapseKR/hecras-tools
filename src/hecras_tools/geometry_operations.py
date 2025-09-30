import h5py
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon
import shapely.errors
from shapely import polygonize_full
import numpy as np
from hecras_tools.utils import safe_literal_eval


class GeometryHdf:
    def __init__(self, hdf_file: str):
        """
        Work with HEC-RAS HDF geometry files.
        """
        self.hdf_file = hdf_file
        self.crs = self.get_crs()

    # -------------------------------------------------------------------------
    # CRS
    # -------------------------------------------------------------------------
    def get_crs(self) -> str | None:
        """Extract CRS string from HDF attributes."""
        with h5py.File(self.hdf_file, "r") as hdf:
            try:
                crs = hdf.attrs["Projection"]
                return crs.decode("utf-8")
            except KeyError:
                return None

    # -------------------------------------------------------------------------
    # Boundary condition lines
    # -------------------------------------------------------------------------
    def get_bc_lines(self) -> gpd.GeoDataFrame:
        """Extract boundary condition lines."""
        with h5py.File(self.hdf_file, "r") as hdf:
            bc_lines = pd.DataFrame(
                hdf["Geometry"]["Boundary Condition Lines"]["Attributes"][:]
            )
            bc_lines = bc_lines.map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

            bc_lines_info = pd.DataFrame(
                hdf["Geometry"]["Boundary Condition Lines"]["Polyline Info"][:]
            )
            bc_lines_info.columns = ["start_index", "no_pts", "2", "3"]
            bc_lines_info["end_index"] = (
                bc_lines_info["start_index"] + bc_lines_info["no_pts"]
            )

            pts_df = pd.DataFrame(
                hdf["Geometry"]["Boundary Condition Lines"]["Polyline Points"][:]
            )
            pts_df.columns = ["x", "y"]

            def create_linestring(row):
                pts = pts_df.loc[
                    row["start_index"]: row["end_index"] - 1, ["x", "y"]
                ].values
                return LineString(pts)

            bc_lines["geometry"] = bc_lines_info.apply(create_linestring, axis=1)

        return gpd.GeoDataFrame(bc_lines, geometry="geometry", crs=self.crs)

    # -------------------------------------------------------------------------
    # 2D Flow Areas boundary
    # -------------------------------------------------------------------------
    def get_2d_boundary(self) -> gpd.GeoDataFrame:
        """Extract 2D Flow Area boundaries as polygons."""
        with h5py.File(self.hdf_file, "r") as hdf:
            fa_names = [
                n.decode("utf-8") for n in hdf["Geometry"]["2D Flow Areas"].keys()
            ]
            records = []
            for fa_name in fa_names:
                boundary_info = pd.DataFrame(
                    hdf["Geometry"]["2D Flow Areas"][fa_name]["Boundary Info"][:]
                )
                boundary_info.columns = ["start_index", "no_pts", "2", "3"]
                boundary_info["end_index"] = (
                    boundary_info["start_index"] + boundary_info["no_pts"]
                )

                pts_df = pd.DataFrame(
                    hdf["Geometry"]["2D Flow Areas"][fa_name]["Boundary Points"][:]
                )
                pts_df.columns = ["x", "y"]

                for _, row in boundary_info.iterrows():
                    pts = pts_df.loc[
                        row["start_index"]: row["end_index"] - 1, ["x", "y"]
                    ].values
                    records.append(
                        {"flow_area": fa_name, "geometry": Polygon(pts)}
                    )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.crs)

    # -------------------------------------------------------------------------
    # 2D Flow Area breaklines
    # -------------------------------------------------------------------------
    def get_2d_breaklines(self) -> gpd.GeoDataFrame:
        """Extract 2D breaklines."""
        columns = ['geometry', 'Cell Spacing Near', 'Cell Spacing Far', 'Near Repeats', 'Protection Radius']
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                breaklines = pd.DataFrame(hdf['Geometry']['2D Flow Area Break Lines']['Attributes'][:])
                bl_info = pd.DataFrame(hdf['Geometry']['2D Flow Area Break Lines']['Polyline Info'][:])
                bl_info.columns = ['start_index', 'no_pts', '2', '3']
                bl_info['end_index'] = bl_info['start_index'] + bl_info['no_pts']
                pts_df = pd.DataFrame(hdf['Geometry']['2D Flow Area Break Lines']['Polyline Points'][:])
                pts_df.columns = ['x', 'y']

                def create_linestring(row):
                    start_idx = row['start_index']
                    end_idx = row['end_index'] - 1
                    points = pts_df.loc[start_idx:end_idx, ['x', 'y']].values
                    try:
                        return LineString(points)
                    except shapely.errors.GEOSException as e:
                        if end_idx - start_idx < 1:
                            print(f'polyline at {start_idx} has less than 2 points')
                            return LineString()
                        else:
                            raise e

                bl_info['geometry'] = bl_info.apply(create_linestring, axis=1)
                breaklines['geometry'] = bl_info['geometry']
            except KeyError:
                return gpd.GeoDataFrame([], columns=columns, geometry='geometry', crs=self.crs)

        return gpd.GeoDataFrame(breaklines, geometry='geometry', crs=self.crs)

    # -------------------------------------------------------------------------
    # 2D Flow Area refinement regions
    # -------------------------------------------------------------------------
    def get_2d_refinement_regions(self) -> gpd.GeoDataFrame:
        """Extract 2D refinement regions."""
        columns = ['geometry', 'Spacing dx', 'Spacing dy', 'Shift dx', 'Shift dx', 'Perimeter Spacing',
                   'Near Spacing Repeats', 'Far Spacing', 'Protection Radius']
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                refinements = pd.DataFrame(hdf['Geometry']['2D Flow Area Refinement Regions']['Attributes'][:])
                r_info = pd.DataFrame(hdf['Geometry']['2D Flow Area Break Lines']['Polygon Info'][:])
                r_info.columns = ['start_index', 'no_pts', '2', '3']
                r_info['end_index'] = r_info['start_index'] + r_info['no_pts']
                pts_df = pd.DataFrame(hdf['Geometry']['2D Flow Area Break Lines']['Polygon Points'][:])
                pts_df.columns = ['x', 'y']

                def create_polygon(row):
                    start_idx = row['start_index']
                    end_idx = row['end_index'] - 1
                    points = pts_df.loc[start_idx:end_idx, ['x', 'y']].values
                    try:
                        return Polygon(points)
                    except shapely.errors.GEOSException as e:
                        if end_idx - start_idx < 3:
                            print(f'polygon at {start_idx} has less than 3 points')
                            return Polygon()
                        else:
                            raise e

                r_info['geometry'] = r_info.apply(create_polygon, axis=1)
                refinements['geometry'] = r_info['geometry']
            except KeyError:
                return gpd.GeoDataFrame([], columns=columns, geometry='geometry', crs=self.crs)

        return gpd.GeoDataFrame(refinements, geometry='geometry', crs=self.crs)

    # -------------------------------------------------------------------------
    # 2D Flow Area mesh cell faces
    # -------------------------------------------------------------------------
    def get_mesh_cell_faces(self) -> gpd.GeoDataFrame:
        """Return 2D flow mesh cell faces.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell faces if 2D areas exist.
        """
        with h5py.File(self.hdf_file) as hdf_file:
            perimeters = [name for name, item in hdf_file['Geometry']['2D Flow Areas'].items()
                          if isinstance(item, h5py.Group)]
            gdf_columns = ['geometry', 'Face_Index', 'Name']
            if not perimeters:
                return gpd.GeoDataFrame([], columns=gdf_columns, crs=self.crs)
            all_faces = pd.DataFrame()
            for each in perimeters:
                face_points = hdf_file['Geometry']['2D Flow Areas'][each]['FacePoints Coordinate'][:]
                face_perim_values = hdf_file['Geometry']['2D Flow Areas'][each]['Faces Perimeter Values'][:]
                faces = pd.DataFrame(hdf_file['Geometry']['2D Flow Areas'][each]['Faces FacePoint Indexes'][:])
                faces.columns = ['begin_pt_index', 'end_pt_index']
                faces[['start_perim_idx', 'count']] = hdf_file['Geometry']['2D Flow Areas'][each]['Faces Perimeter Info']
                faces['end_perim_idx'] = faces['start_perim_idx'] + faces['count']
                faces['Face_Index'] = faces.index
                faces['start_pt'] = faces['begin_pt_index'].apply(lambda x: tuple(face_points[x]))
                faces['end_pt'] = faces['end_pt_index'].apply(lambda x: tuple(face_points[x]))
                faces['mid_coords'] = [
                    list(map(tuple, face_perim_values[start:end]))
                    for start, end in zip(faces['start_perim_idx'], faces['end_perim_idx'])
                ]
                faces['geometry'] = [
                    LineString([start] + list(mid) + [end])
                    for start, mid, end in zip(faces['start_pt'], faces['mid_coords'], faces['end_pt'])
                ]
                faces['Name'] = each
                faces = faces[gdf_columns]
                all_faces = pd.concat([all_faces, faces])
            gdf = gpd.GeoDataFrame(all_faces, geometry='geometry', crs=self.crs)
            return gdf

    # -------------------------------------------------------------------------
    # 2D Flow Area Mesh cell polygons
    # -------------------------------------------------------------------------
    def get_2d_mesh(self) -> gpd.GeoDataFrame:
        """Return 2D flow mesh cell polygons.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell faces if 2D areas exist.
        """
        faces_all = self.get_mesh_cell_faces()
        gdf_columns = ['2D_Area', 'Cell_Index', 'geometry']
        with h5py.File(self.hdf_file) as hdf_file:
            perimeters = [name for name, item in hdf_file['Geometry']['2D Flow Areas'].items()
                          if isinstance(item, h5py.Group)]
            all_mesh = pd.DataFrame([], columns=gdf_columns)
            for each in perimeters:
                # Filter and sort faces
                faces = faces_all.loc[faces_all['2D_Area'] == each].sort_values(by='Face_Index').reset_index(drop=True)

                # Convert geometry column to NumPy array
                face_geoms = faces['geometry'].to_numpy()

                # Load mesh polygon data
                mesh_data = hdf_file['Geometry']['2D Flow Areas'][each]
                mesh_polys = pd.DataFrame(mesh_data['Cells Face and Orientation Info'][:], columns=['start', 'count'])
                mesh_polys['end'] = mesh_polys['start'] + mesh_polys['count']

                # Convert to NumPy for fast slicing
                starts = mesh_polys['start'].to_numpy()
                ends = mesh_polys['end'].to_numpy()
                faces_indices = mesh_data['Cells Face and Orientation Values'][:, 0]

                # Build index_list
                index_list = np.empty(len(mesh_polys), dtype=object)
                for i in range(len(mesh_polys)):
                    index_list[i] = faces_indices[starts[i]:ends[i]]
                mesh_polys['index_list'] = index_list

                # Build geometry
                geometries = np.empty(len(mesh_polys), dtype=object)
                for i, idx_list in enumerate(index_list):
                    if len(idx_list) > 1:
                        result = polygonize_full(face_geoms[idx_list])
                        geometries[i] = Polygon((result[0] or result[3]).geoms[0])
                    else:
                        geometries[i] = None
                mesh_polys['geometry'] = geometries

                mesh_polys['2D_Area'] = each
                mesh_polys['Cell_Index'] = mesh_polys.index
                mesh_polys = mesh_polys.dropna(subset='geometry')
                mesh_polys = mesh_polys[gdf_columns]
                all_mesh = pd.concat([all_mesh, mesh_polys], ignore_index=True)
            gdf = gpd.GeoDataFrame(all_mesh, geometry='geometry', crs=self.crs)
            return gdf

    # -------------------------------------------------------------------------
    # 2D Flow Area cell center points
    # -------------------------------------------------------------------------
    def get_2d_points(self) -> gpd.GeoDataFrame:
        """Extract 2D mesh points."""
        with h5py.File(self.hdf_file) as hdf_file:
            cell_pts = pd.DataFrame(hdf_file['Geometry']['2D Flow Areas']['Cell Points'])
            cell_pts.columns = ['x', 'y']
            cell_pts['geometry'] = gpd.points_from_xy(cell_pts['x'], cell_pts['y'])
        return gpd.GeoDataFrame(cell_pts, crs=self.crs)

    # -------------------------------------------------------------------------
    # 1D Flow Paths
    # -------------------------------------------------------------------------
    def get_flow_paths(self) -> gpd.GeoDataFrame:
        """Extract 1D left overbank and right overbank flow paths."""
        data = {'geometry': [], 'River': [], 'Reach': [], 'Type': []}
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                flow_paths = pd.DataFrame(hdf['Geometry']['River Flow Paths']['Flow Path Lines Info'][:])
            except KeyError:
                return gpd.GeoDataFrame(data, geometry='geometry', crs=self.crs)
            flow_paths.columns = ['start_index', 'no_pts', '2', '3']
            flow_paths['end_index'] = flow_paths['start_index'] + flow_paths['no_pts']
            pts_df = pd.DataFrame(hdf['Geometry']['River Flow Paths']['Flow Path Lines Points'][:])
            pts_df.columns = ['x', 'y']

            def create_linestring(row):
                start_idx = row['start_index']
                end_idx = row['end_index'] - 1
                points = pts_df.loc[start_idx:end_idx, ['x', 'y']].values
                return LineString(points)

            flow_paths['geometry'] = flow_paths.apply(create_linestring, axis=1)

            xs_df = self.get_cross_sections()

            unique_id = list(set(list(zip(xs_df['River'], xs_df['Reach']))))
            for each_river in unique_id:
                sub_xs_df = xs_df[(xs_df['River'] == each_river[0]) & (xs_df['Reach'] == each_river[1])]
                sub_xs_df = sub_xs_df.loc[[sub_xs_df['RS'].idxmin(), sub_xs_df['RS'].idxmax()]]
                for geom in flow_paths['geometry']:
                    if all([geom.intersects(x) for x in sub_xs_df['geometry']]):
                        fp_stations = [x.project(geom.intersection(x)) for x in sub_xs_df['geometry']]
                        lob_dist = [abs(x - y) for x, y in zip(fp_stations, sub_xs_df['Left Bank'])]
                        rob_dist = [abs(x - y) for x, y in zip(fp_stations, sub_xs_df['Right Bank'])]
                        is_lob = [x > y for x, y in zip(rob_dist, lob_dist)]
                        if all(is_lob):
                            data['Type'].append('LOB')
                        elif all(not item for item in is_lob):
                            data['Type'].append('ROB')
                        else:
                            print(f"could not determine type for river {each_river}")
                            data['Type'].append('')
                        data['geometry'].append(geom)
                        data['River'].append(each_river[0])
                        data['Reach'].append(each_river[1])
        return gpd.GeoDataFrame(data, geometry='geometry', crs=self.crs)

    # -------------------------------------------------------------------------
    # 1D Bank Lines
    # -------------------------------------------------------------------------
    def get_bank_lines(self) -> gpd.GeoDataFrame:
        """Extract 1D bank lines."""
        with h5py.File(self.hdf_file, "r") as hdf:
            bank_lines = pd.DataFrame(
                hdf["Geometry"]["Bank Lines"]["Attributes"][:]
            )
            bank_lines = bank_lines.map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

            bank_info = pd.DataFrame(
                hdf["Geometry"]["Bank Lines"]["Polyline Info"][:]
            )
            bank_info.columns = ["start_index", "no_pts", "2", "3"]
            bank_info["end_index"] = bank_info["start_index"] + bank_info["no_pts"]

            pts_df = pd.DataFrame(
                hdf["Geometry"]["Bank Lines"]["Polyline Points"][:]
            )
            pts_df.columns = ["x", "y"]

            def create_linestring(row):
                pts = pts_df.loc[
                    row["start_index"]: row["end_index"] - 1, ["x", "y"]
                ].values
                return LineString(pts)

            bank_lines["geometry"] = bank_info.apply(create_linestring, axis=1)

        return gpd.GeoDataFrame(bank_lines, geometry="geometry", crs=self.crs)

    # -------------------------------------------------------------------------
    # 1D Edge boundaries
    # -------------------------------------------------------------------------
    def get_edge_lines(self, flag: bool = False) -> gpd.GeoDataFrame:
        """Extract edge lines."""
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                edge_lines = pd.DataFrame(hdf['Geometry']['River Edge Lines']['Polyline Info'][:])
            except KeyError:
                return gpd.GeoDataFrame()
            edge_lines.columns = ['start_index', 'no_pts', '2', '3']
            edge_lines['end_index'] = edge_lines['start_index'] + edge_lines['no_pts']
            pts_df = pd.DataFrame(hdf['Geometry']['River Edge Lines']['Polyline Points'][:])
            pts_df.columns = ['x', 'y']

            def create_linestring(row):
                start_idx = row['start_index']
                end_idx = row['end_index'] - 1
                points = pts_df.loc[start_idx:end_idx, ['x', 'y']].values
                return LineString(points)

            edge_lines['geometry'] = edge_lines.apply(create_linestring, axis=1)

            xs_gdf = self.get_cross_sections()
            xs_gdf['geometry'] = xs_gdf['geometry'].buffer(0.1)
            river_gdf = self.get_rivers()

            unique_id = list(set(zip(river_gdf['River'], river_gdf['Reach'])))
            for each_river in unique_id:
                river = each_river[0]
                reach = each_river[1]
                sub_xs_gdf = xs_gdf[(xs_gdf['River'] == river) & (xs_gdf['Reach'] == reach)]
                us_ds_xs = sub_xs_gdf[sub_xs_gdf['RS'].isin([sub_xs_gdf['RS'].min(), sub_xs_gdf['RS'].max()])]
                mask = edge_lines.geometry.apply(lambda geom: all(geom.intersects(x) for x in us_ds_xs.geometry))
                edge_lines.loc[mask, 'River'] = river
                edge_lines.loc[mask, 'Reach'] = reach
        group_counts = edge_lines.groupby(['River', 'Reach']).size()
        invalid_groups = group_counts[group_counts != 2].reset_index()
        if len(invalid_groups):
            print("following rivers may have returned invalid edge lines")
            print(invalid_groups)
            if flag:
                raise IndexError("invalid edge lines returned")
        return gpd.GeoDataFrame(edge_lines, geometry='geometry', crs=self.crs)

    # -------------------------------------------------------------------------
    # 1D Rivers
    # -------------------------------------------------------------------------
    def get_rivers(self) -> gpd.GeoDataFrame:
        """Extract river centerline data."""
        with h5py.File(self.hdf_file, 'r') as hdf:
            river_attrib = pd.DataFrame(hdf['Geometry']['River Centerlines']['Attributes'][:])
            river_attrib = river_attrib.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            no_pts = [a[1] for a in hdf['Geometry']['River Centerlines']['Polyline Info'][:].tolist()]
            river_attrib['no_pts'] = no_pts
            pts_df = pd.DataFrame(hdf['Geometry']['River Centerlines']['Polyline Points'][:])
            pts_df.columns = ['x', 'y']
            points_index = 0
            line_strings = []
            for _, row in river_attrib.iterrows():
                num_points = row['no_pts']
                line_points = pts_df.iloc[points_index:points_index + num_points]
                line_strings.append(LineString(zip(line_points['x'], line_points['y'])))
                points_index += num_points
            river_attrib['geometry'] = line_strings
            river_attrib.rename(columns={'River Name': 'River', 'Reach Name': 'Reach'}, inplace=True)
        return gpd.GeoDataFrame(river_attrib, crs=self.crs)

    # -------------------------------------------------------------------------
    # 1D Cross Sections
    # -------------------------------------------------------------------------
    def get_cross_sections(self) -> gpd.GeoDataFrame:
        """Extract cross sections data."""
        with h5py.File(self.hdf_file, 'r') as hdf:
            xs_attrib = pd.DataFrame(hdf['Geometry']['Cross Sections']['Attributes'][:])
            xs_attrib = xs_attrib.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            xs_attrib['RS'] = xs_attrib['RS'].apply(safe_literal_eval)
            if 'Skew' not in xs_attrib.columns:
                xs_attrib['Skew'] = 0
            no_pts = [a[1] for a in hdf['Geometry']['Cross Sections']['Polyline Info'][:].tolist()]
            xs_attrib['no_pts'] = no_pts
            pts_df = pd.DataFrame(hdf['Geometry']['Cross Sections']['Polyline Points'][:])
            pts_df.columns = ['x', 'y']
            points_index = 0
            line_strings = []
            for _, row in xs_attrib.iterrows():
                num_points = row['no_pts']
                line_points = pts_df.iloc[points_index:points_index + num_points]
                line_strings.append(LineString(zip(line_points['x'], line_points['y'])))
                points_index += num_points
            xs_attrib['geometry'] = line_strings
        return gpd.GeoDataFrame(xs_attrib, geometry='geometry', crs=self.crs)

    # -------------------------------------------------------------------------
    # Structures
    # -------------------------------------------------------------------------
    def get_structures(self) -> gpd.GeoDataFrame:
        """Extract all structures data. This includes both 1D and 2D Structures"""
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                struc_df = pd.DataFrame(hdf['Geometry']['Structures']['Attributes'][:])
            except KeyError:
                return gpd.GeoDataFrame(geometry=[], crs=self.crs)
            struc_df = struc_df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            struc_df['RS'] = struc_df['RS'].apply(safe_literal_eval)
            struc_df['US RS'] = struc_df['US RS'].apply(safe_literal_eval)
            struc_df['DS RS'] = struc_df['DS RS'].apply(safe_literal_eval)
            struc_df[['start_idx', 'no_pts']] = hdf['Geometry']['Structures']['Centerline Info'][:, :2]
            struc_df['end_idx'] = struc_df['start_idx'] + struc_df['no_pts']
            pts_df = pd.DataFrame(hdf['Geometry']['Structures']['Centerline Points'][:])
            pts_df.columns = ['x', 'y']

            def get_linestring(data_row):
                line_points = pts_df.iloc[data_row['start_idx']:data_row['end_idx']]
                return LineString(zip(line_points['x'], line_points['y']))

            struc_df['geometry'] = struc_df.apply(get_linestring, axis=1)
            struc_df['Structure ID'] = struc_df.index
        return gpd.GeoDataFrame(struc_df, crs=self.crs)

    # -------------------------------------------------------------------------
    # 2D Structures
    # -------------------------------------------------------------------------
    def get_sa2d_connections(self) -> gpd.GeoDataFrame:
        """Extract SA/2D connections."""
        with h5py.File(self.hdf_file, 'r') as hdf:
            try:
                struc_df = pd.DataFrame(hdf['Geometry']['Structures']['Attributes'][:])
                struc_df = struc_df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                struc_df['RS'] = struc_df['RS'].apply(safe_literal_eval)
                struc_df['US RS'] = struc_df['US RS'].apply(safe_literal_eval)
                struc_df['DS RS'] = struc_df['DS RS'].apply(safe_literal_eval)
                struc_df[['start_idx', 'no_pts']] = hdf['Geometry']['Structures']['Centerline Info'][:, :2]
                struc_df['end_idx'] = struc_df['start_idx'] + struc_df['no_pts']
                pts_df = pd.DataFrame(hdf['Geometry']['Structures']['Centerline Points'][:])
                pts_df.columns = ['x', 'y']

                def get_linestring(data_row):
                    line_points = pts_df.iloc[data_row['start_idx']:data_row['end_idx']]
                    return LineString(zip(line_points['x'], line_points['y']))

                struc_df['geometry'] = struc_df.apply(get_linestring, axis=1)
                struc_df = struc_df[struc_df['Type'] == 'Connection'].copy()
                struc_df = struc_df.reset_index(drop=True)
            except KeyError:
                struc_df = pd.DataFrame([], columns=['geometry'])

        return gpd.GeoDataFrame(struc_df, geometry='geometry', crs=self.crs)
