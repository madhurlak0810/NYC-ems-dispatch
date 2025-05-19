import osmnx as ox
import networkx as nx
import random
import pulp
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box

def fetch_nyc_graph() -> nx.MultiDiGraph:
    """
    Fetches and processes a road network graph for all of New York City.
    """
    # 1) Grab the full NYC driving network by name
    place_name = "New York City, New York, USA"
    G = ox.graph_from_place(place_name, network_type="drive")
    
    # 2) Project to a local metric CRS (NYLongIsland / EPSG:2263)
    G = ox.project_graph(G, to_crs="EPSG:2263")
    
    # 3) Compute a default travel_time attribute on every edge (10 m/s)
    default_speed_fps = 10
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 0)             # in meters
        speed  = data.get("maxspeed_fps", default_speed_fps)
        data["travel_time"] = (length / speed) if speed > 0 else float("inf")
    
    return G

def sample_origins(G, k=50):
    nodes = list(G.nodes())
    random.seed(42)
    return random.sample(nodes, k)

def compute_travel_times(G, origins, stations):
    tt = {}
    paths = {}
    for o in origins:
        lengths, pathdict = nx.single_source_dijkstra(G, o, weight='travel_time')
        for s in stations:
            tt[(o,s)] = lengths.get(s, float('inf'))
            if s in pathdict:
                paths[(o,s)] = pathdict[s]
    return tt, paths

def optimize_dispatch(origins, stations, demand, tt, total_amb=5, max_t=600):
    valid = [(o,s) for o in origins for s in stations if tt[(o,s)] <= max_t]
    prob = pulp.LpProblem('NYC_Ambulance', pulp.LpMinimize)
    amb_count = pulp.LpVariable.dicts('amb', stations, lowBound=0, upBound=total_amb, cat='Integer')
    assign = pulp.LpVariable.dicts('asgn', valid, cat='Binary')
    # objective
    prob += pulp.lpSum(demand[o] * tt[o,s] * assign[(o,s)] for (o,s) in valid)
    # total ambulances
    prob += pulp.lpSum(amb_count[s] for s in stations) == total_amb
    # each origin must be served by ≥1 ambulance
    for o in origins:
        vs = [s for (oo,s) in valid if oo==o]
        if vs:
            prob += pulp.lpSum(assign[(o,s)] for s in vs) >= 1
    # can't assign from a station with no ambulance
    for (o,s) in valid:
        prob += assign[(o,s)] <= amb_count[s]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {s:int(amb_count[s].varValue) for s in stations}, {(o,s):int(assign[(o,s)].varValue) for (o,s) in valid}


# --- Revised fetch_ems_stations ---
def fetch_ems_stations(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Queries OSM for EMS (ambulance) stations within the bounding box of G
    and returns a GeoDataFrame of their geometries.

    Parameters:
        G (nx.MultiDiGraph): A projected or unprojected graph with CRS and optional 'bbox'.
                             If G.graph['bbox'] is missing, it will be computed from node extents.

    Returns:
        gpd.GeoDataFrame: EMS station features (Point or Polygon) projected to G's CRS.
                          Returns an empty GeoDataFrame if none are found or on error.
    """
    print("[INFO] Fetching EMS station locations from OSM...")
    # Ensure CRS is defined
    if 'crs' not in G.graph:
        raise ValueError("Graph G must have a 'crs' attribute defined in G.graph['crs']")

    # Get or compute bbox in lat/lon (EPSG:4326)
    try:
        if 'bbox' in G.graph:
             # Assuming stored bbox is (west, south, east, north) in EPSG:4326
             west, south, east, north = G.graph['bbox']
             print(f"[INFO] Using stored bbox (EPSG:4326): W={west}, S={south}, E={east}, N={north}")
        else:
            print("[WARN] Graph 'bbox' attribute missing. Calculating from node bounds.")
            # Ensure nodes are in Lat/Lon for bbox calculation if graph isn't already
            if G.graph['crs'].to_epsg() != 4326:
                 gdf_nodes = ox.graph_to_gdfs(G, edges=False).to_crs(epsg=4326)
            else:
                 gdf_nodes = ox.graph_to_gdfs(G, edges=False)
            west, south, east, north = gdf_nodes.total_bounds  # lon/lat
            G.graph['bbox'] = (west, south, east, north) # Store for future use
            print(f"[INFO] Calculated bbox (EPSG:4326): W={west}, S={south}, E={east}, N={north}")

        # Define EMS tag
        tags = {"emergency": "ambulance_station"}

        # Query OSM for ambulance_station features using the correct bbox order for geometries_from_bbox
        gdf = ox.features_from_bbox(G.graph['bbox'],tags)
        print(f"[INFO] Found {len(gdf)} raw features matching tags.")

        # Keep only points/polygons/multipolygons
        gdf = gdf[gdf.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
        print(f"[INFO] {len(gdf)} Point/Polygon/MultiPolygon features remaining.")

        if gdf.empty:
            print("[WARN] No suitable EMS station features found.")
            return gdf # Return empty GeoDataFrame

        # Project to graph CRS
        print(f"[INFO] Projecting features to graph CRS: {G.graph['crs']}")
        gdf = gdf.to_crs(G.graph['crs'])

        return gdf

    except Exception as e:
        print(f"[ERROR] Failed to fetch or process EMS stations: {e}")
        # Return an empty GeoDataFrame with the expected CRS to avoid downstream errors
        return gpd.GeoDataFrame(geometry=[], crs=G.graph['crs'])


# --- Revised snap_ems_to_nodes ---
# Changed return type hint and added name handling
def snap_ems_to_nodes(G: nx.MultiDiGraph, gdf_ems: gpd.GeoDataFrame) -> dict[str | int, int]:
    """
    Snaps EMS station geometries from a GeoDataFrame to the nearest nodes in G.

    Parameters:
        G (nx.MultiDiGraph): Projected graph with node coordinates.
        gdf_ems (gpd.GeoDataFrame): EMS station features in G's CRS.

    Returns:
        dict[str | int, int]: Mapping of station name (if available, otherwise OSM index)
                              to the nearest graph node ID.
    """
    print(f"[INFO] Snapping {len(gdf_ems)} EMS stations to graph nodes...")
    station_nodes = {}
    processed_names = set() # To handle potential duplicate names

    if gdf_ems.empty:
        print("[WARN] GeoDataFrame for snapping is empty.")
        return {}

    for idx, row in gdf_ems.iterrows():
        # Use station name if available and valid, otherwise use OSM index as key
        station_key = row.get('name')
        if not isinstance(station_key, str) or station_key.strip() == "":
            station_key = idx # Fallback to OSM index (which is unique in the gdf)
        else:
            # Handle duplicate names by appending index
            original_key = station_key
            counter = 1
            while station_key in processed_names:
                counter += 1
                station_key = f"{original_key}_{counter}"
            processed_names.add(station_key)


        # Choose centroid for non-points
        geom = row.geometry.centroid if row.geometry.geom_type != 'Point' else row.geometry
        x, y = geom.x, geom.y

        try:
            # Find the nearest node
            node_id = ox.nearest_nodes(G, X=x, Y=y)
            station_nodes[station_key] = node_id
            # print(f"  Snapped '{station_key}' to node {node_id}") # Optional debug print
        except Exception as e:
            # Log error instead of silently skipping
            print(f"[WARN] Could not snap station '{station_key}' (OSM index {idx}) at ({x:.4f}, {y:.4f}): {e}")
            continue # Skip this station

    print(f"[INFO] Successfully snapped {len(station_nodes)} stations.")
    return station_nodes