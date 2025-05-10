import osmnx as ox
import networkx as nx
import random
import pulp
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
from dotenv import load_dotenv
import googlemaps
from typing import List, Tuple, Dict, Optional
import os
from pyproj import Transformer
from math import radians, sin, cos, sqrt, atan2

load_dotenv()  # Load environment variables from .env file
API_KEY=os.getenv("API_KEY")  # Make sure to replace this with your actual API key
gmaps = googlemaps.Client(key=API_KEY)

import osmnx as ox
import networkx as nx

# Initialize the transformer from EPSG:2263 to WGS84
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
def fetch_nyc_graph(place_name="Manhattan, New York City, New York, USA") -> tuple[nx.MultiDiGraph, dict]:
    """
    Fetches and processes a road network graph for all of New York City 
    and returns its bounding box.
    
    Returns:
        - G (nx.MultiDiGraph): The processed road network graph.
        - bounding_box (dict): The bounding box of the graph in the format:
            {
                'min_x': float,
                'min_y': float,
                'max_x': float,
                'max_y': float
            }
    """
    G = ox.graph_from_place(place_name, network_type="drive")
    
    # 2) Project to a local metric CRS (NYLongIsland / EPSG:2263)
    G = ox.project_graph(G, to_crs="EPSG:4326")
    
    # 3) Assign default travel time to each edge
    default_speed_fps = 10
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 0)             # in feet
        speed  = data.get("maxspeed_fps", default_speed_fps)
        data["travel_time"] = (length / speed) if speed > 0 else float("inf")
    
    return G,place_name


def sample_origins(city_graph, k=5):
    nodes = list(city_graph.nodes())
    random.seed(42)
    return random.sample(nodes, k)

def compute_travel_times(city_graph, origins, stations):
    total_time = {}
    paths      = {}
    time_progress = {}

    for origin in origins:
        # Compute shortest path lengths and paths from origin to all nodes
        lengths, pathdict = nx.single_source_dijkstra(city_graph, origin, weight='travel_time')

        for station in stations:
            # Get total travel time from origin to station (or inf if unreachable)
            total_time[(origin, station)] = lengths.get(station, float('inf'))

            if station in pathdict:
                path = pathdict[station]
                paths[(origin, station)] = path

                # Compute cumulative travel times along the path
                cumulative = [0]
                for i in range(1, len(path)):
                    edge_data = city_graph[path[i - 1]][path[i]]

                    # For MultiDiGraph, pick the shortest edge if multiple
                    min_time = min(d.get('travel_time', float('inf')) for d in edge_data.values())

                    cumulative.append(cumulative[-1] + min_time)

                time_progress[(origin, station)] = cumulative

    return total_time, paths, time_progress

def optimize_dispatch(origins, stations, demand, total_time, idle_ambs, max_time=600000):
    print("--------------------------------------------------")
    print("[INFO] Idle ambulances by station:")
    for station in stations:
        print(f"{station}: {idle_ambs[station]}")
    print("--------------------------------------------------")

    # Step 1: Generate valid origin-station pairs within max time
    valid = [
        (origin, station) 
        for origin in origins 
        for station in stations 
        if total_time.get((origin, station), float('inf')) <= max_time
    ]

    if not valid:
        print("[ERROR] No valid origin-station pairs found. Check max_time or total_time.")
        return None, None

    # Step 2: Problem definition
    prob = pulp.LpProblem('NYC_Ambulance_Dispatch', pulp.LpMinimize)

    # Step 3: Variable definitions
    assign = pulp.LpVariable.dicts('assign', valid, cat='Binary')
    amb_count = pulp.LpVariable.dicts('ambulance_count', stations, lowBound=0, upBound=max(idle_ambs.values()), cat='Integer')

    # Step 4: Objective function - Minimize weighted travel time
    prob += pulp.lpSum(demand[origin] * total_time[(origin, station)] * assign[(origin, station)] for (origin, station) in valid)

    # Step 5: Constraints
    ## Total ambulances dispatched should be at least 50% of demand
    prob += pulp.lpSum(amb_count[station] for station in stations) >= len(demand) * 0.5

    ## Each origin must be served by exactly 1 ambulance
    for origin in origins:
        vs = [s for (oo, s) in valid if oo == origin]
        if vs:
            prob += pulp.lpSum(assign[(origin, station)] for station in vs) == 1

    ## Ambulances dispatched from each station must not exceed idle capacity
    for station in stations:
        prob += amb_count[station] <= idle_ambs[station]

    ## Sum of assignments to a station must match the ambulance count from that station
    for station in stations:
        prob += pulp.lpSum(assign[(origin, station)] for (origin, s) in valid if s == station) == amb_count[station]

    # Step 6: Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # Step 7: Solver Status
    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"[ERROR] Optimization failed. Solver Status: {pulp.LpStatus[prob.status]}")
        return None, None

    print("[INFO] Optimization successful!")
    print("==================================================")
    print("Station | Allocated | Remaining")
    print("--------------------------------------------------")

    # Step 8: Output allocations and assignments
    current_alloc = {}
    current_assignment = {}

    for station in stations:
        allocated = int(amb_count[station].varValue)
        remaining = idle_ambs[station] - allocated
        current_alloc[station] = allocated
        print(f"{station} | {allocated} | {remaining}")

    print("--------------------------------------------------")
    print("[INFO] Assignment Details:")
    for (origin, station) in valid:
        assigned = int(assign[(origin, station)].varValue)
        if assigned > 0:
            print(f"{origin} -> {station} | Assigned: {assigned}")
        current_assignment[(origin, station)] = assigned

    print("==================================================")

    return current_alloc, current_assignment


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points 
    on the Earth specified in decimal degrees using the Haversine formula.
    """
    R = 6371000  # Radius of Earth in meters

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def fetch_ems_stations(place_name: str) -> List[Tuple[float, float, str]]:
    """
    Fetches the geographic coordinates and names of EMS stations near a given place
    using the Google Maps Places API.

    Args:
        place_name (str): The name of the place to search for EMS stations.

    Returns:
        List[Tuple[float, float, str]]: A list of (latitude, longitude, name) tuples
                                        representing the EMS stations.
                                        Returns an empty list on error.
    """
    try:
        # Search for EMS stations with a text query
        response = gmaps.places(
            query=f"EMS Station in {place_name}"
        )
        
        print("[DEBUG] Google Maps Response:", response)  # DEBUG
        
        if not response or 'results' not in response:
            print(f"Error: Invalid response from Places API for EMS stations in {place_name}")
            return []

        ems_stations = []
        for result in response.get('results', []):
            station_lat = result['geometry']['location']['lat']
            station_lng = result['geometry']['location']['lng']
            station_name = result.get('name', 'Unnamed Station')
            ems_stations.append((station_lat, station_lng, station_name))

        print(f"[INFO] Found {len(ems_stations)} EMS stations.")
        return ems_stations

    except Exception as e:
        print(f"Error fetching EMS stations: {e}")
        return []


def add_ems_stations_to_graph(G: nx.MultiDiGraph, place_name) -> tuple[nx.MultiDiGraph, list, list]:
    """
    Fetches EMS stations using Google Maps and adds them as nodes to the graph.

    Args:
        G (nx.MultiDiGraph): The graph to add the EMS stations to.
        place_name (str): The name of the place to search for EMS stations.

    Returns:
        tuple: A tuple containing:
            - nx.MultiDiGraph: The updated graph with the EMS stations added as nodes.
            - list: List of EMS stations with their coordinates.
            - list: List of added EMS station node IDs.
    """
    station_nodes = []
    ems_stations = fetch_ems_stations(place_name)
    print(f"[INFO] Found {len(ems_stations)} EMS stations.")

    for station_lat, station_lng, station_name in ems_stations:
        # Find the nearest node in the graph
        nearest_node = None
        min_distance = float('inf')
        for node_id in G.nodes():
            node_lat = G.nodes[node_id]['y']
            node_lng = G.nodes[node_id]['x']
            distance = (node_lat - station_lat) ** 2 + (node_lng - station_lng) ** 2
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        print(f"[DEBUG] Nearest node for {station_name}: {nearest_node} with distance {min_distance:.4f}")
        # Only add if a nearest node is found
        if nearest_node is not None and min_distance != float('inf'):
            # Add the EMS station as a node, with a unique identifier
            ems_node_id = f"ems_{station_name.replace(' ', '_')}_{nearest_node}"  # Ensure unique ID
            if ems_node_id not in G.nodes:
                G.add_node(ems_node_id, y=station_lat, x=station_lng, name=station_name, type='ems_station')
                # Create edges connecting the EMS station to the nearest node
                G.add_edge(ems_node_id, nearest_node, weight=min_distance**0.5, type='to_graph')
                G.add_edge(nearest_node, ems_node_id, weight=min_distance**0.5, type='to_graph')
                print(f"[INFO] Added EMS station '{station_name}' at ({station_lat:.4f}, {station_lng:.4f}) and connected it to node {nearest_node}.")
                station_nodes.append(ems_node_id)
            else:
                print(f"[INFO] EMS station with id '{ems_node_id}' already exists.")
        else:
            print(f"[WARN] Could not find a nearest node for EMS station '{station_name}' at ({station_lat:.4f}, {station_lng:.4f}). Skipping addition.")

    return G, ems_stations, station_nodes