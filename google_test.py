import networkx as nx
import googlemaps
import pickle
from typing import List, Tuple, Dict, Optional
import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from ambulance_network import fetch_nyc_graph,add_ems_stations_to_graph
from realtime_simulation import run_realtime_simulation



if __name__ == "__main__":
    # Example usage
    G,place = fetch_nyc_graph()
    G,ems_stations,station_nodes = add_ems_stations_to_graph(G,place)


    # 3. Run the real-time simulation
    print("[INFO] Starting the real-time simulation...")
    run_realtime_simulation\
        (
            G,
            station_nodes,
            ems_stations, # Pass the dictionary for potential use in summary/annotations
            total_minutes=10,
            calls_per_minute=7,
        )
    print("[INFO] Simulation finished.")
    print("[INFO] Main execution completed.")
    # print(bbox)
    