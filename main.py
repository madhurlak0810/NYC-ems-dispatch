import sys # Import sys for exiting
try:
    from ambulance_network import fetch_nyc_graph, sample_origins, compute_travel_times, optimize_dispatch, fetch_ems_stations,snap_ems_to_nodes
    from realtime_simulation import run_realtime_simulation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure ambulance_network.py and realtime_simulation.py are in the same directory or accessible.")
    sys.exit(1) # Exit if imports fail

if __name__ == "__main__":
    print("[INFO] Starting main execution...")

    # 1. Fetch the NYC graph
    print("[INFO] Fetching graph data...")
    try:
        G = fetch_nyc_graph()
        if G is None or not G.nodes:
             print("[ERROR] Failed to fetch a valid graph. Exiting.")
             sys.exit(1)
        print(f"[INFO] Graph fetched successfully ({len(G.nodes)} nodes, {len(G.edges)} edges).")
    except Exception as e:
        print(f"[ERROR] An exception occurred while fetching the graph: {e}")
        sys.exit(1)

    # 2. Fetch and process EMS stations
    print("[INFO] Fetching and processing EMS station data...")
    ems_stations_nodes = {}
    station_nodes = []
    try:
        ems_stations_gdf = fetch_ems_stations(G)
        if ems_stations_gdf is not None and not ems_stations_gdf.empty:
            print(f"[INFO] Fetched {len(ems_stations_gdf)} potential EMS station locations.")
            ems_stations_nodes = snap_ems_to_nodes(G, ems_stations_gdf)
            if ems_stations_nodes:
                station_nodes = list(ems_stations_nodes.values())
                print(f"[INFO] Successfully snapped {len(station_nodes)} stations to graph nodes.")
            else:
                print("[WARNING] Could not snap any EMS stations to the graph. Simulation might not be meaningful.")
        else:
            print("[WARNING] No EMS station data found or fetched.")
    except Exception as e:
        print(f"[ERROR] An exception occurred while processing EMS stations: {e}")
        # Decide if you want to exit or continue without stations
        # Exiting for now as stations are crucial for the simulation as written
        print("[INFO] Exiting due to error in station processing.")
        sys.exit(1)

    # Check if we have stations before running simulation
    if not station_nodes:
        print("[ERROR] No station nodes available to run the simulation. Exiting.")
        sys.exit(1)

    # 3. Run the real-time simulation
    print("[INFO] Starting the real-time simulation...")
    try:
        run_realtime_simulation(
            G,
            station_nodes,
            ems_stations_nodes, # Pass the dictionary for potential use in summary/annotations
            total_minutes=20,
            calls_per_minute=3,
            total_ambulances=50
        )
        print("[INFO] Simulation finished.")
    except Exception as e:
        print(f"[ERROR] An exception occurred during the simulation run: {e}")
        sys.exit(1)

    print("[INFO] Main execution completed.")