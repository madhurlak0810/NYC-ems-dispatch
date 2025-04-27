from ambulance_network import fetch_nyc_graph, sample_origins, compute_travel_times, optimize_dispatch, fetch_ems_stations,snap_ems_to_nodes
from realtime_simulation import run_realtime_simulation


if __name__ == "__main__":
    # Fetch the NYC graph
    G = fetch_nyc_graph()

    ems_stations = fetch_ems_stations(G)
    
    ems_stations_nodes= snap_ems_to_nodes(G, ems_stations)
    station_nodes=list(ems_stations_nodes.values())
    
    

    # Run the real-time simulation
    run_realtime_simulation(G, station_nodes, ems_stations_nodes, total_minutes=20, calls_per_minute=3,total_ambulances=50)