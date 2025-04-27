import matplotlib # Import matplotlib
matplotlib.use('TkAgg') # Set the backend *before* importing pyplot or calling plotting functions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Assuming ambulance_network.py contains the necessary functions
try:
    from ambulance_network import fetch_nyc_graph, sample_origins, compute_travel_times, optimize_dispatch, fetch_ems_stations, snap_ems_to_nodes
except ImportError:
    print("Warning: Could not import from ambulance_network. Using placeholder functions.")
    # Provide dummy functions if needed for testing structure
    import networkx as nx
    def fetch_nyc_graph(): return nx.DiGraph() # Return empty graph
    def sample_origins(G, k): return list(range(k)) # Return dummy origins
    def compute_travel_times(G, o, s): return { (orig,stat):100 for orig in o for stat in s }, {} # Dummy times/paths
    def optimize_dispatch(o, s, d, tt, total_amb): return {stat: total_amb//len(s) if s else 0 for stat in s}, {} # Dummy alloc/assign
    def fetch_ems_stations(G): return None # Dummy
    def snap_ems_to_nodes(G, gdf): return {} # Dummy

import osmnx as ox
import time
import networkx as nx # Ensure networkx is imported

# --- Globals for Animation ---
# Store artists managed by the update function (lines, text)
dynamic_plot_artists = []
# References to artists that are updated via set_data (scatter plots)
origins_scatter_artist = None
stations_scatter_artist = None # Plot stations once
# --- End Globals ---

def run_realtime_simulation(G, station_nodes, ems_station_nodes, total_minutes=30, calls_per_minute=3, total_ambulances=25):
    """
    Pre-calculates simulation states and then animates the results using FuncAnimation.

    Args:
        G (nx.MultiDiGraph): The road network graph.
        station_nodes (list): List of graph node IDs representing station locations.
        ems_station_nodes (dict): Dictionary mapping station names/IDs to graph node IDs.
                                  Used for annotations.
        total_minutes (int): Duration of the simulation in minutes.
        calls_per_minute (int): Average number of new calls per minute.
        total_ambulances (int): Total number of ambulances in the system.
    """
    global dynamic_plot_artists, origins_scatter_artist, stations_scatter_artist

    # --- Input Validation ---
    if G is None or not G.nodes:
        print("Error: Graph is missing or empty.")
        return
    if not station_nodes:
        print("Warning: station_nodes list is empty.")
        # Decide if simulation should proceed without stations
        # return # Or continue with warnings
    if not ems_station_nodes:
         print("Warning: ems_station_nodes dictionary is empty (needed for annotations).")
    # --- End Input Validation ---

    # --- Pre-calculation Phase ---
    print("[INFO] Pre-calculating simulation states...")
    simulation_history = [] # List to store state for each minute
    active_origins = []
    demand = {}
    total_calls_to_sample = total_minutes * calls_per_minute + 50 # Pre-sample calls
    origin_pool = sample_origins(G, k=total_calls_to_sample) if G.nodes else list(range(total_calls_to_sample))

    start_time_calc = time.time()
    for minute in range(total_minutes):
        print(f"  Calculating minute: {minute+1}/{total_minutes}")

        # 1. Generate new calls for this minute
        start_idx = minute * calls_per_minute
        end_idx = (minute + 1) * calls_per_minute
        new_calls = origin_pool[start_idx:end_idx]
        active_origins.extend(new_calls)
        for call in new_calls:
            demand[call] = 1 # Simple demand weight

        # 2. Compute travel times and optimize (if calls/stations exist)
        current_alloc = {}
        current_assignment = {}
        paths = {}
        if active_origins and station_nodes:
            tt, paths = compute_travel_times(G, active_origins, station_nodes)
            current_alloc, current_assignment = optimize_dispatch(
                active_origins,
                station_nodes,
                demand,
                tt,
                total_amb=total_ambulances
            )
        else:
            print("    No active calls or no stations for optimization.")

        # 3. Store state needed for visualization this minute
        minute_state = {
            "minute": minute + 1,
            "active_origins": list(active_origins), # Store copy
            "allocations": dict(current_alloc),     # Store copy
            "assignments": dict(current_assignment),# Store copy
            "paths": dict(paths)                    # Store copy
        }
        simulation_history.append(minute_state)

    print(f"[INFO] Pre-calculation finished in {time.time() - start_time_calc:.2f} seconds.")
    # --- End Pre-calculation Phase ---


    # --- Animation Phase ---
    print("[INFO] Starting animation...")
    fig, ax = ox.plot_graph(
        G, show=False, close=False, bgcolor='white', node_size=0,
        edge_color='#cccccc', edge_linewidth=0.5, figsize=(12, 12)
    )
    ax.set_title("Real-time Ambulance Dispatch Simulation")

    def init_animation():
        """Initializes the static parts of the animation plot."""
        global dynamic_plot_artists, origins_scatter_artist, stations_scatter_artist
        dynamic_plot_artists = [] # Clear artists list

        # Plot EMS stations once (static background)
        if station_nodes:
            try:
                station_xs = [G.nodes[nid]['x'] for nid in station_nodes]
                station_ys = [G.nodes[nid]['y'] for nid in station_nodes]
                stations_scatter_artist = ax.scatter(
                    station_xs, station_ys, c='red', s=80, marker='^',
                    label='EMS Station', zorder=5
                )
            except KeyError as e:
                 print(f"Error plotting stations: Node {e} not found in graph.")
                 stations_scatter_artist = None
        else:
            stations_scatter_artist = None

        # Initialize origins scatter plot (empty at first)
        origins_scatter_artist, = ax.plot(
            [], [], marker='o', color='blue', markersize=5,
            linestyle='', label='Incident', zorder=4
        )
        ax.legend(loc='upper left')
        # Return artists that FuncAnimation should manage
        return [origins_scatter_artist] + ([stations_scatter_artist] if stations_scatter_artist else [])

    def update_animation(frame_index):
        """Updates the plot for each frame using pre-calculated data."""
        global dynamic_plot_artists, origins_scatter_artist
        # frame_index corresponds to the minute (0 to total_minutes-1)
        if frame_index >= len(simulation_history):
            print(f"Warning: Frame index {frame_index} out of bounds for simulation history.")
            return [origins_scatter_artist] + dynamic_plot_artists + ([stations_scatter_artist] if stations_scatter_artist else [])

        # Retrieve pre-calculated state for this frame/minute
        current_state = simulation_history[frame_index]
        minute = current_state["minute"]
        current_active_origins = current_state["active_origins"]
        current_alloc = current_state["allocations"]
        current_assignment = current_state["assignments"]
        current_paths = current_state["paths"]

        print(f"  Displaying minute: {minute}/{total_minutes}")

        # --- Update Visualization ---
        # Remove old dynamic artists (lines and text from previous frame)
        for artist in dynamic_plot_artists:
            artist.remove()
        dynamic_plot_artists.clear()

        # Update origins scatter plot data
        if current_active_origins:
            try:
                orig_xs = [G.nodes[o]['x'] for o in current_active_origins]
                orig_ys = [G.nodes[o]['y'] for o in current_active_origins]
                origins_scatter_artist.set_data(orig_xs, orig_ys)
            except KeyError as e:
                 print(f"Error updating origins plot: Node {e} not found.")
                 origins_scatter_artist.set_data([], [])
        else:
            origins_scatter_artist.set_data([], [])

        # Plot assignment paths for this frame
        for (o, s_node), val in current_assignment.items():
            if val:
                path = current_paths.get((o, s_node))
                if path:
                    try:
                        xs = [G.nodes[n]['x'] for n in path]
                        ys = [G.nodes[n]['y'] for n in path]
                        line, = ax.plot(xs, ys, linestyle='--', linewidth=0.8, alpha=0.7, color='green', zorder=3)
                        dynamic_plot_artists.append(line)
                    except KeyError as e:
                         print(f"Error plotting path for ({o}, {s_node}): Node {e} not found.")

        # Plot ambulance allocation annotations for this frame
        # Use ems_station_nodes (the dict) for annotations
        for name_or_idx, s_node in ems_station_nodes.items():
            count = current_alloc.get(s_node, 0)
            if count > 0:
                try:
                    txt = ax.text(
                        G.nodes[s_node]['x'], G.nodes[s_node]['y'] + 15,
                        f"A={count}", ha='center', va='bottom', fontsize=8,
                        color='darkred', zorder=6
                    )
                    dynamic_plot_artists.append(txt)
                except KeyError:
                     print(f"[WARN] Node {s_node} (for '{name_or_idx}') not found in graph for annotation.")

        ax.set_title(f"Real-time Simulation (Minute: {minute})")
        # --- End Update Visualization ---

        # Return all artists for this frame
        return [origins_scatter_artist] + dynamic_plot_artists + ([stations_scatter_artist] if stations_scatter_artist else [])

    # Create and run the animation using pre-calculated data
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=total_minutes, # Number of frames = number of minutes calculated
        init_func=init_animation,
        blit=False,
        repeat=False,
        interval=500 # Milliseconds between frames
    )

    plt.show() # Display the animation window
    print("[INFO] Animation finished.")
    # --- End Animation Phase ---

