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
# We recreate these each frame in this version
dynamic_plot_artists = []
# References to artists that are updated via set_data (scatter plots)
origins_scatter_artist = None
stations_scatter_artist = None # Plot stations once
# --- End Globals ---

# Removed plot_dynamic_elements function, logic moved to update()

def run_realtime_simulation(G, station_nodes, ems_station_nodes, total_minutes=30, calls_per_minute=3, total_ambulances=25):
    """
    Runs and animates a real-time ambulance dispatch simulation using FuncAnimation.

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

    if G is None or not G.nodes:
        print("Error: Graph is missing or empty.")
        return
    if not station_nodes:
        print("Warning: station_nodes list is empty.")
    if not ems_station_nodes:
         print("Warning: ems_station_nodes dictionary is empty (needed for annotations).")


    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor='white',
        node_size=0,
        edge_color='#cccccc',
        edge_linewidth=0.5,
        figsize=(12, 12)
    )
    ax.set_title("Real-time Ambulance Dispatch Simulation")

    # --- Simulation State Setup ---
    total_calls_to_sample = total_minutes * calls_per_minute + 50 # Pre-sample calls
    # Ensure sample_origins can handle potentially empty graph for placeholder
    origin_pool = sample_origins(G, k=total_calls_to_sample) if G.nodes else list(range(total_calls_to_sample))

    # Store simulation state in a dictionary for clarity within update function
    sim_state = {
        "origin_pool": origin_pool,
        "active_origins": [],
        "demand": {},
        "calls_per_minute": calls_per_minute,
        "station_nodes": station_nodes, # The list of node IDs
        "ems_station_nodes": ems_station_nodes, # The dict for annotations
        "total_ambulances": total_ambulances,
        "G": G,
        "current_assignment": {}, # Store current assignment paths
        "current_alloc": {}      # Store current allocation counts
    }
    # --- End Simulation State Setup ---

    # --- Animation Functions ---
    def init_animation():
        """Initializes the animation plot."""
        global dynamic_plot_artists, origins_scatter_artist, stations_scatter_artist
        dynamic_plot_artists = [] # Clear artists list

        # Plot EMS stations once (static background) using station_nodes list
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
                 stations_scatter_artist = None # Failed to plot
        else:
            stations_scatter_artist = None # No stations to plot

        # Initialize origins scatter plot (empty at first)
        # Using plot returns a list, so we unpack with comma
        origins_scatter_artist, = ax.plot(
            [], [], marker='o', color='blue', markersize=5,
            linestyle='', label='Incident', zorder=4
        )

        # Add legend
        ax.legend(loc='upper left')

        # Return artists that FuncAnimation should manage (only scatter initially)
        # Other artists (lines, text) will be added/removed in update
        return [origins_scatter_artist] + ([stations_scatter_artist] if stations_scatter_artist else [])


    def update_animation(minute):
        """Updates the animation for each frame (minute)."""
        global dynamic_plot_artists, origins_scatter_artist
        print(f"Minute: {minute+1}/{total_minutes}")

        # --- Simulation Step ---
        start_idx = minute * sim_state["calls_per_minute"]
        end_idx = (minute + 1) * sim_state["calls_per_minute"]
        new_calls = sim_state["origin_pool"][start_idx:end_idx]

        sim_state["active_origins"].extend(new_calls)
        for call in new_calls:
            sim_state["demand"][call] = 1 # Simple demand weight

        # Only run optimization if there are active origins and stations
        if not sim_state["active_origins"] or not sim_state["station_nodes"]:
             print("  No active calls or no stations.")
             sim_state["current_alloc"] = {}
             sim_state["current_assignment"] = {}
             paths = {} # No paths needed
        else:
            print(f"  Active calls: {len(sim_state['active_origins'])}")
            # Use the list of station_nodes for computation
            tt, paths = compute_travel_times(
                sim_state["G"],
                sim_state["active_origins"],
                sim_state["station_nodes"] # Pass the list
            )
            # Use the list of station_nodes for optimization
            sim_state["current_alloc"], sim_state["current_assignment"] = optimize_dispatch(
                sim_state["active_origins"],
                sim_state["station_nodes"], # Pass the list
                sim_state["demand"],
                tt,
                total_amb=sim_state["total_ambulances"]
            )
        # --- End Simulation Step ---

        # --- Update Visualization ---
        # Remove old dynamic artists (lines and text from previous frame)
        for artist in dynamic_plot_artists:
            artist.remove()
        dynamic_plot_artists.clear() # Clear the list

        # Update origins scatter plot data
        if sim_state["active_origins"]:
            try:
                orig_xs = [sim_state["G"].nodes[o]['x'] for o in sim_state["active_origins"]]
                orig_ys = [sim_state["G"].nodes[o]['y'] for o in sim_state["active_origins"]]
                origins_scatter_artist.set_data(orig_xs, orig_ys)
            except KeyError as e:
                 print(f"Error updating origins plot: Node {e} not found.")
                 origins_scatter_artist.set_data([], []) # Clear plot on error
        else:
            origins_scatter_artist.set_data([], [])

        # Plot new assignment paths and store artists
        for (o, s_node), val in sim_state["current_assignment"].items():
            if val:
                path = paths.get((o, s_node))
                if path:
                    try:
                        xs = [sim_state["G"].nodes[n]['x'] for n in path]
                        ys = [sim_state["G"].nodes[n]['y'] for n in path]
                        line, = ax.plot(xs, ys, linestyle='--', linewidth=0.8, alpha=0.7, color='green', zorder=3)
                        dynamic_plot_artists.append(line) # Add new line artist
                    except KeyError as e:
                         print(f"Error plotting path for ({o}, {s_node}): Node {e} not found.")


        # Plot new ambulance allocation annotations and store artists
        # Use ems_station_nodes (the dict) for annotations
        for name_or_idx, s_node in sim_state["ems_station_nodes"].items():
            count = sim_state["current_alloc"].get(s_node, 0)
            if count > 0:
                try:
                    txt = ax.text(
                        sim_state["G"].nodes[s_node]['x'],
                        sim_state["G"].nodes[s_node]['y'] + 15, # Offset slightly above station marker
                        f"A={count}",
                        ha='center', va='bottom', fontsize=8, color='darkred', zorder=6
                    )
                    dynamic_plot_artists.append(txt) # Add new text artist
                except KeyError:
                     print(f"[WARN] Node {s_node} (for '{name_or_idx}') not found in graph for annotation.")

        ax.set_title(f"Real-time Simulation (Minute: {minute+1})")
        # --- End Update Visualization ---

        # Return all artists that need to be managed/drawn for this frame
        return [origins_scatter_artist] + dynamic_plot_artists + ([stations_scatter_artist] if stations_scatter_artist else [])
    # --- End Animation Functions ---

    # Create and run the animation
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=total_minutes,
        init_func=init_animation,
        blit=False,  # Blitting often problematic with changing artist counts/types
        repeat=False,
        interval=500  # Milliseconds between frames (e.g., 500ms = 2 fps)
    )

    plt.show() # Display the animation window
