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
from collections import defaultdict # Import defaultdict

# --- Globals for Animation ---
# These are reset/managed within run_realtime_simulation or its inner functions
served_origins = set()
served_scatter_artist = None
dynamic_plot_artists = []
origins_scatter_artist = None
stations_scatter_artist = None
moving_ambulances = []
ambulance_scatter_artist = None # Add artist for moving ambulances
title_artist = None # Artist for the title text
# --- End Globals ---

def run_realtime_simulation(G, station_nodes, ems_station_nodes, total_minutes=30, calls_per_minute=3, total_ambulances=25):
    """
    Pre-calculates simulation states, animates the results, and prints/saves a summary.

    Args:
        G (nx.MultiDiGraph): The road network graph.
        station_nodes (list): List of graph node IDs representing station locations.
        ems_station_nodes (dict): Dictionary mapping station names/IDs to graph node IDs.
                                  Used for annotations and summary.
        total_minutes (int): Duration of the simulation in minutes.
        calls_per_minute (int): Average number of new calls per minute.
        total_ambulances (int): Total number of ambulances in the system.
    """
    # Reset globals specific to animation state for this run
    global dynamic_plot_artists, origins_scatter_artist, stations_scatter_artist, served_scatter_artist, moving_ambulances, served_origins, ambulance_scatter_artist, title_artist
    dynamic_plot_artists = []
    origins_scatter_artist = None
    stations_scatter_artist = None
    served_scatter_artist = None
    moving_ambulances = []
    served_origins = set() # Reset served origins for animation
    ambulance_scatter_artist = None
    title_artist = None

    total_seconds = total_minutes * 60

    # --- Input Validation ---
    if G is None or not G.nodes:
        print("Error: Graph is missing or empty.")
        return
    if not station_nodes:
        print("Warning: station_nodes list is empty.")
    # Allow ems_station_nodes to be empty, summary will adapt
    # --- End Input Validation ---

    # --- Pre-calculation Phase ---
    print("[INFO] Pre-calculating simulation states...")
    simulation_history = [] # List to store state for each minute
    all_active_origins_ever = set() # Keep track of all origins that appeared
    precalc_served_origins = set() # Track origins served during pre-calculation
    demand = {}
    # Pre-sample enough potential origins
    total_calls_to_sample = total_minutes * calls_per_minute + 100 # Add buffer
    origin_pool = sample_origins(G, k=total_calls_to_sample) if G.nodes else list(range(total_calls_to_sample))

    # Initialize structures for summary statistics
    response_times_per_station = defaultdict(list)
    final_alloc = {} # Store allocation from the last minute

    start_time_calc = time.time()
    for minute in range(total_minutes):
        print(f"  Calculating minute: {minute+1}/{total_minutes}")

        # 1. Generate new calls for this minute
        start_idx = minute * calls_per_minute
        end_idx = (minute + 1) * calls_per_minute
        new_calls = origin_pool[start_idx:end_idx]
        all_active_origins_ever.update(new_calls)
        for call in new_calls:
            demand[call] = 1 # Simple demand weight

        # 2. Identify currently unserved origins for optimization
        currently_unserved_origins = list(all_active_origins_ever - precalc_served_origins)

        # 3. Compute travel times and optimize (if unserved calls/stations exist)
        current_alloc = {}
        current_assignment = {}
        paths = {}
        tt = {} # Initialize tt for the case where optimization doesn't run
        if currently_unserved_origins and station_nodes:
            # Compute times only for unserved origins to available stations
            tt, paths = compute_travel_times(G, currently_unserved_origins, station_nodes)

            # Filter demand to only include unserved origins
            current_demand = {origin: demand[origin] for origin in currently_unserved_origins if origin in demand}

            if current_demand: # Ensure there's demand to optimize for
                current_alloc, current_assignment = optimize_dispatch(
                    currently_unserved_origins,
                    station_nodes,
                    current_demand,
                    tt,
                    total_amb=total_ambulances
                )

                # Accumulate response times and mark as served for NEXT iteration's calculation
                for (origin, station_node), assigned in current_assignment.items():
                    if assigned == 1: # If this station is assigned to this origin
                        travel_time = tt.get((origin, station_node))
                        if travel_time is not None:
                            # Store response time for summary
                            response_times_per_station[station_node].append(travel_time)
                            # Mark origin as served for the purpose of the *next* minute's optimization
                            precalc_served_origins.add(origin)
                        else:
                            print(f"Warning: Missing travel time for assigned pair ({origin}, {station_node}) during pre-calculation.")
            else:
                 print("    No demand among unserved origins for optimization.")

        else:
            if not currently_unserved_origins:
                 print("    No unserved calls for optimization.")
            if not station_nodes:
                 print("    No stations for optimization.")


        # Store final allocation from the last minute
        if minute == total_minutes - 1:
            final_alloc = dict(current_alloc)

        # 4. Store state needed for visualization this minute
        # Note: active_origins for visualization should include ALL origins active up to this minute
        minute_state = {
            "minute": minute + 1,
            "active_origins": list(all_active_origins_ever), # All origins seen so far for plotting
            "allocations": dict(current_alloc),     # Allocation decided this minute
            "assignments": dict(current_assignment),# Assignments decided this minute
            "paths": dict(paths),                   # Paths calculated this minute (for unserved)
            "travel_times": dict(tt)                # Travel times calculated this minute (for unserved)
        }
        simulation_history.append(minute_state)

    print(f"[INFO] Pre-calculation finished in {time.time() - start_time_calc:.2f} seconds.")
    # --- End Pre-calculation Phase ---

    # Pause for user input before proceeding to animation
    input("Pre‑calculation complete. Press Enter to start animation…")


    # --- Animation Phase ---
    print("[INFO] Starting animation...")
    fig, ax = ox.plot_graph(
        G, show=False, close=False, bgcolor='white', node_size=0,
        edge_color='#cccccc', edge_linewidth=0.5, figsize=(12, 12)
    )

    def init_animation():
        """Initializes the static parts of the animation plot."""
        global dynamic_plot_artists, origins_scatter_artist, stations_scatter_artist, served_scatter_artist, ambulance_scatter_artist, title_artist
        dynamic_plot_artists = [] # Clear artists list

        # Plot EMS stations once (static background)
        if station_nodes:
            try:
                station_xs = [G.nodes[nid]['x'] for nid in station_nodes]
                station_ys = [G.nodes[nid]['y'] for nid in station_nodes]
                # Use plot for legend consistency, scatter doesn't always play nice with FuncAnimation blitting
                stations_scatter_artist, = ax.plot(
                    station_xs, station_ys, marker='^', color='red', markersize=10, # Increased size slightly
                    linestyle='', label='EMS Station', zorder=5
                )
            except KeyError as e:
                 print(f"Error plotting stations: Node {e} not found in graph.")
                 stations_scatter_artist = None
        else:
            stations_scatter_artist = None

        # Initialize origins scatter plot (empty at first)
        origins_scatter_artist, = ax.plot(
            [], [], marker='o', color='blue', markersize=5,
            linestyle='', label='Incident (unserved)', zorder=4
        )

        served_scatter_artist, = ax.plot(
            [], [], marker='o', color='green', markersize=5,
            linestyle='', label='Incident (served)', zorder=4
        )

        # Initialize moving ambulance scatter plot (empty at first)
        ambulance_scatter_artist, = ax.plot(
            [], [], marker='>', color='darkred', markersize=7, # Slightly larger marker
            linestyle='', label='Ambulance (moving)', zorder=6
        )

        # Set title
        title_artist = ax.set_title("Real-time Simulation (Initializing...)")

        # Create legend based on labeled artists
        ax.legend(loc='upper left')

        # Return artists that FuncAnimation should manage
        base_artists = [origins_scatter_artist, served_scatter_artist, ambulance_scatter_artist, title_artist]
        if stations_scatter_artist:
            base_artists.append(stations_scatter_artist)
        return base_artists

    def update_animation(frame_index):
        """Updates the animation for each frame (second)."""
        global dynamic_plot_artists, moving_ambulances, served_origins, origins_scatter_artist, served_scatter_artist, stations_scatter_artist, ambulance_scatter_artist, title_artist
        if frame_index >= total_seconds:
            # Optionally set a final title
            title_artist.set_text(f"Real-time Simulation (Finished at {total_minutes} min)")
            return [] # Stop animation updates

        # Determine current minute and second
        minute_idx = frame_index // 60
        second_in_min = frame_index % 60

        # At the start of each new minute, dispatch ambulances based on pre-calculated assignments
        if second_in_min == 0:
            if minute_idx < len(simulation_history):
                state = simulation_history[minute_idx]
                current_assignment = state["assignments"]
                paths = state["paths"]
                travel_times = state["travel_times"] # Get travel times for this minute

                for (origin, station_node), assigned in current_assignment.items():
                    if assigned:
                        # Check if already served *in the animation* or already moving
                        if origin in served_origins or any(amb['origin'] == origin for amb in moving_ambulances):
                            continue # Skip if already served or ambulance en route

                        path_key = (origin, station_node)
                        if path_key not in paths:
                            print(f"Warning: Path not found for assigned pair {path_key} in minute {minute_idx+1}. Skipping dispatch.")
                            continue
                        if path_key not in travel_times:
                             print(f"Warning: Travel time not found for assigned pair {path_key} in minute {minute_idx+1}. Skipping dispatch.")
                             continue # Skip dispatch if time is missing
                        else:
                             travel_time_seconds = travel_times[path_key]
                             # Basic check for sensible travel time (e.g., must be positive)
                             if travel_time_seconds <= 0:
                                 print(f"Warning: Invalid travel time ({travel_time_seconds}s) for {path_key}. Skipping dispatch.")
                                 continue


                        try:
                            path = paths[path_key]
                            # Ensure path is not empty and nodes exist
                            if not path:
                                print(f"Warning: Empty path for {path_key}. Skipping dispatch.")
                                continue
                            coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in reversed(path)]

                            moving_ambulances.append({
                                "origin": origin,
                                "station": station_node,
                                "coords": coords,
                                "start_second": frame_index, # Start time is the current frame (second)
                                "travel_time": travel_time_seconds, # Use calculated travel time
                            })
                        except KeyError as e:
                            # This error means a node in the path wasn't found in the graph G
                            print(f"Error accessing node coordinates for path {path_key}: Node {e} not found in graph G. Skipping dispatch.")
                        except Exception as e: # Catch other potential errors during coord lookup
                             print(f"Error processing path for {path_key}: {e}. Skipping dispatch.")


        # --- Clear dynamic artists (paths) from previous frame ---
        # Note: Scatter artists (origins, served, ambulances) are updated via set_data, not removed
        for artist in dynamic_plot_artists:
            artist.remove()
        dynamic_plot_artists.clear()

        # --- Update and Plot Moving Ambulances ---
        active_x, active_y = [], []
        ambulances_to_keep = []

        for amb in moving_ambulances:
            elapsed = frame_index - amb["start_second"]
            total_time = amb["travel_time"] # Use the stored travel time for this ambulance

            # Ensure total_time is positive to avoid division by zero or weird behavior
            if total_time <= 0:
                 print(f"Warning: Ambulance for origin {amb['origin']} has invalid travel time {total_time}. Removing.")
                 served_origins.add(amb["origin"]) # Mark as served anyway? Or handle differently? Mark served for now.
                 continue

            path_len = len(amb["coords"])
            if path_len == 0: # Should not happen if checked at dispatch, but safety check
                 print(f"Warning: Ambulance for origin {amb['origin']} has empty coords. Removing.")
                 served_origins.add(amb["origin"])
                 continue


            if elapsed >= total_time:
                # Ambulance reached destination
                served_origins.add(amb["origin"]) # Mark origin as served *in the animation*

                # Draw the completed path temporarily as part of dynamic artists
                if path_len > 0:
                    full_xs, full_ys = zip(*amb["coords"])
                    full_path_artist, = ax.plot(
                        full_xs, full_ys,
                        linestyle='--',
                        linewidth=1.5,
                        color='darkgreen',  # darker green to show completed trip
                        alpha=0.7,
                        zorder=2
                        # No label here
                    )
                    dynamic_plot_artists.append(full_path_artist)

                continue  # DO NOT keep this ambulance in moving_ambulances

            # Ambulance is still moving
            # Calculate current position along the path
            if path_len > 1:
                 progress = elapsed / total_time
                 idx = min(max(0, int(progress * (path_len - 1))), path_len - 1)
            elif path_len == 1:
                 idx = 0 # If path has only one node, stay there
            else:
                 continue # Should not happen


            x, y = amb["coords"][idx]
            active_x.append(x)
            active_y.append(y)

            # --- Plot partial path up to current point ---
            if idx >= 0 and path_len > 0: # Need at least one point to plot path start
                path_xs, path_ys = zip(*amb["coords"][:idx+1])
                line_artist, = ax.plot(
                    path_xs, path_ys,
                    linestyle='--',
                    linewidth=1.0,
                    color='green',  # lighter green during movement
                    alpha=0.5,
                    zorder=3
                    # No label here
                )
                dynamic_plot_artists.append(line_artist)

            ambulances_to_keep.append(amb) # Keep this ambulance for the next frame

        # Update the list of moving ambulances
        moving_ambulances = ambulances_to_keep

        # --- Update current ambulance positions ---
        if active_x and active_y:
            ambulance_scatter_artist.set_data(active_x, active_y)
        else:
            ambulance_scatter_artist.set_data([], []) # Clear if no ambulances moving

        # --- Update incident markers (served/unserved) ---
        current_state = simulation_history[min(minute_idx, len(simulation_history) - 1)]
        origins_to_display = set(current_state["active_origins"])

        unserved_display = [o for o in origins_to_display if o not in served_origins]
        served_display = [o for o in origins_to_display if o in served_origins]

        # Update unserved (blue) scatter
        try:
            if unserved_display:
                ux = [G.nodes[o]['x'] for o in unserved_display]
                uy = [G.nodes[o]['y'] for o in unserved_display]
                origins_scatter_artist.set_data(ux, uy)
            else:
                origins_scatter_artist.set_data([], [])
        except KeyError as e:
            print(f"Error updating unserved incidents: Node {e} not found.")
            origins_scatter_artist.set_data([], []) # Clear on error

        # Update served (green) scatter
        try:
            if served_display:
                sx = [G.nodes[o]['x'] for o in served_display]
                sy = [G.nodes[o]['y'] for o in served_display]
                served_scatter_artist.set_data(sx, sy)
            else:
                served_scatter_artist.set_data([], [])
        except KeyError as e:
            print(f"Error updating served incidents: Node {e} not found.")
            served_scatter_artist.set_data([], []) # Clear on error

        # Update title text
        title_artist.set_text(f"Real-time Simulation (Minute {minute_idx+1}, Second {second_in_min})")

        # --- Return all artists that were modified ---
        # Base artists (updated via set_data) + dynamic path artists
        base_artists = [origins_scatter_artist, served_scatter_artist, ambulance_scatter_artist, title_artist]
        if stations_scatter_artist: # Include static stations if they exist
             base_artists.append(stations_scatter_artist)
        return base_artists + dynamic_plot_artists


    # Create and run the animation using pre-calculated data
    ani = animation.FuncAnimation(
        fig,
        update_animation,
        frames=total_seconds, # Iterate through each second
        init_func=init_animation,
        blit=False, # Use blitting for potential performance improvement
                   # Set to False if experiencing rendering issues/artifacts
        repeat=False,
        interval=50 # Milliseconds between frames (e.g., 50ms = 20fps) - adjust for desired speed
    )

    plt.show() # Display the animation window (blocks until closed)
    print("[INFO] Animation window closed.")
    # --- End Animation Phase ---

    # --- Summary Phase ---
    print("[INFO] Generating simulation summary...")
    summary_lines = [] # Store lines for printing and file writing

    # Create reverse mapping from node ID to name/idx for easier summary printing
    # Handle potential empty ems_station_nodes
    node_to_name_map = {v: k for k, v in ems_station_nodes.items()} if ems_station_nodes else {}

    # 1. Per-station response time summary
    summary_lines.append("\nResponse time summary by station (seconds):")
    if not response_times_per_station:
        summary_lines.append("  No calls were assigned and completed during the simulation.")
    else:
        # Iterate through known stations first to ensure they are listed
        station_nodes_processed = set()
        if ems_station_nodes:
             for name_or_idx, station_node in ems_station_nodes.items():
                 times = response_times_per_station.get(station_node, [])
                 station_name = name_or_idx # Use the key from ems_station_nodes as name
                 if times: # Ensure list is not empty
                     avg = sum(times) / len(times)
                     mx  = max(times)
                     mn  = min(times)
                     summary_lines.append(f"  Station '{station_name}' (Node {station_node}): served {len(times)} calls, "
                                          f"min={mn:.1f}s  avg={avg:.1f}s  max={mx:.1f}s")
                 else:
                      summary_lines.append(f"  Station '{station_name}' (Node {station_node}): served 0 calls")
                 station_nodes_processed.add(station_node)

        # Add any stations found in response times but not in ems_station_nodes (should ideally not happen)
        for station_node, times in response_times_per_station.items():
             if station_node not in station_nodes_processed:
                 station_name = f"Unknown Station {station_node}"
                 if times:
                     avg = sum(times) / len(times)
                     mx  = max(times)
                     mn  = min(times)
                     summary_lines.append(f"  Station '{station_name}' (Node {station_node}): served {len(times)} calls, "
                                          f"min={mn:.1f}s  avg={avg:.1f}s  max={mx:.1f}s")
                 # No else needed, as it wouldn't be in response_times_per_station if times was empty


    # 2. Overall response time stats
    all_times = [t for times in response_times_per_station.values() for t in times]
    if all_times:
        overall_avg = sum(all_times) / len(all_times)
        overall_max = max(all_times)
        overall_min = min(all_times)
        summary_lines.append(f"\nOverall: {len(all_times)} calls served, "
                             f"min response = {overall_min:.1f}s, "
                             f"average response = {overall_avg:.1f}s, "
                             f"max response = {overall_max:.1f}s")
    else:
        summary_lines.append("\nOverall: 0 calls served.")

    # 3. Final ambulance allocation per station
    summary_lines.append("\nFinal ambulance allocation per station (at end of simulation):")
    if not final_alloc:
         summary_lines.append("  No final allocation data available (simulation might have ended early or had no stations/calls).")
    else:
        allocated_nodes_processed = set()
        # Use ems_station_nodes to iterate through known stations for consistent naming
        if ems_station_nodes:
            for name_or_idx, station_node in ems_station_nodes.items():
                count = final_alloc.get(station_node, 0) # Get count from final allocation, default to 0
                summary_lines.append(f"  '{name_or_idx}' (Node {station_node}): {count} ambulances")
                allocated_nodes_processed.add(station_node)

        # Check if final_alloc contains nodes not listed in ems_station_nodes
        unknown_nodes = set(final_alloc.keys()) - allocated_nodes_processed
        for node in unknown_nodes:
             summary_lines.append(f"  'Unknown Station Node {node}': {final_alloc[node]} ambulances")


    # Print summary to console
    print("\n--- Simulation Summary ---")
    for line in summary_lines:
        print(line)
    print("------------------------")

    # Save summary to file
    summary_filename = "simulation_summary.txt"
    try:
        with open(summary_filename, "w") as f:
            f.write("--- Simulation Summary ---\n")
            f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n") # Add timestamp
            f.write(f"Simulation duration: {total_minutes} minutes\n")
            f.write(f"Calls per minute: {calls_per_minute}\n")
            f.write(f"Total ambulances: {total_ambulances}\n")
            for line in summary_lines:
                f.write(line + "\n")
            f.write("------------------------\n")
        print(f"[INFO] Summary saved to {summary_filename}")
    except IOError as e:
        print(f"[ERROR] Could not write summary to file {summary_filename}: {e}")
    # --- End Summary Phase ---