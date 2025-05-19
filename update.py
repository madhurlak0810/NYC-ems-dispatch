import os
import pickle
import sys
import time
import pulp
import random
import geopandas as gpd
from collections import defaultdict

import numpy as np
import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# These are reset/managed within run_realtime_simulation or its inner functions
served_origins           = set()
served_scatter_artist    = None
dynamic_plot_artists     = []
origins_scatter_artist   = None
stations_scatter_artist  = None
moving_ambulances        = []
ambulance_scatter_artist = None # Add artist for moving ambulances
title_artist             = None # Artist for the title text

def run_realtime_simulation(city_graph, station_nodes, ems_station_nodes, total_minutes, calls_per_minute):
    # Pre-calculates simulation states, animates the results, and prints/saves a summary.
    # Args:
    # city_graph        (nx.MultiDiGraph): The road network graph.
    # station_nodes     (list)           : List of graph node IDs representing station locations.
    # ems_station_nodes (dict)           : Dictionary mapping station names/IDs to graph node IDs. Used for annotations and summary.
    # total_minutes     (int)            : Duration of the simulation in minutes.
    # calls_per_minute  (int)            : Average number of new calls per minute.

    total_seconds = total_minutes * 60

    # Pre-calculation Phase
    print("[INFO] Pre-calculating simulation states...")

    simulation_history      = []    # List to store state for each minute
    all_active_origins      = set() # Keep track of all origins that appeared
    precalc_served_origins  = set() # Track origins served during pre-calculation
    on_the_way_origins      = set()
    on_the_way_origins_info = {}
    demand                  = {}

    # Currently we randomly generate number of ambulances for each station.
    # TODO Make it an argument passed into this function.
    final_alloc             = {station: random.randint(1, 10) for station in station_nodes}
    # final_alloc             = {station: 2 for station in station_nodes}
    idle_ambs               = final_alloc.copy()
    unused_ambs             = idle_ambs.copy()
    total_ambulances        = sum([idle_ambs[station] for station in idle_ambs])
    print(station_nodes)
    print(idle_ambs)
    assert(len(station_nodes) == len(idle_ambs))

    # Generate incidents
    total_calls_to_sample = total_minutes * calls_per_minute + 100 # Add buffer
    origin_pool           = sample_origins(city_graph, k=total_calls_to_sample) if city_graph.nodes else list(range(total_calls_to_sample))

    # Initialize structures for summary statistics
    response_times_per_station = defaultdict(list)

    print("\n")

    start_time = time.time()
    for minute in range(total_minutes):
        print(f"Calculating minute: {minute+1}/{total_minutes}")

        # 1. Generate new calls for this minute
        start_idx = minute * calls_per_minute
        end_idx   = (minute + 1) * calls_per_minute
        new_calls = origin_pool[start_idx: end_idx]

        all_active_origins.update(new_calls)

        # TODO: randomize weight
        # Simple demand weight
        for call in new_calls:
            demand[call] = 1

        # 2. Identify currently unserved origins for optimization
        currently_unserved_origins = list(all_active_origins - precalc_served_origins - on_the_way_origins)

        # Check if on-the-way ambulances finished their trip
        for origin in list(on_the_way_origins):
            assert(origin in on_the_way_origins_info)
            if on_the_way_origins_info[origin]["finished_time"] <= minute:
                station = on_the_way_origins_info[origin]["station"]
                print(f"Call from {origin} is resolved by ambulance from station {station}")

                # Remove from on-the-way list, add to served list
                on_the_way_origins.remove(origin)
                on_the_way_origins_info[origin] = None
                precalc_served_origins.add(origin)

                # Add back the ambulance
                idle_ambs[station] += 1

        assert(sum([idle_ambs[station] for station in idle_ambs]) + len(on_the_way_origins) == total_ambulances)

        # 3. Compute travel times and optimize (if unserved calls/stations exist)
        current_alloc      = {}
        current_assignment = {}
        paths              = {}
        travel_time        = {} # Initialize tt for the case where optimization doesn't run

        if currently_unserved_origins and station_nodes:
            # Compute times only for unserved origins to available stations
            travel_time, paths, time_progress = compute_travel_times(city_graph, currently_unserved_origins, station_nodes)

            # Filter demand to only include unserved origins
            current_demand = {origin: demand[origin] for origin in currently_unserved_origins if origin in demand}

            if current_demand: # Ensure there's demand to optimize for
                current_alloc, current_assignment = optimize_dispatch\
                (
                    currently_unserved_origins,
                    station_nodes,
                    current_demand,
                    travel_time,
                    idle_ambs
                )

                # Accumulate response times and mark as served for NEXT iteration's calculation
                for (origin, station_node), assigned in current_assignment.items():
                    # If this station is assigned to this origin
                    if assigned == 1 and current_alloc[station_node] > 0:
                        response_time = travel_time.get((origin, station_node))

                        if response_time is not None:
                            # Store response time for summary
                            response_times_per_station[station_node].append(response_time)

                            # Mark origin as served for the purpose of the *next* minute's optimization
                            on_the_way_origins.add(origin)
                            on_the_way_origins_info[origin] =\
                            {
                                "finished_time": minute + (response_time + 59) / 60,
                                "station"      : station_node
                            }

                            assert(idle_ambs[station_node] > 0)
                            idle_ambs[station_node] -= 1
                        else:
                            print(f"[WARNING] Missing travel time for assigned pair ({origin}, {station_node}) during pre-calculation")

                # Update number of unused ambulance: i.e. minimal number of idling ambulances
                for station_node in station_nodes:
                    unused_ambs[station_node] = min(unused_ambs[station_node], idle_ambs[station_node])
            else:
                print("[WARNING] No demand among unserved origins for optimization.")
        else:
            if not currently_unserved_origins:
                print("[ERROR] No unserved calls for optimization.")
            if not station_nodes:
                print("[ERROR] No stations for optimization.")

        # 4. Store state needed for visualization this minute
        # Note: active_origins for visualization should include ALL origins active up to this minute
        minute_state =\
        {
            "minute"         : minute + 1,
            "active_origins" : list(all_active_origins), # All origins seen so far for plotting
            "allocations"    : dict(current_alloc),      # Allocation decided this minute
            "assignments"    : dict(current_assignment), # Assignments decided this minute
            "paths"          : dict(paths),              # Paths calculated this minute (for unserved)
            "travel_times"   : dict(travel_time),        # Travel times calculated this minute (for unserved)
            "time_progresses": dict(time_progress)
        }
        simulation_history.append(minute_state)

    print(f"[INFO] Pre-calculation finished in {time.time() - start_time:.2f} seconds.")

    # Animation Phase
    print("[INFO] Starting animation...")
    fig, ax = ox.plot_graph\
    (
        city_graph,
        show=False,
        close=False,
        bgcolor='white',
        node_size=0,
        edge_color='#cccccc',
        edge_linewidth=0.5,
        figsize=(12, 12)
    )

    def init_animation():
        global dynamic_plot_artists
        global origins_scatter_artist
        global stations_scatter_artist
        global served_scatter_artist
        global ambulance_scatter_artist
        global title_artist
        dynamic_plot_artists = [] # Clear artists list

        # Plot EMS stations once (static background)
        if station_nodes:
            station_xs = [city_graph.nodes[nid]['x'] for nid in station_nodes]
            station_ys = [city_graph.nodes[nid]['y'] for nid in station_nodes]

            # Use plot for legend consistency, scatter doesn't always play nice with FuncAnimation blitting
            stations_scatter_artist, = ax.plot\
            (
                station_xs,
                station_ys,
                marker='^',
                color='red',
                markersize=10, # Increased size slightly
                linestyle='',
                label='EMS Station',
                zorder=5
            )
        else:
            stations_scatter_artist = None

        # Initialize origins scatter plot (empty at first)
        origins_scatter_artist, = ax.plot\
        (
            [],
            [],
            marker='o',
            color='blue',
            markersize=5,
            linestyle='',
            label='Incident (unserved)',
            zorder=4
        )

        # ambulance visited origins scatter plot
        served_scatter_artist, = ax.plot\
        (
            [],
            [],
            marker='o',
            color='green',
            markersize=5,
            linestyle='',
            label='Incident (served)',
            zorder=4
        )

        # Initialize moving ambulance scatter plot
        ambulance_scatter_artist, = ax.plot\
        (
            [],
            [],
            marker='>',
            color='darkred',
            markersize=7, # Slightly larger marker
            linestyle='',
            label='Ambulance (moving)',
            zorder=6
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

    # Updates the animation for each frame (second).
    def update_animation(frame_index):
        global dynamic_plot_artists
        global moving_ambulances
        global served_origins
        global origins_scatter_artist
        global served_scatter_artist
        global stations_scatter_artist
        global ambulance_scatter_artist
        global title_artist

        # Determine current minute and second
        minute_idx    = frame_index // 60
        second_in_min = frame_index % 60

        if frame_index >= total_seconds and len(moving_ambulances) == 0:
            # Optionally set a final title
            finished_time = f"Minute {minute_idx + 1}, Second {second_in_min}"
            title_artist.set_text(f"Real-time Simulation (Finished at {finished_time})")
            ani.event_source.stop()
            return [] # Stop animation updates

        # At the start of each new minute, dispatch ambulances based on pre-calculated assignments
        if second_in_min == 0:
            if minute_idx < len(simulation_history):
                state              = simulation_history[minute_idx]
                current_assignment = state["assignments"]
                paths              = state["paths"]
                travel_times       = state["travel_times"] # Get travel times for this minute
                time_progresses    = state["time_progresses"]

                for (origin, station_node), assigned in current_assignment.items():
                    if assigned:
                        # Check if already served in the animation or already moving
                        if origin in served_origins or any(amb['origin'] == origin for amb in moving_ambulances):
                            continue # Skip if already served or ambulance en route

                        path_key = (origin, station_node)
                        if path_key not in paths:
                            print(f"[WARNING] Path not found for assigned pair {path_key} in minute {minute_idx+1}. Skipping dispatch.")
                            continue

                        if path_key not in travel_times:
                            print(f"[WARNING] Travel time not found for assigned pair {path_key} in minute {minute_idx+1}. Skipping dispatch.")
                            continue # Skip dispatch if time is missing
                        else:
                            travel_time_seconds = travel_times[path_key]

                            # Basic check for sensible travel time (e.g., must be positive)
                            if travel_time_seconds <= 0:
                                print(f"[WARNING] Invalid travel time ({travel_time_seconds}s) for {path_key}. Skipping dispatch.")
                                continue

                        try:
                            path = paths[path_key]
                            # Ensure path is not empty and nodes exist
                            if not path:
                                print(f"[WARNING] Empty path for {path_key}. Skipping dispatch.")
                                continue

                            # direction: ems station -> accident scene
                            path   = paths[path_key][::-1]
                            coords = [(city_graph.nodes[n]['x'], city_graph.nodes[n]['y']) for n in path]
                            # print("coords\n", coords)

                            # for idx in range(1, len(coords)):
                            #     print(np.linalg.norm(np.array(coords[idx]) - np.array(coords[idx - 1])))

                            # print(time_progresses[path_key])

                            moving_ambulances.append\
                            ({
                                "origin": origin,
                                "station": station_node,
                                "coords": coords,
                                "start_second": frame_index,        # Start time is the current frame (second)
                                "travel_time": travel_time_seconds, # Use calculated travel time
                                "time_progress": time_progresses[path_key]
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
        active_x           = []
        active_y           = []
        ambulances_to_keep = []

        for amb in moving_ambulances:
            elapsed    = frame_index - amb["start_second"]
            total_time = amb["travel_time"] # Use the stored travel time for this ambulance

            # Ensure total_time is positive to avoid division by zero or weird behavior
            if total_time <= 0:
                print(f"[WARNING] Ambulance for origin {amb['origin']} has invalid travel time {total_time}. Removing.")
                served_origins.add(amb["origin"]) # Mark as served anyway? Or handle differently? Mark served for now.
                continue

            path_len = len(amb["coords"])
            if path_len == 0: # Should not happen if checked at dispatch, but safety check
                print(f"[WARNING] Ambulance for origin {amb['origin']} has empty coords. Removing.")
                served_origins.add(amb["origin"])
                continue

            if elapsed >= total_time:
                # Ambulance reached destination
                served_origins.add(amb["origin"]) # Mark origin as served *in the animation*

                # Draw the completed path temporarily as part of dynamic artists
                if path_len > 0:
                    full_xs, full_ys = zip(*amb["coords"])
                    full_path_artist, = ax.plot\
                    (
                        full_xs, full_ys,
                        linestyle='--',
                        linewidth=1.5,
                        color='darkgreen',  # darker green to show completed trip
                        alpha=0.7,
                        zorder=2
                    )
                    dynamic_plot_artists.append(full_path_artist)

                continue  # DO NOT keep this ambulance in moving_ambulances

            # Ambulance is still moving
            # Calculate current position along the path
            if path_len > 1:
                USE_REAL_IDX = True
                if not USE_REAL_IDX:
                    progress = elapsed / total_time
                    idx = min(max(0, int(progress * (path_len - 1))), path_len - 1)
                else:
                    idx = 0
                    time_progress = amb["time_progress"]
                    for i in range(len(time_progress)):
                        if time_progress[i] >= elapsed:
                            idx = max(0, i - 1)
                            break
            elif path_len == 1:
                idx = 0 # If path has only one node, stay there
            else:
                continue # Should not happen

            x, y = amb["coords"][idx]
            active_x.append(x)
            active_y.append(y)

            # --- Plot partial path up to current point ---
            if idx >= 0 and path_len > 0: # Need at least one point to plot path start
                path_xs, path_ys = zip(*amb["coords"][:idx+1]) # from start to current position
                line_artist, = ax.plot\
                (
                    path_xs, path_ys,
                    linestyle='--',
                    linewidth=1.0,
                    color='green', # lighter green during movement
                    alpha=0.5,
                    zorder=3
                )
                dynamic_plot_artists.append(line_artist)

            ambulances_to_keep.append(amb) # Keep this ambulance for the next frame

        # Update the list of moving ambulances
        moving_ambulances = ambulances_to_keep

        # Update current ambulance positions
        if active_x and active_y:
            ambulance_scatter_artist.set_data(active_x, active_y)
        else:
            ambulance_scatter_artist.set_data([], []) # Clear if no ambulances moving

        # Update incident markers (served/unserved)
        current_state      = simulation_history[min(minute_idx, len(simulation_history) - 1)]
        origins_to_display = set(current_state["active_origins"])

        unserved_display = [o for o in origins_to_display if o not in served_origins]
        served_display   = [o for o in origins_to_display if o in served_origins]

        # Update unserved (blue) scatter
        if unserved_display:
            ux = [city_graph.nodes[o]['x'] for o in unserved_display]
            uy = [city_graph.nodes[o]['y'] for o in unserved_display]
            origins_scatter_artist.set_data(ux, uy)
        else:
            origins_scatter_artist.set_data([], [])

        # Update served (green) scatter
        if served_display:
            sx = [city_graph.nodes[o]['x'] for o in served_display]
            sy = [city_graph.nodes[o]['y'] for o in served_display]
            served_scatter_artist.set_data(sx, sy)
        else:
            served_scatter_artist.set_data([], [])

        # Update title text
        title_artist.set_text(f"Real-time Simulation (Minute {minute_idx+1}, Second {second_in_min})")

        # Update legend labels
        origins_scatter_artist.set_label(f"Incident (unserved): {len(unserved_display)}")
        served_scatter_artist.set_label(f"Incident (served): {len(served_display)}")
        ax.legend(loc='upper left')

        # Return all artists that were modified
        # Base artists (updated via set_data) + dynamic path artists
        base_artists = [origins_scatter_artist, served_scatter_artist, ambulance_scatter_artist, title_artist]
        if stations_scatter_artist: # Include static stations if they exist
             base_artists.append(stations_scatter_artist)
        return base_artists + dynamic_plot_artists

    # Create and run the animation using pre-calculated data
    ani = animation.FuncAnimation\
    (
        fig,
        update_animation,
        frames=total_seconds * 2, # Iterate through each second
        init_func=init_animation,
        blit=False, # Set to False if experiencing rendering issues/artifacts
        repeat=False,
        interval=50 # Milliseconds between frames (e.g., 50ms = 20fps) - adjust for desired speed
    )

    # Add Stop and Resume buttons
    stop_ax   = plt.axes([0.80, 0.01, 0.05, 0.05])   # [left, bottom, width, height]
    resume_ax = plt.axes([0.86, 0.01, 0.05, 0.05])

    stop_button   = Button(stop_ax, 'Stop', hovercolor='0.975')
    resume_button = Button(resume_ax, 'Resume', hovercolor='0.975')

    def stop(event):
        print("[INFO] Stop button pressed.")
        ani.event_source.stop()
        title_artist.set_text("Real-time Simulation (Paused)")
        plt.draw()

    def resume(event):
        print("[INFO] Resume button pressed.")
        ani.event_source.start()
        title_artist.set_text("Real-time Simulation (Resumed)")
        plt.draw()

    stop_button.on_clicked(stop)
    resume_button.on_clicked(resume)

    plt.show() # Display the animation window (blocks until closed)
    print("[INFO] Animation window closed.")

    print("[INFO] Generating simulation summary...")
    summary_lines = [] # Store lines for printing and file writing

    # Create reverse mapping from node ID to name/idx for easier summary printing
    # Handle potential empty ems_station_nodes
    # node_to_name_map = {v: k for k, v in ems_station_nodes.items()} if ems_station_nodes else {}

    # 1. Per-station response time summary
    summary_lines.append("\nResponse time summary by station (seconds):")
    if not response_times_per_station:
        summary_lines.append("No calls were assigned and completed during the simulation.")
    else:
        # Iterate through known stations first to ensure they are listed
        station_nodes_processed = set()
        if ems_station_nodes:
             for name_or_idx, station_node in ems_station_nodes.items():
                times        = response_times_per_station.get(station_node, [])
                station_name = name_or_idx # Use the key from ems_station_nodes as name

                if station_node in station_nodes_processed:
                    # TODO handle duplicated ones.
                    print(f"[WARN] Found duplicated station node: ({name_or_idx}, {station_node})")
                    continue

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
                if station_node in allocated_nodes_processed:
                    # TODO handle duplicated ones.
                    print(f"[WARN] Found duplicated station node: ({name_or_idx}, {station_node})")
                    continue

                # Get count from final allocation, default to 0
                count = final_alloc.get(station_node, 0)
                unused_count = unused_ambs.get(station_node, 0)
                summary_lines.append(f"  '{name_or_idx}' (Node {station_node}): {count-unused_count}/{count} ambulances")
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
    with open(summary_filename, "w") as f:
        f.write("--- Simulation Summary ---\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n") # Add timestamp
        f.write(f"Simulation duration: {total_minutes} minutes\n")
        f.write(f"Calls per minute: {calls_per_minute}\n")
        # f.write(f"Total ambulances: {total_ambulances}\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("------------------------\n")
    print(f"[INFO] Summary saved to {summary_filename}")

def cache_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_cached_object(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Fetches and processes a road network graph for all of New York City.
def fetch_nyc_graph() -> nx.MultiDiGraph:
    # 1) Grab the full NYC driving network by name
    # place_name = "New York City, New York, USA"
    place_name = "Manhattan, New York City, New York, USA"
    # place_name = "San Diego, California, USA"
    # place_name = "Highland Park, New Jersey, USA"
    cache_file = place_name + ".pkl"

    # Try loading from cache
    city_graph = load_cached_object(cache_file)

    if city_graph is None:
        print(f"Cache not found for \"{place_name}\". Downloading...")
        city_graph = ox.graph_from_place(place_name, network_type="drive")

        # 2) Project to a local metric CRS (NYLongIsland / EPSG:2263)
        city_graph = ox.project_graph(city_graph, to_crs="EPSG:2263")

        # 3) Compute a default travel_time attribute on every edge (10 m/s)
        default_speed_fps = 10
        for u, v, k, data in city_graph.edges(keys=True, data=True):
            length = data.get("length", 0)             # in meters
            speed  = data.get("maxspeed_fps", default_speed_fps)
            data["travel_time"] = (length / speed) if speed > 0 else float("inf")

        print(f"Cache saving to file \"{cache_file}\"...")
        cache_object(city_graph, cache_file)
    else:
        print(f"Loaded from cache file \"{cache_file}\":", city_graph)

    return city_graph

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

def optimize_dispatch(origins, stations, demand, total_time, idle_ambs, max_time=600):
    print("--------------------------------------------------")
    print("idle_ambs")
    for station in stations:
        print(station, idle_ambs[station])
    print("--------------------------------------------------")

    valid     = [(origin, station) for origin in origins for station in stations if total_time[(origin, station)] <= max_time]
    prob      = pulp.LpProblem('NYC_Ambulance', pulp.LpMinimize)
    assign    = pulp.LpVariable.dicts('asgn', valid, cat='Binary')

    upBound = max([idle_ambs[station] for station in idle_ambs])
    print(f"upBound: {upBound}")
    print(f"len(demand): {len(demand)}")
    amb_count = pulp.LpVariable.dicts('amb', stations, lowBound=0, upBound=upBound, cat='Integer')

    # objective
    prob += pulp.lpSum(demand[origin] * total_time[origin, station] * assign[(origin, station)] for (origin, station) in valid)

    # total ambulances
    # prob += pulp.lpSum(amb_count[station] for station in stations) == len(demand)
    prob += pulp.lpSum(amb_count[station] for station in stations) >= len(demand) * 0.5

    # each origin must be served by => 1 ambulance
    for origin in origins:
        vs = [s for (oo, s) in valid if oo==origin]
        if vs:
            prob += pulp.lpSum(assign[(origin, station)] for station in vs) == 1

    # For every station, number of dispatched ambulances must <= number of available ambulances
    for station in stations:
        prob += amb_count[station] <= idle_ambs[station]

    # Sum of the dispatched ambulances from station s must == number of dispatched ambulances from this station s
    for station in stations:
        prob += pulp.lpSum(assign[(origin, station)] for (origin, s) in valid if s == station) == amb_count[station]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print("==================================================")
    print("Status:", pulp.LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # If LpStatus is not optimal, the optimization is unsolvable.
    # Check if there's too many demands but not enough ambulances?
    assert(pulp.LpStatus[prob.status] == "Optimal")

    print("--------------------------------------------------")
    print("station alloc remaining")
    for station in stations:
        print(station, int(amb_count[station].varValue), idle_ambs[station] - int(amb_count[station].varValue))
    print("==================================================")

    for station in stations:
        assert(int(amb_count[station].varValue) <= idle_ambs[station])

    # (current_alloc, current_assignment)
    current_alloc      = {station:int(amb_count[station].varValue) for station in stations}
    current_assignment = {(origin, station):int(assign[(origin, station)].varValue) for (origin, station) in valid}
    return current_alloc, current_assignment

# Revised fetch_ems_stations
def fetch_ems_stations(city_graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    print("[INFO] Fetching EMS station locations from OSM...")
    # Ensure CRS is defined
    if 'crs' not in city_graph.graph:
        raise ValueError("Graph G must have a 'crs' attribute defined in G.graph['crs']")

    # Get or compute bbox in lat/lon (EPSG:4326)
    if 'bbox' in city_graph.graph:
        # Assuming stored bbox is (west, south, east, north) in EPSG:4326
        west, south, east, north = city_graph.graph['bbox']
        print(f"[INFO] Using stored bbox (EPSG:4326): W={west}, S={south}, E={east}, N={north}")
    else:
        print("[WARN] Graph 'bbox' attribute missing. Calculating from node bounds.")
        # Ensure nodes are in Lat/Lon for bbox calculation if graph isn't already
        if city_graph.graph['crs'].to_epsg() != 4326:
            gdf_nodes = ox.graph_to_gdfs(city_graph, edges=False).to_crs(epsg=4326)
        else:
            gdf_nodes = ox.graph_to_gdfs(city_graph, edges=False)
        west, south, east, north = gdf_nodes.total_bounds  # lon/lat
        city_graph.graph['bbox'] = (west, south, east, north) # Store for future use
        print(f"[INFO] Calculated bbox (EPSG:4326): W={west}, S={south}, E={east}, N={north}")

    # Define EMS tag
    tags = {"emergency": "ambulance_station"}

    # Query OSM for ambulance_station features using the correct bbox order for geometries_from_bbox
    gdf = ox.features_from_bbox(city_graph.graph['bbox'],tags)
    print(f"[INFO] Found {len(gdf)} raw features matching tags.")

    # Keep only points/polygons/multipolygons
    gdf = gdf[gdf.geometry.type.isin(['Point', 'Polygon', 'MultiPolygon'])]
    print(f"[INFO] {len(gdf)} Point/Polygon/MultiPolygon features remaining.")

    if gdf.empty:
        print("[WARN] No suitable EMS station features found.")
        return gdf # Return empty GeoDataFrame

    # Project to graph CRS
    print(f"[INFO] Projecting features to graph CRS: {city_graph.graph['crs']}")
    gdf = gdf.to_crs(city_graph.graph['crs'])

    return gdf

# Revised snap_ems_to_nodes
# Changed return type hint and added name handling
def snap_ems_to_nodes(city_graph: nx.MultiDiGraph, gdf_ems: gpd.GeoDataFrame) -> dict[str | int, int]:
    print(f"[INFO] Snapping {len(gdf_ems)} EMS stations to graph nodes...")
    station_nodes   = {}
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
            counter      = 1
            while station_key in processed_names:
                counter    += 1
                station_key = f"{original_key}_{counter}"
            processed_names.add(station_key)

        # Choose centroid for non-points
        geom = row.geometry.centroid if row.geometry.geom_type != 'Point' else row.geometry
        x, y = geom.x, geom.y

        # Find the nearest node
        node_id = ox.nearest_nodes(city_graph, X=x, Y=y)
        station_nodes[station_key] = node_id
        # print(f"  Snapped '{station_key}' to node {node_id}")

    print(f"[INFO] Successfully snapped {len(station_nodes)} stations.")
    return station_nodes

if __name__ == "__main__":
    print("[INFO] Starting main execution...")

    # 1. Fetch the NYC graph
    print("[INFO] Fetching graph data...")
    city_graph = fetch_nyc_graph()

    # 2. Fetch and process EMS stations
    print("[INFO] Fetching and processing EMS station data...")
    ems_stations_nodes = {}
    station_nodes      = []
    ems_stations_gdf   = fetch_ems_stations(city_graph)

    # drop nan stations, won't display on graph
    ems_stations_gdf = ems_stations_gdf[ems_stations_gdf["name"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    if ems_stations_gdf is not None and not ems_stations_gdf.empty:
        print(f"[INFO] Fetched {len(ems_stations_gdf)} potential EMS station locations.")
        ems_stations_nodes = snap_ems_to_nodes(city_graph, ems_stations_gdf)

        if ems_stations_nodes:
            print(ems_stations_nodes)
            station_nodes = list(ems_stations_nodes.values())
            print(f"[INFO] Successfully snapped {len(station_nodes)} stations to graph nodes.")
        else:
            print("[WARNING] Could not snap any EMS stations to the graph. Simulation might not be meaningful.")
    else:
        print("[WARNING] No EMS station data found or fetched.")

    # Check if we have stations before running simulation
    if not station_nodes:
        print("[ERROR] No station nodes available to run the simulation. Exiting.")
        sys.exit(1)

    # Unique list
    station_nodes = list(set(station_nodes))

    # 3. Run the real-time simulation
    print("[INFO] Starting the real-time simulation...")
    run_realtime_simulation\
        (
            city_graph,
            station_nodes,
            ems_stations_nodes, # Pass the dictionary for potential use in summary/annotations
            total_minutes=10,
            calls_per_minute=7,
        )
    print("[INFO] Simulation finished.")
    print("[INFO] Main execution completed.")