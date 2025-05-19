# Ambulance Dispatch Simulation

This project simulates real-time ambulance dispatching within a specified urban area (e.g., New York City) using OpenStreetMap (OSM) data and a location allocation optimization model.

## Features

*   Fetches road network data for a specified area from OpenStreetMap using OSMnx.
*   Identifies potential EMS (ambulance) station locations from OSM data.
*   Snaps station locations to the nearest nodes on the road network graph.
*   Simulates incoming emergency calls over a defined period.
*   Calculates travel times between active calls and available stations.
*   Optimizes ambulance allocation to stations and assignment to calls at each time step (using a placeholder optimization function).
*   Visualizes the simulation in real-time using Matplotlib animation, showing call locations, station locations, ambulance allocations, and dispatch paths.
*   Generates a summary report of response times (per station and overall) and final ambulance allocations.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd ambulance-dispatch
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: install tkinter for your respective system.*
    *Note: Depending on your system, installing `geopandas` and its dependencies (like `GDAL`, `pyproj`) might require additional system libraries. Refer to the [geopandas installation guide](https://geopandas.org/en/stable/getting_started/install.html) if you encounter issues.*

## Usage

1.  **Run the update simulation script:**
    ```bash
    python update.py
    ```

2.  **Observe the output:**
    *   Console logs will show the progress of fetching data, pre-calculating simulation steps, and displaying animation frames.
    *   A Matplotlib window will open, displaying the animated simulation.
    *   After the animation window is closed, a summary of response times and final ambulance allocations will be printed to the console.
    *   The same summary will be saved to a file named `simulation_summary.txt` in the project directory.

## Project Files

*   `main.py`: The main entry point for the application. Fetches data, prepares inputs, and starts the simulation.
*   `ambulance_network.py`: Contains functions for fetching the graph, sampling origins (calls), finding/snapping EMS stations, computing travel times, and optimizing dispatch (currently includes placeholder logic).
*   `realtime_simulation.py`: Contains the logic for running the simulation step-by-step, pre-calculating states, managing the Matplotlib animation, and generating the final summary.
*   `requirements.txt`: Lists the required Python packages.
*   `README.md`: This file.
*   `update.py`: updated version of main (#TODO have to split into main ambulance_network and realtime_simulation)
*   `initial_solution.py`: naive approach of the EMS dispatch on a synthetic map.
## Dependencies

Key Python libraries used:

*   `osmnx`
*   `networkx`
*   `geopandas`
*   `matplotlib`
*   `shapely`
*   `pandas`
*   `numpy`
*   `pulp`
