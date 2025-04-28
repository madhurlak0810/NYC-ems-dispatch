import pulp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.patches import Circle

# Set random seed for reproducibility
np.random.seed(42)

# Define problem parameters
n = 10  # Number of demand zones
m = 5   # Number of station locations
K = 8   # Total number of ambulances
T_max = 15  # Maximum allowed response time (minutes)

# Generate synthetic coordinates for demand zones and stations
# Let's assume a city grid of 20km x 20km
city_size = 20

# Generate station locations - place them somewhat strategically
station_coords = {
    1: (5, 5),    # Southwest area
    2: (15, 5),   # Southeast area
    3: (10, 10),  # City center
    4: (5, 15),   # Northwest area
    5: (15, 15)   # Northeast area
}

# Generate demand zone locations randomly across the city
zone_coords = {}
for i in range(1, n+1):
    zone_coords[i] = (random.uniform(0, city_size), random.uniform(0, city_size))

# Calculate travel distances (in minutes) assuming 0.5 km/minute speed
# and adding random variation to account for road networks and traffic
d = {}
for i in range(1, n+1):
    for j in range(1, m+1):
        # Euclidean distance
        distance = np.sqrt((zone_coords[i][0] - station_coords[j][0])**2 + 
                         (zone_coords[i][1] - station_coords[j][1])**2)
        # Convert to travel time (minutes) with some random variation to simulate road networks
        travel_time = (distance / 0.5) * (1 + random.uniform(-0.1, 0.3))
        d[(i, j)] = travel_time

# Generate demand weights based on population density
# Zones closer to the center tend to have higher demand
w = {}
city_center = (city_size/2, city_size/2)
for i in range(1, n+1):
    # Distance from city center
    dist_from_center = np.sqrt((zone_coords[i][0] - city_center[0])**2 + 
                              (zone_coords[i][1] - city_center[1])**2)
    # Higher weight for zones closer to center, with some randomness
    w[i] = max(1, 10 * (1 - dist_from_center/city_size) + random.uniform(-1, 3))

# Normalize weights to sum to 1
total_weight = sum(w.values())
for i in w:
    w[i] = w[i] / total_weight

# Create the optimization model
prob = pulp.LpProblem("Ambulance_Dispatch", pulp.LpMinimize)

# Define variables
x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, upBound=K, cat='Integer') for j in range(1, m+1)}
y = {(i, j): pulp.LpVariable(f"y_{i}_{j}", cat='Binary') for i in range(1, n+1) for j in range(1, m+1)}

# Objective function: minimize weighted average response time
prob += pulp.lpSum(w[i] * d[(i, j)] * y[(i, j)] for i in range(1, n+1) for j in range(1, m+1))

# Constraints
# Total number of ambulances equals K
prob += pulp.lpSum(x[j] for j in range(1, m+1)) == K

# Each demand zone is served by exactly one station
for i in range(1, n+1):
    prob += pulp.lpSum(y[(i, j)] for j in range(1, m+1)) == 1

# A demand zone can only be served by a station with ambulances
for i in range(1, n+1):
    for j in range(1, m+1):
        prob += y[(i, j)] <= x[j]

# Travel time constraint: if travel time exceeds T_max, y_ij must be 0
for i in range(1, n+1):
    for j in range(1, m+1):
        if d[(i, j)] > T_max:
            prob += y[(i, j)] == 0

# Solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=True))

# Print the status
print(f"Status: {pulp.LpStatus[prob.status]}")

# Print the optimal solution
if prob.status == pulp.LpStatusOptimal:
    print("\nOptimal Solution:")
    print("\nNumber of ambulances at each station:")
    for j in range(1, m+1):
        if x[j].value() > 0:
            print(f"Station {j}: {int(x[j].value())} ambulances")
    
    print("\nAssignment of demand zones to stations:")
    for i in range(1, n+1):
        for j in range(1, m+1):
            if y[(i, j)].value() > 0.5:  # Use 0.5 as threshold due to potential floating point errors
                print(f"Zone {i} (demand weight: {w[i]:.3f}) is served by Station {j} - Travel time: {d[(i, j)]:.2f} minutes")
    
    print(f"\nTotal weighted average response time: {pulp.value(prob.objective):.2f} minutes")
    
    # Calculate unweighted average response time for comparison
    total_time = sum(d[(i, j)] * y[(i, j)].value() for i in range(1, n+1) for j in range(1, m+1))
    print(f"Unweighted average response time: {total_time/n:.2f} minutes")

# Visualize the solution - FIX THE LEGEND
plt.figure(figsize=(10, 8))

# Create empty scatter plots for legend
demand_zone_scatter = plt.scatter([], [], c='blue', marker='o', s=100, label='Demand Zones (size = weight)')
active_station_scatter = plt.scatter([], [], c='red', marker='s', s=150, label='Active Stations (size = # of ambulances)')
inactive_station_scatter = plt.scatter([], [], c='gray', marker='s', s=100, label='Inactive Stations (0 ambulances)')
assignment_line = plt.plot([], [], 'k--', alpha=0.5, label='Zone-Station Assignments')[0]
coverage_circle = plt.plot([], [], 'r-', alpha=0.2, label='Maximum Coverage Area (T_max)')[0]

# Plot demand zones
for i in range(1, n+1):
    plt.scatter(zone_coords[i][0], zone_coords[i][1], c='blue', marker='o', s=100*w[i]*n)
    plt.text(zone_coords[i][0], zone_coords[i][1], f'Z{i}', fontsize=9)

# Plot stations with number of ambulances
for j in range(1, m+1):
    marker_size = 100
    if x[j].value() > 0:
        marker_size = 100 + 50 * x[j].value()
        plt.scatter(station_coords[j][0], station_coords[j][1], c='red', marker='s', s=marker_size)
        plt.text(station_coords[j][0], station_coords[j][1], f'S{j}\n({int(x[j].value())})', fontsize=9)
    else:
        plt.scatter(station_coords[j][0], station_coords[j][1], c='gray', marker='s', s=marker_size)
        plt.text(station_coords[j][0], station_coords[j][1], f'S{j}\n(0)', fontsize=9)

# Plot assignment relationships
for i in range(1, n+1):
    for j in range(1, m+1):
        if y[(i, j)].value() > 0.5:
            plt.plot([zone_coords[i][0], station_coords[j][0]], 
                     [zone_coords[i][1], station_coords[j][1]], 'k--', alpha=0.3)

# Draw circles representing max travel time from each active station
for j in range(1, m+1):
    if x[j].value() > 0:
        # Convert T_max to distance using the same speed factor (0.5 km/min)
        radius = T_max * 0.5
        circle = Circle(station_coords[j], radius, fill=False, alpha=0.2, color='red')
        plt.gca().add_patch(circle)

plt.xlim(0, city_size)
plt.ylim(0, city_size)
plt.title("Ambulance Dispatch Optimization")

# Create proper legend with the placeholder elements
plt.legend(handles=[demand_zone_scatter, active_station_scatter, inactive_station_scatter, 
                   assignment_line, coverage_circle])

plt.xlabel("X Coordinate (km)")
plt.ylabel("Y Coordinate (km)")
plt.grid(True, alpha=0.3)
plt.savefig('ambulance_optimization_solution.png')
plt.show()