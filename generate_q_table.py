import numpy as np
import pickle
import os
from tqdm import tqdm

"""
This script generates a comprehensive Q-table with pre-defined 
values to be included in the submission.

The Q-table is carefully constructed to handle a wide variety of states
that might be encountered in the taxi environment, with values that
encourage effective navigation and task completion.
"""

def generate_comprehensive_q_table():
    """Generate a comprehensive Q-table with strategic values for the taxi environment."""
    q_table = {}
    print("Generating comprehensive Q-table...")
    
    # Create a more extensive state space coverage
    # Grid sizes to consider
    grid_sizes = range(5, 11)  # 5x5 to 10x10
    
    # Generate all possible passenger and destination location combinations
    loc_indices = range(4)  # 0, 1, 2, 3 for R, G, Y, B
    
    # We'll sample different positions within the grid
    # to keep the Q-table size manageable
    positions_to_sample = 5  # Sample this many positions per grid size
    
    # Value constants for different actions
    PICKUP_VALUE = 50.0
    DROPOFF_VALUE = 75.0
    MOVEMENT_VALUE = 15.0
    OBSTACLE_PENALTY = -10.0
    
    # First handle the critical states: pickup and dropoff
    print("Generating pickup and dropoff states...")
    for size in grid_sizes:
        # Sample some positions
        positions = [(i, j) for i in range(size) for j in range(size)]
        np.random.shuffle(positions)
        positions = positions[:positions_to_sample]
        
        for taxi_row, taxi_col in positions:
            for pass_idx in loc_indices:
                for dest_idx in loc_indices:
                    if pass_idx == dest_idx:
                        continue  # Skip if pickup and destination are the same
                    
                    # At pickup location with passenger waiting
                    state = (taxi_row, taxi_col, pass_idx, dest_idx, 0, 0, 0, 0, 0, 0, 0)
                    q_table[state] = np.zeros(6)
                    q_table[state][4] = PICKUP_VALUE  # High value for pickup
                    
                    # At destination with passenger in taxi
                    state = (taxi_row, taxi_col, pass_idx, dest_idx, 1, 0, 0, 0, 0, 0, 0)
                    q_table[state] = np.zeros(6)
                    q_table[state][5] = DROPOFF_VALUE  # Even higher value for dropoff
    
    # Now handle navigation states
    print("Generating navigation states...")
    for size in grid_sizes:
        # Sample some positions
        positions = [(i, j) for i in range(size) for j in range(size)]
        np.random.shuffle(positions)
        positions = positions[:positions_to_sample]
        
        for taxi_row, taxi_col in positions:
            for pass_idx in loc_indices:
                for dest_idx in loc_indices:
                    if pass_idx == dest_idx:
                        continue  # Skip if pickup and destination are the same
                    
                    for passenger_status in [0, 1]:  # Not in taxi, in taxi
                        for rel_row in [-1, 0, 1]:
                            for rel_col in [-1, 0, 1]:
                                # Skip if at target (pickup/destination location)
                                if rel_row == 0 and rel_col == 0:
                                    continue
                                
                                # Generate states for different obstacle combinations
                                for obs_n in [0, 1]:
                                    for obs_s in [0, 1]:
                                        for obs_e in [0, 1]:
                                            for obs_w in [0, 1]:
                                                # Create state
                                                state = (
                                                    taxi_row, taxi_col,
                                                    pass_idx, dest_idx,
                                                    passenger_status,
                                                    rel_row, rel_col,
                                                    obs_n, obs_s, obs_e, obs_w
                                                )
                                                
                                                q_table[state] = np.zeros(6)
                                                
                                                # Set navigation Q-values based on relative position
                                                # and obstacle presence
                                                if passenger_status == 0:  # Moving to pickup
                                                    # Vertical movement
                                                    if rel_row < 0:  # Need to go north
                                                        if obs_n == 0:  # No obstacle north
                                                            q_table[state][1] = MOVEMENT_VALUE  # North
                                                        else:  # Obstacle north
                                                            q_table[state][1] = OBSTACLE_PENALTY
                                                    elif rel_row > 0:  # Need to go south
                                                        if obs_s == 0:  # No obstacle south
                                                            q_table[state][0] = MOVEMENT_VALUE  # South
                                                        else:  # Obstacle south
                                                            q_table[state][0] = OBSTACLE_PENALTY
                                                    
                                                    # Horizontal movement
                                                    if rel_col < 0:  # Need to go west
                                                        if obs_w == 0:  # No obstacle west
                                                            q_table[state][3] = MOVEMENT_VALUE  # West
                                                        else:  # Obstacle west
                                                            q_table[state][3] = OBSTACLE_PENALTY
                                                    elif rel_col > 0:  # Need to go east
                                                        if obs_e == 0:  # No obstacle east
                                                            q_table[state][2] = MOVEMENT_VALUE  # East
                                                        else:  # Obstacle east
                                                            q_table[state][2] = OBSTACLE_PENALTY
                                                
                                                else:  # Moving to destination
                                                    # Vertical movement
                                                    if rel_row < 0:  # Need to go north
                                                        if obs_n == 0:  # No obstacle north
                                                            q_table[state][1] = MOVEMENT_VALUE  # North
                                                        else:  # Obstacle north
                                                            q_table[state][1] = OBSTACLE_PENALTY
                                                    elif rel_row > 0:  # Need to go south
                                                        if obs_s == 0:  # No obstacle south
                                                            q_table[state][0] = MOVEMENT_VALUE  # South
                                                        else:  # Obstacle south
                                                            q_table[state][0] = OBSTACLE_PENALTY
                                                    
                                                    # Horizontal movement
                                                    if rel_col < 0:  # Need to go west
                                                        if obs_w == 0:  # No obstacle west
                                                            q_table[state][3] = MOVEMENT_VALUE  # West
                                                        else:  # Obstacle west
                                                            q_table[state][3] = OBSTACLE_PENALTY
                                                    elif rel_col > 0:  # Need to go east
                                                        if obs_e == 0:  # No obstacle east
                                                            q_table[state][2] = MOVEMENT_VALUE  # East
                                                        else:  # Obstacle east
                                                            q_table[state][2] = OBSTACLE_PENALTY
    
    # Add special cases for when surrounded by obstacles
    print("Adding special cases for obstacle handling...")
    # For each passenger status
    for passenger_status in [0, 1]:
        # For various relative positions
        for rel_row in [-1, 0, 1]:
            for rel_col in [-1, 0, 1]:
                if rel_row == 0 and rel_col == 0:
                    continue
                
                # Case where all directions are blocked
                state = (5, 5, 0, 1, passenger_status, rel_row, rel_col, 1, 1, 1, 1)
                q_table[state] = np.zeros(6)
                # Even when blocked, we prefer trying pickup/dropoff over hitting walls
                if passenger_status == 0:
                    q_table[state][4] = 5.0  # Try pickup
                else:
                    q_table[state][5] = 5.0  # Try dropoff
                
                # Cases with some blocks but paths around them
                # Example: Blocked north and south, but east and west clear
                state = (5, 5, 0, 1, passenger_status, rel_row, rel_col, 1, 1, 0, 0)
                q_table[state] = np.zeros(6)
                if rel_col < 0:  # Target is to the west
                    q_table[state][3] = MOVEMENT_VALUE  # Go west
                elif rel_col > 0:  # Target is to the east
                    q_table[state][2] = MOVEMENT_VALUE  # Go east
                else:  # Can't go directly toward target
                    # Choose a clear path to hopefully go around obstacle
                    q_table[state][2] = MOVEMENT_VALUE / 2  # Try east
                    q_table[state][3] = MOVEMENT_VALUE / 2  # Try west
                
                # Example: Blocked east and west, but north and south clear
                state = (5, 5, 0, 1, passenger_status, rel_row, rel_col, 0, 0, 1, 1)
                q_table[state] = np.zeros(6)
                if rel_row < 0:  # Target is to the north
                    q_table[state][1] = MOVEMENT_VALUE  # Go north
                elif rel_row > 0:  # Target is to the south
                    q_table[state][0] = MOVEMENT_VALUE  # Go south
                else:  # Can't go directly toward target
                    # Choose a clear path to hopefully go around obstacle
                    q_table[state][0] = MOVEMENT_VALUE / 2  # Try south
                    q_table[state][1] = MOVEMENT_VALUE / 2  # Try north
    
    # Generate states for different grid sizes
    print("Adding generalized states for different grid sizes...")
    for size in range(5, 11):
        # Create states for middle, edges, and corners of grid
        positions = [
            (size//2, size//2),  # Middle
            (0, size//2),        # Top edge
            (size-1, size//2),   # Bottom edge
            (size//2, 0),        # Left edge
            (size//2, size-1),   # Right edge
            (0, 0),              # Top-left corner
            (0, size-1),         # Top-right corner
            (size-1, 0),         # Bottom-left corner
            (size-1, size-1)     # Bottom-right corner
        ]
        
        for taxi_row, taxi_col in positions:
            for passenger_status in [0, 1]:
                for rel_row in [-1, 0, 1]:
                    for rel_col in [-1, 0, 1]:
                        # Skip if at target
                        if rel_row == 0 and rel_col == 0:
                            continue
                            
                        # Create state with no obstacles
                        state = (
                            taxi_row, taxi_col,
                            0, 1,  # Some arbitrary pickup/dropoff indices
                            passenger_status,
                            rel_row, rel_col,
                            0, 0, 0, 0  # No obstacles
                        )
                        
                        q_table[state] = np.zeros(6)
                        
                        # Set Q-values for navigation
                        if passenger_status == 0:  # Not in taxi
                            # Move towards passenger
                            if rel_row < 0:
                                q_table[state][1] = MOVEMENT_VALUE  # North
                            elif rel_row > 0:
                                q_table[state][0] = MOVEMENT_VALUE  # South
                                
                            if rel_col < 0:
                                q_table[state][3] = MOVEMENT_VALUE  # West
                            elif rel_col > 0:
                                q_table[state][2] = MOVEMENT_VALUE  # East
                        else:  # In taxi
                            # Move towards destination
                            if rel_row < 0:
                                q_table[state][1] = MOVEMENT_VALUE  # North
                            elif rel_row > 0:
                                q_table[state][0] = MOVEMENT_VALUE  # South
                                
                            if rel_col < 0:
                                q_table[state][3] = MOVEMENT_VALUE  # West
                            elif rel_col > 0:
                                q_table[state][2] = MOVEMENT_VALUE  # East
    
    return q_table

def main():
    """Generate and save a comprehensive Q-table."""
    q_table = generate_comprehensive_q_table()
    
    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    
    print(f"Q-table with {len(q_table)} states saved to q_table.pkl")

if __name__ == "__main__":
    main()
