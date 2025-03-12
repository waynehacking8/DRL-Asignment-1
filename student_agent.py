import numpy as np
import pickle
import os
from collections import deque

class TaxiAgent:
    """
    A robust agent for the Taxi environment that uses a direct state-mapping approach.
    This agent includes cycle detection to avoid getting stuck in loops.
    """
    
    def __init__(self):
        # Define action names for clearer debugging
        self.actions = {
            0: "SOUTH",
            1: "NORTH",
            2: "EAST",
            3: "WEST",
            4: "PICKUP",
            5: "DROPOFF"
        }
        
        # Initialize cycle detection
        self.history = deque(maxlen=10)  # Store recent positions
        self.stuck_detector = 0  # Counter for detecting when agent is stuck
        
        # Try to load Q-table but don't rely on it
        try:
            q_table_path = os.path.join(os.path.dirname(__file__), "q_table.pkl")
            if os.path.exists(q_table_path):
                with open(q_table_path, "rb") as f:
                    self.q_table = pickle.load(f)
                print(f"Loaded Q-table with {len(self.q_table)} states")
            else:
                self.q_table = {}
        except Exception:
            self.q_table = {}
    
    def get_action(self, obs):
        """
        Select the best action for the given observation.
        
        Args:
            obs: The observation from the environment
            
        Returns:
            int: The selected action (0-5)
        """
        # Convert observation to tuple if needed
        state = tuple(obs) if isinstance(obs, (list, np.ndarray)) else obs
        
        # Extract information from the state
        try:
            if len(state) >= 11:
                taxi_row, taxi_col, pass_idx, dest_idx, in_taxi = state[0:5]
                rel_row, rel_col = state[5:7]
                obstacles = state[7:11]  # north, south, east, west
            else:
                # If state format is unexpected, take random action
                return np.random.randint(0, 6)
            
            # Check for cycles by tracking position
            position = (taxi_row, taxi_col, in_taxi)
            
            # If we've been in this position before in recent history, we might be stuck
            if position in self.history:
                self.stuck_detector += 1
            else:
                self.stuck_detector = 0
                
            # Add current position to history
            self.history.append(position)
            
            # If we're stuck in a cycle, take a random exploratory action
            if self.stuck_detector >= 3:
                # Reset stuck detector
                self.stuck_detector = 0
                
                # Get available actions (non-obstacle directions)
                obs_north, obs_south, obs_east, obs_west = obstacles
                available_actions = []
                
                if obs_north == 0:
                    available_actions.append(1)  # NORTH
                if obs_south == 0:
                    available_actions.append(0)  # SOUTH
                if obs_east == 0:
                    available_actions.append(2)  # EAST
                if obs_west == 0:
                    available_actions.append(3)  # WEST
                
                # If there are available movements, choose randomly
                if available_actions:
                    return np.random.choice(available_actions)
                # Otherwise try pickup/dropoff
                else:
                    return 4 if in_taxi == 0 else 5
            
            # Check if we can pick up or drop off
            if in_taxi == 0 and rel_row == 0 and rel_col == 0:
                return 4  # Pickup
            
            if in_taxi == 1 and rel_row == 0 and rel_col == 0:
                return 5  # Dropoff
            
            # Determine the best direction to move
            return self._get_best_movement(rel_row, rel_col, obstacles, in_taxi)
            
        except Exception as e:
            # Fallback to a random action if any error occurs
            print(f"Error in get_action: {e}")
            return np.random.randint(0, 6)
    
    def _get_best_movement(self, rel_row, rel_col, obstacles, in_taxi):
        """
        Determine the best movement action based on relative position and obstacles.
        
        Args:
            rel_row (int): Relative row position (-1, 0, 1)
            rel_col (int): Relative column position (-1, 0, 1)
            obstacles (tuple): Obstacles in each direction (N, S, E, W)
            in_taxi (int): Whether passenger is in taxi (0 or 1)
            
        Returns:
            int: The best movement action (0-3)
        """
        # Extract obstacle information
        obs_north, obs_south, obs_east, obs_west = obstacles
        
        # Calculate Manhattan distance for vertical and horizontal components
        vert_distance = abs(rel_row)
        horiz_distance = abs(rel_col)
        
        # Add randomness to break ties and prevent cycles
        random_bias = np.random.choice([True, False], p=[0.2, 0.8])
        
        # Determine if we need to move vertically or horizontally
        # Prioritize the direction with the greater distance first,
        # but occasionally randomize to break potential cycles
        if (vert_distance > horiz_distance and not random_bias) or (vert_distance <= horiz_distance and random_bias):
            # Prioritize vertical movement
            if rel_row > 0:  # Need to go south
                if obs_south == 0:  # No obstacle south
                    return 0  # Move SOUTH
                # South is blocked, try horizontal
                if rel_col > 0 and obs_east == 0:
                    return 2  # Move EAST
                elif rel_col < 0 and obs_west == 0:
                    return 3  # Move WEST
                # Try north as last resort
                elif obs_north == 0:
                    return 1  # Move NORTH
            else:  # Need to go north
                if obs_north == 0:  # No obstacle north
                    return 1  # Move NORTH
                # North is blocked, try horizontal
                if rel_col > 0 and obs_east == 0:
                    return 2  # Move EAST
                elif rel_col < 0 and obs_west == 0:
                    return 3  # Move WEST
                # Try south as last resort
                elif obs_south == 0:
                    return 0  # Move SOUTH
        else:
            # Prioritize horizontal movement
            if rel_col > 0:  # Need to go east
                if obs_east == 0:  # No obstacle east
                    return 2  # Move EAST
                # East is blocked, try vertical
                if rel_row > 0 and obs_south == 0:
                    return 0  # Move SOUTH
                elif rel_row < 0 and obs_north == 0:
                    return 1  # Move NORTH
                # Try west as last resort
                elif obs_west == 0:
                    return 3  # Move WEST
            else:  # Need to go west
                if obs_west == 0:  # No obstacle west
                    return 3  # Move WEST
                # West is blocked, try vertical
                if rel_row > 0 and obs_south == 0:
                    return 0  # Move SOUTH
                elif rel_row < 0 and obs_north == 0:
                    return 1  # Move NORTH
                # Try east as last resort
                elif obs_east == 0:
                    return 2  # Move EAST
        
        # If all preferred directions are blocked, move in any available direction
        # with a preference for directions not recently taken (to avoid cycles)
        available_moves = []
        
        if obs_north == 0:
            available_moves.append(1)  # NORTH
        if obs_south == 0:
            available_moves.append(0)  # SOUTH
        if obs_east == 0:
            available_moves.append(2)  # EAST
        if obs_west == 0:
            available_moves.append(3)  # WEST
        
        if available_moves:
            return np.random.choice(available_moves)
        
        # All directions blocked - try pickup/dropoff as last resort
        return 4 if in_taxi == 0 else 5  # PICKUP or DROPOFF


# Instantiate the agent
agent = TaxiAgent()

# Function required by the evaluation system
def get_action(obs):
    """
    Get the best action for the given observation.
    This function is called by the evaluation system.
    
    Args:
        obs: The observation from the environment
    
    Returns:
        int: The selected action (0-5)
    """
    return agent.get_action(obs)
