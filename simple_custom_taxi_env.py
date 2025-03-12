import numpy as np
import time
import os
from student_agent import get_action

class SimpleTaxiEnv:
    """
    A simplified version of the custom taxi environment for testing.
    This environment follows the same format as the evaluation environment.
    """
    
    def __init__(self, grid_size=7, obstacle_density=0.1, max_steps=200):
        """Initialize the environment."""
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.steps = 0
        self.grid = None
        self.locations = None
        self.taxi_row = None
        self.taxi_col = None
        self.passenger_location = None
        self.destination_location = None
        self.passenger_in_taxi = 0
        
        # Reset to initialize
        self.reset()
    
    def _generate_grid(self):
        """Generate a random grid with obstacles and locations."""
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate random obstacles
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.random.random() < self.obstacle_density:
                    self.grid[i, j] = 1  # 1 represents an obstacle
        
        # Ensure there are at least 4 empty cells for R, G, Y, B locations
        empty_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    empty_cells.append((i, j))
        
        # If there are not enough empty cells, remove some obstacles
        while len(empty_cells) < 5:  # 4 locations + 1 for taxi
            i, j = np.random.randint(0, self.grid_size, size=2)
            if self.grid[i, j] == 1:
                self.grid[i, j] = 0
                empty_cells.append((i, j))
        
        # Randomly select 4 empty cells for R, G, Y, B
        np.random.shuffle(empty_cells)
        self.locations = {
            'R': empty_cells[0],
            'G': empty_cells[1],
            'Y': empty_cells[2],
            'B': empty_cells[3]
        }
        
        # Randomly place taxi in an empty cell
        remaining_cells = empty_cells[4:]
        if remaining_cells:
            self.taxi_row, self.taxi_col = remaining_cells[0]
        else:
            # If no empty cells left, clear a random cell for the taxi
            i, j = np.random.randint(0, self.grid_size, size=2)
            self.grid[i, j] = 0
            self.taxi_row, self.taxi_col = i, j
    
    def _encode_state(self):
        """
        Encode the current state for the agent.
        
        Returns:
            tuple: A tuple representing the state features
        """
        # Passenger locations
        passenger_locations = list(self.locations.keys())
        
        # Encode passenger start and destination location indices
        passenger_start_idx = passenger_locations.index(self.passenger_location)
        dest_idx = passenger_locations.index(self.destination_location)
        
        # Encode passenger status (0: waiting, 1: in taxi)
        passenger_status = self.passenger_in_taxi
        
        # Calculate relative distances from taxi to target (passenger or destination)
        pass_loc = self.locations[self.passenger_location]
        dest_loc = self.locations[self.destination_location]
        
        if not self.passenger_in_taxi:
            # If passenger not in taxi, calculate distance to passenger
            dist_to_target_row = pass_loc[0] - self.taxi_row
            dist_to_target_col = pass_loc[1] - self.taxi_col
        else:
            # If passenger in taxi, calculate distance to destination
            dist_to_target_row = dest_loc[0] - self.taxi_row
            dist_to_target_col = dest_loc[1] - self.taxi_col
        
        # Encode distances as relative (-1 for negative, 0 for 0, 1 for positive)
        rel_dist_row = np.sign(dist_to_target_row)
        rel_dist_col = np.sign(dist_to_target_col)
        
        # Check if obstacles are in immediate surroundings (N, S, E, W)
        obstacle_north = 1 if self.taxi_row > 0 and self.grid[self.taxi_row - 1, self.taxi_col] == 1 else 0
        obstacle_south = 1 if self.taxi_row < self.grid_size - 1 and self.grid[self.taxi_row + 1, self.taxi_col] == 1 else 0
        obstacle_east = 1 if self.taxi_col < self.grid_size - 1 and self.grid[self.taxi_row, self.taxi_col + 1] == 1 else 0
        obstacle_west = 1 if self.taxi_col > 0 and self.grid[self.taxi_row, self.taxi_col - 1] == 1 else 0
        
        return (
            self.taxi_row, 
            self.taxi_col,
            passenger_start_idx,
            dest_idx,
            passenger_status,
            rel_dist_row,
            rel_dist_col,
            obstacle_north,
            obstacle_south,
            obstacle_east,
            obstacle_west
        )
    
    def reset(self):
        """Reset the environment and return the initial observation."""
        # Generate a new random grid
        self._generate_grid()
        
        # Initialize passenger and destination
        loc_keys = list(self.locations.keys())
        self.passenger_location = np.random.choice(loc_keys)
        remaining_locs = [loc for loc in loc_keys if loc != self.passenger_location]
        self.destination_location = np.random.choice(remaining_locs)
        
        # Passenger starts outside the taxi
        self.passenger_in_taxi = 0
        
        # Reset step counter
        self.steps = 0
        
        return self._encode_state()
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Action to take
                0: Move south
                1: Move north
                2: Move east
                3: Move west
                4: Pickup passenger
                5: Dropoff passenger
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.steps += 1
        done = False
        reward = -0.1  # Default small negative reward for each step
        
        # Parse action
        if action == 0:  # Move south
            if self.taxi_row < self.grid_size - 1:
                if self.grid[self.taxi_row + 1, self.taxi_col] == 0:  # No obstacle
                    self.taxi_row += 1
                else:
                    reward = -5  # Penalty for hitting obstacle
            else:
                reward = -5  # Penalty for hitting boundary
        
        elif action == 1:  # Move north
            if self.taxi_row > 0:
                if self.grid[self.taxi_row - 1, self.taxi_col] == 0:  # No obstacle
                    self.taxi_row -= 1
                else:
                    reward = -5  # Penalty for hitting obstacle
            else:
                reward = -5  # Penalty for hitting boundary
        
        elif action == 2:  # Move east
            if self.taxi_col < self.grid_size - 1:
                if self.grid[self.taxi_row, self.taxi_col + 1] == 0:  # No obstacle
                    self.taxi_col += 1
                else:
                    reward = -5  # Penalty for hitting obstacle
            else:
                reward = -5  # Penalty for hitting boundary
        
        elif action == 3:  # Move west
            if self.taxi_col > 0:
                if self.grid[self.taxi_row, self.taxi_col - 1] == 0:  # No obstacle
                    self.taxi_col -= 1
                else:
                    reward = -5  # Penalty for hitting obstacle
            else:
                reward = -5  # Penalty for hitting boundary
        
        elif action == 4:  # Pickup passenger
            if (self.taxi_row, self.taxi_col) == self.locations[self.passenger_location] and not self.passenger_in_taxi:
                self.passenger_in_taxi = 1
                reward = 10  # Bonus for successful pickup
            else:
                reward = -10  # Penalty for incorrect pickup
        
        elif action == 5:  # Dropoff passenger
            if (self.taxi_row, self.taxi_col) == self.locations[self.destination_location] and self.passenger_in_taxi:
                self.passenger_in_taxi = 0
                reward = 50  # Large reward for successful dropoff
                done = True  # Episode ends with successful dropoff
            else:
                reward = -10  # Penalty for incorrect dropoff
        
        # Check if out of fuel (steps > max_steps)
        if self.steps >= self.max_steps:
            reward = -10  # Penalty for running out of fuel
            done = True
            
        return self._encode_state(), reward, done, {}
    
    def render(self):
        """Render the environment (text-based)."""
        grid_display = np.full((self.grid_size, self.grid_size), ' ', dtype=object)
        
        # Fill with obstacles
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 1:
                    grid_display[i, j] = 'X'  # Obstacle
        
        # Place R, G, Y, B locations
        for loc, (row, col) in self.locations.items():
            grid_display[row, col] = loc
        
        # Highlight passenger location and destination
        pass_loc = self.locations[self.passenger_location]
        dest_loc = self.locations[self.destination_location]
        
        if not self.passenger_in_taxi:
            grid_display[pass_loc[0], pass_loc[1]] = f"{self.passenger_location}*"
        
        grid_display[dest_loc[0], dest_loc[1]] = f"{self.destination_location}!"
        
        # Place taxi
        taxi_symbol = 'T' if not self.passenger_in_taxi else 'T+'
        
        # Save original value at taxi position
        orig_val = grid_display[self.taxi_row, self.taxi_col]
        grid_display[self.taxi_row, self.taxi_col] = taxi_symbol
        
        # Print the grid
        print('-' * (self.grid_size * 4 + 1))
        for i in range(self.grid_size):
            print('|', end='')
            for j in range(self.grid_size):
                print(f' {grid_display[i, j]:2}|', end='')
            print()
            print('-' * (self.grid_size * 4 + 1))
        
        # Print status
        print(f"Passenger location: {self.passenger_location} (in taxi: {bool(self.passenger_in_taxi)})")
        print(f"Destination: {self.destination_location}")
        print(f"Steps: {self.steps}/{self.max_steps}")
        
        # Restore original value at taxi position for next render
        grid_display[self.taxi_row, self.taxi_col] = orig_val


def simulate_episode(env, max_steps=200, render=True):
    """
    Simulate a full episode using the student agent.
    
    Args:
        env: The taxi environment
        max_steps: Maximum number of steps per episode
        render: Whether to render the environment after each step
    
    Returns:
        tuple: (total_reward, success) where success is a boolean indicating if the passenger was delivered
    """
    obs = env.reset()
    if render:
        print("\n=== New Episode ===")
        env.render()
    
    total_reward = 0
    done = False
    step = 0
    success = False  # Flag to track successful deliveries
    
    while not done and step < max_steps:
        # Get action from student agent
        action = get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, _ = env.step(action)
        
        # Update total reward
        total_reward += reward
        
        # Check if this was a successful dropoff (reward of 50 indicates successful delivery)
        if reward >= 50:
            success = True
        
        # Display information
        if render:
            print(f"\nStep {step+1}")
            print(f"Action: {action} ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][action]})")
            print(f"Reward: {reward}")
            print(f"Total Reward: {total_reward}")
            # Print state information for debugging
            print(f"State: taxi=({obs[0]},{obs[1]}), pass_idx={obs[2]}, dest_idx={obs[3]}, in_taxi={obs[4]}")
            print(f"       rel_pos=({obs[5]},{obs[6]}), obstacles={obs[7:11]}")
            env.render()
            time.sleep(0.1)  # Slow down rendering for visibility
        
        # Update for next iteration
        obs = next_obs
        step += 1
    
    if render:
        print(f"\n=== Episode Complete ===")
        print(f"Total Reward: {total_reward}")
        print(f"Steps Taken: {step}")
        if success:
            print("Success: Passenger delivered to destination!")
        else:
            print("Failed: Maximum steps reached or other failure.")
        
    return total_reward, success


def evaluate_agent(num_episodes=10, render=True):
    """
    Evaluate the student agent over multiple episodes.
    
    Args:
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        tuple: (avg_reward, success_rate) Average reward and success rate
    """
    # Create different environments for testing generalization
    grid_sizes = list(range(5, 11))  # 5x5 to 10x10
    obstacle_densities = [0.1, 0.15, 0.2]  # Different obstacle densities
    
    total_rewards = []
    successes = 0  # Counter for successful deliveries
    
    for episode in range(num_episodes):
        # Randomly select grid size and obstacle density
        grid_size = np.random.choice(grid_sizes)
        obstacle_density = np.random.choice(obstacle_densities)
        
        # Create environment
        env = SimpleTaxiEnv(grid_size=grid_size, obstacle_density=obstacle_density)
        
        # Print configuration if rendering
        if render:
            print(f"\n\n=== Episode {episode+1}/{num_episodes} ===")
            print(f"Grid Size: {grid_size}x{grid_size}")
            print(f"Obstacle Density: {obstacle_density}")
        
        # Simulate episode - FIXED: Capture both reward and success
        reward, success = simulate_episode(env, render=render)
        total_rewards.append(reward)
        
        # Update success counter
        if success:
            successes += 1
        
        # Print episode result
        if render:
            print(f"Episode {episode+1} Reward: {reward}, Success: {success}")
    
    # Calculate average reward and success rate
    avg_reward = np.mean(total_rewards)
    success_rate = (successes / num_episodes) * 100
    
    # Print evaluation results
    print(f"\n=== Evaluation Complete ===")
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}% ({successes}/{num_episodes} episodes)")
    
    return avg_reward, success_rate


if __name__ == "__main__":
    # First, make sure a q_table.pkl exists or create a basic one
    if not os.path.exists("q_table.pkl"):
        print("No q_table.pkl found. Generating a basic one for testing...")
        from generate_q_table import main as generate_q_table
        generate_q_table()
    
    # Evaluate the agent with more episodes for better accuracy
    num_eval_episodes = 20  # Increase this for more accurate evaluation
    avg_reward, success_rate = evaluate_agent(num_episodes=num_eval_episodes, render=True)
    
    # Print summary
    print(f"\n=== Final Results ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
