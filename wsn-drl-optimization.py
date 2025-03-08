import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from datetime import datetime
import os

# Create directory for saving results
os.makedirs("results", exist_ok=True)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

############################## WSN Parameters ##############################

# Sensing Field Dimensions (meters)
FIELD_X = 100
FIELD_Y = 100

# Network parameters
NUM_NODES = 100
SINK_X = 50
SINK_Y = 50
TRANSMISSION_RANGE = 20
CH_PERCENTAGE = 0.1  # 10% of nodes as Cluster Heads

# Energy parameters (all in Joules)
INITIAL_ENERGY_MIN = 1.0
INITIAL_ENERGY_MAX = 2.0
E_ELEC = 50e-9        # Energy for running transceiver circuitry (J/bit)
E_AMP = 100e-12       # Energy for transmitter amplifier (J/bit/m²)
E_DA = 5e-9           # Energy for data aggregation (J/bit)
PACKET_SIZE = 4000    # Size of data packet (bits)

# Simulation parameters
MAX_ROUNDS = 3000
DEAD_NODE_THRESHOLD = 0.0

############################## DRL Parameters ##############################

# DQN Hyperparameters
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99          # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10    # Update target network every N episodes
LEARNING_RATE = 0.001

# Neural Network Parameters
STATE_SIZE = 7        # Features describing the state
ACTION_SIZE = 1       # Action space (next hop selection)
HIDDEN_SIZE = 64      # Hidden layer size

############################## Define Neural Network ##############################

class QNetwork(nn.Module):
    """Deep Q-Network for WSN routing decisions"""
    
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output Q-value for a node
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

############################## Experience Replay ##############################

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

############################## DQN Agent ##############################

class DQNAgent:
    """DQN Agent for WSN routing optimization"""
    
    def __init__(self, state_size, action_size, hidden_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q networks (online and target)
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Initialize exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_end = EPSILON_END
        
        # Initialize step counter
        self.t_step = 0
    
    def get_state(self, current_node, network, sink_pos):
        """Extract state features from node and network"""
        # Normalize values to [0,1] range for better learning
        normalized_energy = current_node['E'] / INITIAL_ENERGY_MAX
        normalized_x = current_node['x'] / FIELD_X
        normalized_y = current_node['y'] / FIELD_Y
        normalized_dist_to_sink = current_node['dts'] / np.sqrt(FIELD_X**2 + FIELD_Y**2)
        normalized_hop_count = current_node['hop'] / (FIELD_X/TRANSMISSION_RANGE)
        
        # Calculate percentage of remaining network energy
        total_energy = sum(node['E'] for node in network if node['cond'] == 1)
        max_possible_energy = NUM_NODES * INITIAL_ENERGY_MAX
        network_energy_percentage = total_energy / max_possible_energy
        
        # Calculate congestion (based on number of active nodes in proximity)
        proximity_nodes = sum(1 for node in network if node['cond'] == 1 and
                             np.sqrt((node['x'] - current_node['x'])**2 + 
                                    (node['y'] - current_node['y'])**2) <= TRANSMISSION_RANGE)
        normalized_congestion = proximity_nodes / NUM_NODES
        
        # Return state as tensor
        state = torch.tensor([
            normalized_energy,
            normalized_x,
            normalized_y,
            normalized_dist_to_sink,
            normalized_hop_count,
            network_energy_percentage,
            normalized_congestion
        ], dtype=torch.float32).unsqueeze(0)
        
        return state
    
    def get_action(self, state, available_nodes):
        """Select action (next hop node) using epsilon-greedy policy"""
        if not available_nodes:
            return None
            
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random next hop
            return random.choice(available_nodes)
        else:
            # Exploit: select node with highest Q-value
            with torch.no_grad():
                q_values = []
                for node in available_nodes:
                    node_state = self.get_state(node, available_nodes, (SINK_X, SINK_Y))
                    q_value = self.q_network(node_state)
                    q_values.append((node, q_value.item()))
                
                # Select node with highest Q-value
                return max(q_values, key=lambda x: x[1])[0]
    
    def update_epsilon(self):
        """Decay epsilon for exploration-exploitation tradeoff"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        """Update model weights based on batch of experiences"""
        # Check if enough samples in memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample random batch from memory
        experiences = self.memory.sample(BATCH_SIZE)
        
        # Convert batch to tensors
        states = torch.cat([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).unsqueeze(-1)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).unsqueeze(-1)
        next_states = torch.cat([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).unsqueeze(-1)
        
        # Get expected Q values
        q_expected = self.q_network(states)
        
        # Get next Q values from target network
        with torch.no_grad():
            q_targets_next = self.target_network(next_states)
            q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.t_step += 1
        if self.t_step % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()

############################## WSN Functions ##############################

def initialize_network(num_nodes, field_x, field_y, sink_x, sink_y, range_c):
    """Initialize WSN with randomly placed nodes"""
    network = []
    
    # Create sensor nodes
    for i in range(num_nodes):
        node = {}
        node['id'] = i
        node['x'] = random.uniform(0, field_x)
        node['y'] = random.uniform(0, field_y)
        node['E'] = random.uniform(INITIAL_ENERGY_MIN, INITIAL_ENERGY_MAX)  # Random initial energy
        node['Eo'] = node['E']  # Store original energy
        node['cond'] = 1  # 1=alive, 0=dead
        # Calculate distance to sink
        node['dts'] = np.sqrt((sink_x - node['x'])**2 + (sink_y - node['y'])**2)
        # Estimate hop count to sink
        node['hop'] = np.ceil(node['dts'] / range_c)
        node['role'] = 0  # 0=regular node, 1=cluster head
        node['closest'] = 0
        node['cluster'] = None
        node['prev'] = 0
        network.append(node)
    
    return network

def select_cluster_heads(network, ch_percentage):
    """Select cluster heads based on energy and position"""
    num_ch = int(len(network) * ch_percentage)
    
    # Score nodes based on energy and position
    for node in network:
        if node['cond'] == 1:  # Only consider alive nodes
            # Combine energy level and centrality
            energy_factor = node['E'] / node['Eo']  # Normalized remaining energy
            position_factor = 1 - (node['dts'] / np.sqrt(FIELD_X**2 + FIELD_Y**2))  # Relative position
            node['score'] = 0.7 * energy_factor + 0.3 * position_factor
    
    # Sort by score and select top nodes as CHs
    alive_nodes = [node for node in network if node['cond'] == 1]
    sorted_nodes = sorted(alive_nodes, key=lambda x: x['score'], reverse=True)
    
    # Reset all roles first
    for node in network:
        node['role'] = 0
    
    # Assign CH roles to top nodes
    ch_count = 0
    cluster_heads = []
    for node in sorted_nodes:
        if node['dts'] >= TRANSMISSION_RANGE and ch_count < num_ch:
            node['role'] = 1  # Set as cluster head
            cluster_heads.append(node)
            ch_count += 1
    
    return cluster_heads

def form_clusters(network, cluster_heads):
    """Assign nodes to their nearest cluster head"""
    # Reset cluster assignments
    for node in network:
        if node['cond'] == 1 and node['role'] == 0:
            node['cluster'] = None
            min_dist = float('inf')
            nearest_ch = None
            
            # Find nearest cluster head
            for ch in cluster_heads:
                dist = np.sqrt((node['x'] - ch['x'])**2 + (node['y'] - ch['y'])**2)
                if dist < min_dist and dist <= TRANSMISSION_RANGE:
                    min_dist = dist
                    nearest_ch = ch
            
            # Assign to cluster if within range
            if nearest_ch:
                node['cluster'] = nearest_ch['id']
                node['dts_ch'] = min_dist
    
    # Group nodes by cluster
    clusters = {}
    for ch in cluster_heads:
        clusters[ch['id']] = [ch]
    
    for node in network:
        if node['cond'] == 1 and node['role'] == 0 and node['cluster'] is not None:
            if node['cluster'] in clusters:
                clusters[node['cluster']].append(node)
    
    return clusters

def calculate_energy_consumption(source, destination, packet_size):
    """Calculate energy consumed for transmission between two nodes"""
    distance = np.sqrt((source['x'] - destination['x'])**2 + (source['y'] - destination['y'])**2)
    
    # Energy for transmission
    e_tx = (E_ELEC + E_DA) * packet_size + E_AMP * packet_size * (distance ** 2)
    
    # Energy for reception
    e_rx = E_ELEC * packet_size
    
    return e_tx, e_rx, distance

def calculate_reward(source_node, path_length, success, energy_used):
    """Calculate reward for reinforcement learning"""
    if not success:
        return -10  # Penalize failed transmissions
    
    # Reward components
    energy_efficiency = source_node['E'] / source_node['Eo']  # Normalized remaining energy
    path_efficiency = 1 / (1 + path_length)  # Shorter paths are better
    
    # Energy usage component - reward using less energy
    max_possible_energy = 2 * E_ELEC * PACKET_SIZE + E_AMP * PACKET_SIZE * (np.sqrt(2) * FIELD_X)**2
    energy_usage_factor = 1 - (energy_used / max_possible_energy)
    
    # Combined reward
    reward = (0.4 * energy_efficiency + 0.3 * path_efficiency + 0.3 * energy_usage_factor) * 10
    
    return reward

############################## Visualization Functions ##############################

def plot_network(network, cluster_heads, sink_pos, round_num=0):
    """Visualize the current state of the network"""
    plt.figure(figsize=(10, 8))
    
    # Plot regular nodes
    regular_x = [node['x'] for node in network if node['role'] == 0 and node['cond'] == 1]
    regular_y = [node['y'] for node in network if node['role'] == 0 and node['cond'] == 1]
    plt.scatter(regular_x, regular_y, c='blue', marker='o', label='Regular Node')
    
    # Plot cluster heads
    ch_x = [ch['x'] for ch in cluster_heads if ch['cond'] == 1]
    ch_y = [ch['y'] for ch in cluster_heads if ch['cond'] == 1]
    plt.scatter(ch_x, ch_y, c='green', marker='^', s=100, label='Cluster Head')
    
    # Plot dead nodes
    dead_x = [node['x'] for node in network if node['cond'] == 0]
    dead_y = [node['y'] for node in network if node['cond'] == 0]
    plt.scatter(dead_x, dead_y, c='red', marker='x', label='Dead Node')
    
    # Plot sink
    plt.scatter(sink_pos[0], sink_pos[1], c='black', marker='s', s=200, label='Sink')
    
    # Add cluster boundaries (simplified)
    for ch in cluster_heads:
        if ch['cond'] == 1:
            circle = plt.Circle((ch['x'], ch['y']), TRANSMISSION_RANGE, fill=False, linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)
    
    plt.xlim(0, FIELD_X)
    plt.ylim(0, FIELD_Y)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'WSN Topology - Round {round_num}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    
    # Save and show
    plt.savefig(f'results/network_round_{round_num}.png', dpi=300)
    plt.close()

def plot_metrics(metrics, save=True):
    """Plot performance metrics from the simulation"""
    rounds = metrics['rounds']
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Alive nodes over time
    axs[0, 0].plot(rounds, metrics['alive_nodes'], 'b-')
    axs[0, 0].set_title('Network Lifetime')
    axs[0, 0].set_xlabel('Round')
    axs[0, 0].set_ylabel('Number of Alive Nodes')
    axs[0, 0].grid(True)
    
    # Plot 2: Average energy over time
    axs[0, 1].plot(rounds, metrics['avg_energy'], 'g-')
    axs[0, 1].set_title('Average Node Energy')
    axs[0, 1].set_xlabel('Round')
    axs[0, 1].set_ylabel('Energy (J)')
    axs[0, 1].grid(True)
    
    # Plot 3: Packets delivered over time
    axs[1, 0].plot(rounds, metrics['packets_delivered'], 'r-')
    axs[1, 0].set_title('Packets Delivered')
    axs[1, 0].set_xlabel('Round')
    axs[1, 0].set_ylabel('Number of Packets')
    axs[1, 0].grid(True)
    
    # Plot 4: Average reward over time
    axs[1, 1].plot(rounds, metrics['avg_reward'], 'm-')
    axs[1, 1].set_title('Average Reward')
    axs[1, 1].set_xlabel('Round')
    axs[1, 1].set_ylabel('Reward')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('results/simulation_metrics.png', dpi=300)
    
    plt.show()

############################## Main Simulation ##############################

def run_simulation():
    """Run the DRL-based WSN simulation"""
    print(f"Starting WSN simulation with {NUM_NODES} nodes at {datetime.now()}")
    
    # Initialize network
    network = initialize_network(NUM_NODES, FIELD_X, FIELD_Y, SINK_X, SINK_Y, TRANSMISSION_RANGE)
    
    # Initialize DRL agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    
    # Initialize metrics
    metrics = {
        'rounds': [],
        'alive_nodes': [],
        'dead_nodes': [],
        'avg_energy': [],
        'packets_delivered': [],
        'energy_consumed': [],
        'avg_reward': [],
        'losses': []
    }
    
    total_packets_delivered = 0
    
    # Main simulation loop
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n===== ROUND {round_num} =====")
        
        # Check for alive nodes
        alive_nodes = [node for node in network if node['cond'] == 1]
        dead_nodes = [node for node in network if node['cond'] == 0]
        
        print(f"Alive nodes: {len(alive_nodes)}")
        print(f"Dead nodes: {len(dead_nodes)}")
        
        # Stop if no more alive nodes
        if len(alive_nodes) == 0:
            print(f"All nodes dead at round {round_num}, ending simulation.")
            break
        
        # Update cluster heads and form clusters
        cluster_heads = select_cluster_heads(network, CH_PERCENTAGE)
        clusters = form_clusters(network, cluster_heads)
        
        print(f"Selected {len(cluster_heads)} cluster heads")
        
        # Save network visualization every 100 rounds
        if round_num % 100 == 0 or round_num == 1:
            plot_network(network, cluster_heads, (SINK_X, SINK_Y), round_num)
        
        # Communication phase
        round_packets_delivered = 0
        round_energy_consumed = 0
        round_rewards = []
        
        # Each active node sends data
        for source_node in alive_nodes:
            if source_node['role'] == 0:  # Only regular nodes initiate data sending
                print(f"\nSource Node {source_node['id']} initiating transmission")
                
                # Initialize tracking variables
                current_node = source_node
                path = [current_node['id']]
                transmission_success = True
                path_energy = 0
                
                # Multi-hop routing until reaching cluster head
                while current_node['role'] == 0 and current_node['cond'] == 1:
                    # Get state for current node
                    state = agent.get_state(current_node, network, (SINK_X, SINK_Y))
                    
                    # Find potential next hops (closer to sink or cluster heads)
                    next_hops = [
                        node for node in alive_nodes
                        if (node['role'] == 1 or node['dts'] < current_node['dts'])
                        and node['E'] > 0
                        and node['id'] != current_node['id']
                        and np.sqrt((node['x'] - current_node['x'])**2 + (node['y'] - current_node['y'])**2) <= TRANSMISSION_RANGE
                    ]
                    
                    if not next_hops:
                        print(f"  No available next hops from node {current_node['id']}")
                        transmission_success = False
                        break
                    
                    # Select next hop using DRL agent
                    next_node = agent.get_action(state, next_hops)
                    
                    if not next_node:
                        print(f"  Failed to select next hop from node {current_node['id']}")
                        transmission_success = False
                        break
                    
                    print(f"  Hop: {current_node['id']} -> {next_node['id']}")
                    
                    # Calculate energy consumption
                    e_tx, e_rx, distance = calculate_energy_consumption(current_node, next_node, PACKET_SIZE)
                    
                    print(f"    Distance: {distance:.2f}m")
                    print(f"    Tx Energy: {e_tx:.6f}J, Rx Energy: {e_rx:.6f}J")
                    
                    # Check if nodes have enough energy
                    if current_node['E'] >= e_tx and next_node['E'] >= e_rx:
                        # Consume energy
                        current_node['E'] -= e_tx
                        next_node['E'] -= e_rx
                        
                        # Track energy usage
                        path_energy += (e_tx + e_rx)
                        round_energy_consumed += (e_tx + e_rx)
                        
                        print(f"    Node {current_node['id']} remaining energy: {current_node['E']:.6f}J")
                        print(f"    Node {next_node['id']} remaining energy: {next_node['E']:.6f}J")
                        
                        # Check for node death
                        if current_node['E'] <= DEAD_NODE_THRESHOLD:
                            current_node['cond'] = 0
                            print(f"    Node {current_node['id']} died during transmission")
                            transmission_success = False
                            break
                        
                        if next_node['E'] <= DEAD_NODE_THRESHOLD:
                            next_node['cond'] = 0
                            print(f"    Node {next_node['id']} died during reception")
                            transmission_success = False
                            break
                    else:
                        print(f"    Insufficient energy for transmission")
                        transmission_success = False
                        break
                    
                    # Move to next node in path
                    path.append(next_node['id'])
                    
                    # Get next state for learning
                    next_state = agent.get_state(next_node, network, (SINK_X, SINK_Y))
                    
                    # Store experience for learning
                    is_terminal = next_node['role'] == 1
                    agent.memory.push(state, next_node['id'], 0, next_state, is_terminal)  # Temporary 0 reward
                    
                    # Move to next node
                    current_node = next_node
                    
                    # If we've reached a cluster head, we're done routing
                    if current_node['role'] == 1:
                        break
                
                # Final transmission from cluster head to sink
                if transmission_success and current_node['role'] == 1:
                    print(f"  Final hop: CH {current_node['id']} -> Sink")
                    
                    # Calculate energy for transmission to sink
                    sink_distance = current_node['dts']
                    final_tx_energy = (E_ELEC + E_DA) * PACKET_SIZE + E_AMP * PACKET_SIZE * (sink_distance ** 2)
                    
                    print(f"    Distance to sink: {sink_distance:.2f}m")
                    print(f"    Final tx energy: {final_tx_energy:.6f}J")
                    
                    if current_node['E'] >= final_tx_energy:
                        # Consume energy
                        current_node['E'] -= final_tx_energy
                        path_energy += final_tx_energy
                        round_energy_consumed += final_tx_energy
                        
                        print(f"    CH {current_node['id']} remaining energy: {current_node['E']:.6f}J")
                        
                        # Check for node death
                        if current_node['E'] <= DEAD_NODE_THRESHOLD:
                            current_node['cond'] = 0
                            print(f"    CH {current_node['id']} died during transmission to sink")
                    else:
                        print(f"    Insufficient energy for transmission to sink")
                        transmission_success = False
                
                # Calculate final reward for the path
                reward = calculate_reward(source_node, len(path), transmission_success, path_energy)
                round_rewards.append(reward)
                
                print(f"  Transmission {'succeeded' if transmission_success else 'failed'}")
                print(f"  Path: {path}")
                print(f"  Reward: {reward:.4f}")
                
                # Update experiences with actual rewards
                for i in range(len(path) - 1):
                    # Get the corresponding node objects
                    node_id = path[i]
                    next_node_id = path[i+1]
                    node = next((n for n in network if n['id'] == node_id), None)
                    next_node = next((n for n in network if n['id'] == next_node_id), None)
                    
                    if node and next_node:
                        # Get states
                        state = agent.get_state(node, network, (SINK_X, SINK_Y))
                        next_state = agent.get_state(next_node, network, (SINK_X, SINK_Y))
                        
                        # Terminal is true if next node is cluster head
                        is_terminal = next_node['role'] == 1
                        
                        # Store experience with actual reward
                        agent.memory.push(state, next_node['id'], reward, next_state, is_terminal)
                
                # Count successful packet delivery
                if transmission_success:
                    round_packets_delivered += 1
                    total_packets_delivered += 1
        
        # DRL learning step
        loss = agent.learn()
        if loss is not None:
            metrics['losses'].append(loss)
            print(f"DRL training loss: {loss:.6f}")
        
        # Update exploration rate
        agent.update_epsilon()
        print(f"Epsilon updated to: {agent.epsilon:.4f}")
        
        # Calculate metrics for this round
        avg_energy = np.mean([node['E'] for node in network if node['cond'] == 1]) if alive_nodes else 0
        avg_reward = np.mean(round_rewards) if round_rewards else 0
        
        # Store metrics
        metrics['rounds'].append(round_num)
        metrics['alive_nodes'].append(len(alive_nodes))
        metrics['dead_nodes'].append(len(dead_nodes))
        metrics['avg_energy'].append(avg_energy)
        metrics['packets_delivered'].append(round_packets_delivered)
        metrics['energy_consumed'].append(round_energy_consumed)
        metrics['avg_reward'].append(avg_reward)
        
        # Print round summary
        print(f"\nROUND {round_num} SUMMARY:")
        print(f"  Alive nodes: {len(alive_nodes)}")
        print(f"  Average node energy: {avg_energy:.6f}J")
        print(f"  Packets delivered: {round_packets_delivered}")
        print(f"  Total energy consumed: {round_energy_consumed:.6f}J")
        print(f"  Average reward: {avg_reward:.4f}")
        
        # Save model periodically
        if round_num % 500 == 0:
            torch.save(agent.q_network.state_dict(), f'results/dqn_model_round_{round_num}.pth')
    
    # End of simulation
    print(f"\nSimulation complete at {datetime.now()}")
    print(f"Network survived for {round_num} rounds")
    print(f"Total packets delivered: {total_packets_delivered}")
    
    # Plot final results
    plot_metrics(metrics)
    
    return metrics, network, agent

# Run the simulation
if __name__ == "__main__":
    metrics, final_network, trained_agent = run_simulation()

    # Save final model
    torch.save(trained_agent.q_network.state_dict(), 'results/dqn_model_final.pth')

    # Print final statistics
    alive_nodes = len([node for node in final_network if node['cond'] == 1])
    first_dead_round = next((i for i, count in enumerate(metrics['alive_nodes'])
                            if count < NUM_NODES), MAX_ROUNDS)
    last_alive_round = len(metrics['rounds'])
    total_packets = sum(metrics['packets_delivered'])
    total_energy = sum(metrics['energy_consumed'])
    avg_reward = np.mean(metrics['avg_reward'])

    print("\nFINAL SIMULATION STATISTICS:")
    print("=" * 50)
    print(f"Network Lifetime: {last_alive_round} rounds")
    print(f"First Node Death: Round {first_dead_round}")
    print(f"Remaining Alive Nodes: {alive_nodes}")
    print(f"Total Packets Delivered: {total_packets}")
    print(f"Total Energy Consumed: {total_energy:.6f} J")
    print(f"Average Network Reward: {avg_reward:.4f}")

    # Save metrics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"results/simulation_metrics_{timestamp}.txt"

    with open(metrics_filename, 'w') as f:
        f.write("WSN-DRL Simulation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Simulation Date: {datetime.now()}\n")
        f.write(f"Network Size: {NUM_NODES} nodes\n")
        f.write(f"Field Size: {FIELD_X}m x {FIELD_Y}m\n")
        f.write(f"Transmission Range: {TRANSMISSION_RANGE}m\n")
        f.write(f"Initial Energy Range: {INITIAL_ENERGY_MIN}-{INITIAL_ENERGY_MAX} J\n")
        f.write("\nPerformance Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Network Lifetime: {last_alive_round} rounds\n")
        f.write(f"First Node Death: Round {first_dead_round}\n")
        f.write(f"Remaining Alive Nodes: {alive_nodes}\n")
        f.write(f"Total Packets Delivered: {total_packets}\n")
        f.write(f"Total Energy Consumed: {total_energy:.6f} J\n")
        f.write(f"Average Network Reward: {avg_reward:.4f}\n")

        # Add DRL parameters
        f.write("\nDRL Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Discount Factor (Gamma): {GAMMA}\n")
        f.write(f"Final Epsilon: {trained_agent.epsilon:.4f}\n")
        f.write(f"Memory Size: {MEMORY_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")

    print(f"\nDetailed metrics saved to: {metrics_filename}")
