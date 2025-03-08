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
import heapq  # For Dijkstra's algorithm

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
MAX_ROUNDS = 3000     # Updated value to run for 3000 rounds
DEAD_NODE_THRESHOLD = 0.05  # Updated: node is dead when energy falls below this threshold

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
    num_ch = int(len([n for n in network if n['cond'] == 1]) * ch_percentage)
    
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
        if node['cond'] == 1:
            node['role'] = 0
    
    # Assign CH roles to top nodes
    ch_count = 0
    cluster_heads = []
    
    # Make sure we have at least 1 cluster head if there are any alive nodes
    num_ch = max(1, num_ch) if alive_nodes else 0
    
    for node in sorted_nodes:
        if ch_count < num_ch:
            node['role'] = 1  # Set as cluster head
            cluster_heads.append(node)
            ch_count += 1
            
            # Ensure some CHs are close to the sink
            if ch_count == 1 and node['dts'] > TRANSMISSION_RANGE:
                # Find the node closest to sink with good energy
                close_to_sink = sorted(alive_nodes, key=lambda x: x['dts'])[:5]
                if close_to_sink:
                    best_close = max(close_to_sink, key=lambda x: x['E']/x['Eo'])
                    if best_close['id'] != node['id']:
                        best_close['role'] = 1
                        cluster_heads.append(best_close)
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

def dijkstra_shortest_path(graph, start, end):
    """
    Implementation of Dijkstra's algorithm to find the shortest path 
    from start node to end node in the graph.
    """
    # Initialize distances with infinity for all nodes except start
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Initialize dictionary to track previous nodes for path reconstruction
    previous = {node: None for node in graph}
    
    # Priority queue for nodes to visit
    priority_queue = [(0, start)]
    
    # Track visited nodes
    visited = set()
    
    while priority_queue:
        # Get node with smallest distance from priority queue
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If we reached the end node, we're done
        if current_node == end:
            break
        
        # Skip if we've already processed this node
        if current_node in visited:
            continue
        
        # Mark as visited
        visited.add(current_node)
        
        # Check all neighbors
        for neighbor, weight in graph[current_node].items():
            # Calculate potential new distance
            distance = current_distance + weight
            
            # If we found a better path, update distance and previous node
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                
                # Add to priority queue
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = previous[current]
    
    # Return the path in correct order
    path.reverse()
    
    # Return path and distance (infinity if no path found)
    return path, distances[end]

def build_ch_graph(cluster_heads, sink_pos):
    """Build a graph of cluster heads for Dijkstra's algorithm"""
    # Create a graph representation for Dijkstra
    graph = {}
    
    # Add all cluster heads to the graph
    for ch in cluster_heads:
        ch_id = ch['id']
        graph[ch_id] = {}
        
        # Add connections to other cluster heads if within range
        for other_ch in cluster_heads:
            if ch['id'] != other_ch['id']:
                distance = np.sqrt((ch['x'] - other_ch['x'])**2 + (ch['y'] - other_ch['y'])**2)
                if distance <= TRANSMISSION_RANGE:
                    # Use energy-weighted distance as edge weight
                    energy_factor = 1 / (other_ch['E'] / other_ch['Eo'])
                    graph[ch_id][other_ch['id']] = distance * energy_factor
    
    # Add the sink as a special node (we'll use -1 as its ID)
    graph[-1] = {}
    
    # Connect cluster heads to sink if within range
    for ch in cluster_heads:
        distance = np.sqrt((ch['x'] - sink_pos[0])**2 + (ch['y'] - sink_pos[1])**2)
        if distance <= TRANSMISSION_RANGE:
            graph[ch['id']][-1] = distance
            graph[-1][ch['id']] = distance  # Bidirectional connection
    
    return graph

def calculate_energy_consumption(source, destination, packet_size):
    """Calculate energy consumed for transmission between two nodes"""
    # Improved energy model for more realistic energy consumption
    distance = np.sqrt((source['x'] - destination['x'])**2 + (source['y'] - destination['y'])**2)
    
    # Energy for transmission (factoring in distance more accurately)
    if distance <= 0:
        return 0, 0, 0
    
    # More realistic energy model for transmission
    e_tx = E_ELEC * packet_size + E_AMP * packet_size * (distance ** 2)
    
    # Add data aggregation cost for cluster heads
    if source['role'] == 1:
        e_tx += E_DA * packet_size
        
    # Energy for reception
    e_rx = E_ELEC * packet_size
    
    return e_tx, e_rx, distance

def calculate_reward(source_node, path_length, success, energy_used, remaining_energy_ratio):
    """Calculate reward for reinforcement learning"""
    if not success:
        return -10  # Penalize failed transmissions
    
    # Enhanced reward components
    energy_efficiency = remaining_energy_ratio  # Higher ratio = better energy conservation
    path_efficiency = 1 / (1 + path_length)  # Shorter paths are better
    
    # Energy usage component - reward using less energy
    max_possible_energy = 2 * E_ELEC * PACKET_SIZE + E_AMP * PACKET_SIZE * (np.sqrt(2) * FIELD_X)**2
    energy_usage_factor = 1 - (energy_used / max_possible_energy)
    
    # Combined reward with adjusted weights
    reward = (0.5 * energy_efficiency + 0.3 * path_efficiency + 0.2 * energy_usage_factor) * 10
    
    return reward

def check_node_status(node):
    """Check and update node status based on energy level"""
    if node['E'] <= DEAD_NODE_THRESHOLD and node['cond'] == 1:
        node['cond'] = 0
        print(f"Node {node['id']} died (energy: {node['E']:.6f}J)")
        return True
    return False

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

def print_node_energy_levels(network):
    """Print the energy levels of all nodes"""
    print("\n===== NODE ENERGY LEVELS =====")
    print(f"{'ID':<5} {'Role':<10} {'Status':<10} {'Energy':<10} {'% of Initial':<15} {'Position':<15}")
    print("-" * 65)
    
    # Sort by node ID
    sorted_nodes = sorted(network, key=lambda x: x['id'])
    
    for node in sorted_nodes:
        role = "CH" if node['role'] == 1 else "Regular"
        status = "Alive" if node['cond'] == 1 else "Dead"
        energy = node['E']
        percent = (node['E'] / node['Eo']) * 100
        position = f"({node['x']:.1f}, {node['y']:.1f})"
        
        print(f"{node['id']:<5} {role:<10} {status:<10} {energy:.6f}J {percent:.2f}% {position:<15}")
    
    # Summary statistics
    alive_count = sum(1 for node in network if node['cond'] == 1)
    ch_count = sum(1 for node in network if node['role'] == 1 and node['cond'] == 1)
    avg_energy = np.mean([node['E'] for node in network if node['cond'] == 1]) if alive_count > 0 else 0
    
    print("-" * 65)
    print(f"Alive nodes: {alive_count}/{len(network)}")
    print(f"Active cluster heads: {ch_count}")
    print(f"Average energy of alive nodes: {avg_energy:.6f}J")

############################## Main Simulation ##############################

def run_simulation():
    """Run the DRL-based WSN simulation with Dijkstra for CH routing"""
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
    first_dead_round = -1
    
    # Main simulation loop
    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n===== ROUND {round_num} =====")
        
        # Check for alive nodes
        alive_nodes = [node for node in network if node['cond'] == 1]
        dead_nodes = [node for node in network if node['cond'] == 0]
        
        if len(dead_nodes) > 0 and first_dead_round == -1:
            first_dead_round = round_num
            print(f"First node died at round {round_num}")
        
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
        
        # Build graph of cluster heads for Dijkstra routing
        ch_graph = build_ch_graph(cluster_heads, (SINK_X, SINK_Y))
        
        # Save network visualization every 100 rounds or at first/last round
        if round_num % 100 == 0 or round_num == 1 or len(alive_nodes) < NUM_NODES * 0.1:
            plot_network(network, cluster_heads, (SINK_X, SINK_Y), round_num)
        
        # Communication phase
        round_packets_delivered = 0
        round_energy_consumed = 0
        round_rewards = []
        
        # Each active regular node sends data
        for source_node in alive_nodes:
            if source_node['role'] == 0:  # Only regular nodes initiate data sending
                print(f"\nSource Node {source_node['id']} initiating transmission")
                
                # Skip nodes not assigned to any cluster
                if source_node['cluster'] is None:
                    print(f"  Node {source_node['id']} not in any cluster, skipping")
                    continue
                
                # Find the cluster head for this node
                first_ch = next((ch for ch in cluster_heads if ch['id'] == source_node['cluster']), None)
                
                if not first_ch:
                    print(f"  Could not find cluster head for node {source_node['id']}, skipping")
                    continue
                
                # Initialize tracking variables
                current_node = source_node
                path = [current_node['id']]
                transmission_success = True
                path_energy = 0
                
                # Step 1: Node to its cluster head
                print(f"  Step 1: Node {source_node['id']} -> CH {first_ch['id']}")
                
                # Get state for current node
                state = agent.get_state(current_node, network, (SINK_X, SINK_Y))
                
                # Calculate energy for first hop
                e_tx, e_rx, distance = calculate_energy_consumption(current_node, first_ch, PACKET_SIZE)
                
                print(f"    Distance: {distance:.2f}m")
                print(f"    Tx Energy: {e_tx:.6f}J, Rx Energy: {e_rx:.6f}J")
                
                # Check if nodes have enough energy
                if current_node['E'] >= e_tx and first_ch['E'] >= e_rx:
                    # Consume energy
                    current_node['E'] -= e_tx
                    first_ch['E'] -= e_rx
                    
                    # Track energy usage
                    path_energy += (e_tx + e_rx)
                    round_energy_consumed += (e_tx + e_rx)
                    
                    print(f"    Node {current_node['id']} remaining energy: {current_node['E']:.6f}J")
                    print(f"    CH {first_ch['id']} remaining energy: {first_ch['E']:.6f}J")
                    
                    # Check for node death
                    if check_node_status(current_node):
                        transmission_success = False
                    
                    if check_node_status(first_ch):
                        transmission_success = False
                    
                    # Update current node
                    current_node = first_ch
                    path.append(current_node['id'])
                else:
                    print(f"    Not enough energy for transmission, failing")
                    transmission_success = False
                
                # Step 2: Cluster head to sink (direct or multi-hop through other CHs)
                if transmission_success:
                    print(f"  Step 2: CH {current_node['id']} -> Sink")
                    
                    # Calculate direct distance to sink
                    dist_to_sink = np.sqrt((current_node['x'] - SINK_X)**2 + (current_node['y'] - SINK_Y)**2)
                    
                    if dist_to_sink <= TRANSMISSION_RANGE:
                        # Direct transmission to sink
                        print(f"    Direct transmission to sink (distance: {dist_to_sink:.2f}m)")
                        
                        # Calculate energy for direct transmission
                        sink_node = {'x': SINK_X, 'y': SINK_Y, 'E': float('inf')}
                        e_tx, _, distance = calculate_energy_consumption(current_node, sink_node, PACKET_SIZE)
                        
                        print(f"    Tx Energy: {e_tx:.6f}J")
                        
                        if current_node['E'] >= e_tx:
                            current_node['E'] -= e_tx
                            path_energy += e_tx
                            round_energy_consumed += e_tx
                            path.append(-1)  # -1 represents sink
                            print(f"    CH {current_node['id']} remaining energy: {current_node['E']:.6f}J")
                            
                            # Check for node death
                            check_node_status(current_node)
                        else:
                            print(f"    Not enough energy for transmission to sink, failing")
                            transmission_success = False
                    else:
                        # Multi-hop transmission through other CHs using Dijkstra
                        print(f"    Multi-hop transmission to sink")
                        
                        # Find path to sink using Dijkstra
                        shortest_path, path_length = dijkstra_shortest_path(ch_graph, current_node['id'], -1)
                        
                        if not shortest_path or shortest_path == []:
                            print(f"    No path to sink found, failing")
                            transmission_success = False
                        else:
                            print(f"    Found path: {shortest_path}")
                            
                            # Transmit through the path
                            for i in range(len(shortest_path) - 1):
                                from_id = shortest_path[i]
                                to_id = shortest_path[i + 1]
                                
                                # Skip sink node (already handled)
                                if to_id == -1:
                                    sink_node = {'x': SINK_X, 'y': SINK_Y, 'E': float('inf')}
                                    from_node = next((n for n in network if n['id'] == from_id), None)
                                    
                                    if from_node:
                                        e_tx, _, distance = calculate_energy_consumption(from_node, sink_node, PACKET_SIZE)
                                        
                                        print(f"    Hop {i+1}: Node {from_id} -> Sink (distance: {distance:.2f}m)")
                                        print(f"    Tx Energy: {e_tx:.6f}J")
                                        
                                        if from_node['E'] >= e_tx:
                                            from_node['E'] -= e_tx
                                            path_energy += e_tx
                                            round_energy_consumed += e_tx
                                            print(f"    Node {from_id} remaining energy: {from_node['E']:.6f}J")
                                            
                                            # Check for node death
                                            check_node_status(from_node)
                                        else:
                                            print(f"    Not enough energy for transmission, failing")
                                            transmission_success = False
                                            break
                                else:
                                    from_node = next((n for n in network if n['id'] == from_id), None)
                                    to_node = next((n for n in network if n['id'] == to_id), None)
                                    
                                    if from_node and to_node:
                                        e_tx, e_rx, distance = calculate_energy_consumption(from_node, to_node, PACKET_SIZE)
                                        
                                        print(f"    Hop {i+1}: Node {from_id} -> Node {to_id} (distance: {distance:.2f}m)")
                                        print(f"    Tx Energy: {e_tx:.6f}J, Rx Energy: {e_rx:.6f}J")
                                        
                                        if from_node['E'] >= e_tx and to_node['E'] >= e_rx:
                                            from_node['E'] -= e_tx
                                            to_node['E'] -= e_rx
                                            path_energy += (e_tx + e_rx)
                                            round_energy_consumed += (e_tx + e_rx)
                                            print(f"    Node {from_id} remaining energy: {from_node['E']:.6f}J")
                                            print(f"    Node {to_id} remaining energy: {to_node['E']:.6f}J")
                                            
                                            # Check for node death
                                            if check_node_status(from_node) or check_node_status(to_node):
                                                transmission_success = False
                                                break
                                        else:
                                            print(f"    Not enough energy for transmission, failing")
                                            transmission_success = False
                                            break
                
                # Calculate reward and learn from this experience
                if transmission_success:
                    print(f"  Transmission successful! Path: {path}")
                    round_packets_delivered += 1
                    total_packets_delivered += 1
                    
                    # Calculate remaining energy ratio
                    energy_ratio = current_node['E'] / current_node['Eo']
                    
                    # Calculate reward
                    reward = calculate_reward(source_node, len(path), True, path_energy, energy_ratio)
                else:
                    print(f"  Transmission failed!")
                    reward = calculate_reward(source_node, 0, False, path_energy, 0)
                
                round_rewards.append(reward)
                print(f"  Reward: {reward:.4f}")
                
                # Get next state based on network after transmission
                next_state = agent.get_state(source_node, network, (SINK_X, SINK_Y))
                
                # Store experience in replay memory
                agent.memory.push(state, current_node['id'], reward, next_state, not transmission_success)
        
        # Update agent and learn from experiences
        loss = agent.learn()
        if loss:
            metrics['losses'].append(loss)
        
        # Update exploration policy
        agent.update_epsilon()
        
        # Log metrics
        avg_energy = np.mean([node['E'] for node in network if node['cond'] == 1]) if len(alive_nodes) > 0 else 0
        avg_reward = np.mean(round_rewards) if round_rewards else 0
        
        metrics['rounds'].append(round_num)
        metrics['alive_nodes'].append(len(alive_nodes))
        metrics['dead_nodes'].append(len(dead_nodes))
        metrics['avg_energy'].append(avg_energy)
        metrics['packets_delivered'].append(round_packets_delivered)
        metrics['energy_consumed'].append(round_energy_consumed)
        metrics['avg_reward'].append(avg_reward)
        
        print(f"\nRound {round_num} Summary:")
        print(f"  Packets delivered: {round_packets_delivered}")
        print(f"  Energy consumed: {round_energy_consumed:.6f}J")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Average node energy: {avg_energy:.6f}J")
        print(f"  Agent epsilon: {agent.epsilon:.4f}")
        print(f"  Total lifetime packets: {total_packets_delivered}")
        
        # Print detailed energy levels every 100 rounds
        if round_num % 100 == 0 or round_num == 1:
            print_node_energy_levels(network)
    
    # End of simulation
    print("\n===== SIMULATION COMPLETE =====")
    print(f"Simulation ended at round {metrics['rounds'][-1]}")
    print(f"First node died at round: {first_dead_round}")
    print(f"Last node died at round: {metrics['rounds'][-1] if len(alive_nodes) == 0 else 'N/A'}")
    print(f"Total packets delivered: {total_packets_delivered}")
    
    # Plot final metrics
    plot_metrics(metrics)
    
    return metrics, network, agent

# Run the simulation
if __name__ == "__main__":
    # Start timing the simulation
    start_time = datetime.now()
    print(f"Starting simulation at {start_time}")
    
    # Run the simulation
    metrics, final_network, trained_agent = run_simulation()
    
    # Calculate total execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    # Save final trained model
    model_path = 'results/dqn_model_final.pth'
    torch.save(trained_agent.q_network.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")
    
    # Calculate final statistics
    alive_nodes = len([node for node in final_network if node['cond'] == 1])
    first_dead_round = next((i+1 for i, count in enumerate(metrics['alive_nodes']) 
                           if count < NUM_NODES), MAX_ROUNDS)
    last_alive_round = len(metrics['rounds'])
    total_packets = sum(metrics['packets_delivered'])
    total_energy = sum(metrics['energy_consumed'])
    avg_reward = np.mean(metrics['avg_reward'])
    
    # Calculate energy efficiency
    energy_per_packet = total_energy / total_packets if total_packets > 0 else 0
    network_lifetime_hours = last_alive_round * 0.1  # Assuming each round is 0.1 hours
    
    # Print final statistics
    print("\nFINAL SIMULATION STATISTICS:")
    print("=" * 60)
    print(f"Network Lifetime: {last_alive_round} rounds ({network_lifetime_hours:.2f} hours)")
    print(f"First Node Death: Round {first_dead_round}")
    print(f"Remaining Alive Nodes: {alive_nodes}/{NUM_NODES} ({alive_nodes/NUM_NODES*100:.2f}%)")
    print(f"Total Packets Delivered: {total_packets}")
    print(f"Total Energy Consumed: {total_energy:.6f} J")
    print(f"Energy Efficiency: {energy_per_packet:.6f} J/packet")
    print(f"Average Network Reward: {avg_reward:.4f}")
    print(f"Total Execution Time: {execution_time}")
    
    # Create unique timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to file
    metrics_filename = f"results/simulation_metrics_{timestamp}.txt"
    with open(metrics_filename, 'w') as f:
        f.write("WSN-DRL Simulation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Simulation Date: {datetime.now()}\n")
        f.write(f"Execution Time: {execution_time}\n\n")
        
        f.write("Network Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Network Size: {NUM_NODES} nodes\n")
        f.write(f"Field Size: {FIELD_X}m x {FIELD_Y}m\n")
        f.write(f"Transmission Range: {TRANSMISSION_RANGE}m\n")
        f.write(f"Cluster Head Percentage: {CH_PERCENTAGE*100}%\n")
        f.write(f"Initial Energy Range: {INITIAL_ENERGY_MIN}-{INITIAL_ENERGY_MAX} J\n")
        f.write(f"Packet Size: {PACKET_SIZE} bits\n")
        f.write(f"Maximum Rounds: {MAX_ROUNDS}\n")
        f.write(f"Dead Node Threshold: {DEAD_NODE_THRESHOLD} J\n\n")
        
        f.write("Energy Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"E_ELEC: {E_ELEC} J/bit\n")
        f.write(f"E_AMP: {E_AMP} J/bit/m²\n")
        f.write(f"E_DA: {E_DA} J/bit\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Network Lifetime: {last_alive_round} rounds ({network_lifetime_hours:.2f} hours)\n")
        f.write(f"First Node Death: Round {first_dead_round}\n")
        f.write(f"Remaining Alive Nodes: {alive_nodes}/{NUM_NODES} ({alive_nodes/NUM_NODES*100:.2f}%)\n")
        f.write(f"Total Packets Delivered: {total_packets}\n")
        f.write(f"Total Energy Consumed: {total_energy:.6f} J\n")
        f.write(f"Energy Efficiency: {energy_per_packet:.6f} J/packet\n")
        f.write(f"Average Network Reward: {avg_reward:.4f}\n\n")
        
        f.write("DRL Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"State Size: {STATE_SIZE}\n")
        f.write(f"Action Size: {ACTION_SIZE}\n")
        f.write(f"Hidden Size: {HIDDEN_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Discount Factor (Gamma): {GAMMA}\n")
        f.write(f"Initial Epsilon: {EPSILON_START}\n")
        f.write(f"Final Epsilon: {trained_agent.epsilon:.4f}\n")
        f.write(f"Epsilon Decay: {EPSILON_DECAY}\n")
        f.write(f"Memory Size: {MEMORY_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Target Update: {TARGET_UPDATE} steps\n\n")
        
        # Add round-by-round summary
        f.write("Round-by-Round Summary:\n")
        f.write("-" * 40 + "\n")
        f.write("Round | Alive Nodes | Packets Delivered | Energy Consumed | Avg Energy | Avg Reward\n")
        for i in range(len(metrics['rounds'])):
            f.write(f"{metrics['rounds'][i]:5d} | {metrics['alive_nodes'][i]:11d} | {metrics['packets_delivered'][i]:17d} | {metrics['energy_consumed'][i]:15.6f} | {metrics['avg_energy'][i]:9.6f} | {metrics['avg_reward'][i]:10.4f}\n")
    
    # Save metrics data to CSV for potential further analysis
    csv_filename = f"results/simulation_data_{timestamp}.csv"
    with open(csv_filename, 'w') as f:
        f.write("Round,AliveNodes,DeadNodes,AvgEnergy,PacketsDelivered,EnergyConsumed,AvgReward\n")
        for i in range(len(metrics['rounds'])):
            f.write(f"{metrics['rounds'][i]},{metrics['alive_nodes'][i]},{metrics['dead_nodes'][i]},{metrics['avg_energy'][i]},{metrics['packets_delivered'][i]},{metrics['energy_consumed'][i]},{metrics['avg_reward'][i]}\n")
    
    # Create and save final network state plot
    final_ch = [node for node in final_network if node['role'] == 1 and node['cond'] == 1]
    plot_network(final_network, final_ch, (SINK_X, SINK_Y), last_alive_round)
    
    # Plot and save all metrics
    plot_metrics(metrics, save=True)
    
    print(f"\nDetailed metrics saved to: {metrics_filename}")
    print(f"CSV data saved to: {csv_filename}")
    print("\nSimulation complete!")
