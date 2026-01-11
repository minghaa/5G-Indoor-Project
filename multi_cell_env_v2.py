"""
Multi-Cell 5G Gymnasium Environment V2

State Space: 173 dimensions (17 sim + 12 network + 12×12 cell)
Action Space: Continuous [0,1] × 12 cells
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple

from simulator_v2 import IndoorHotspotSimulatorV2


class MultiCellEnvV2(gym.Env):
    """
    Gymnasium Environment for Multi-Cell 5G Power Control
    
    Matches mentor's DataDescription.xlsx specifications:
    - 173-dim observation space
    - 12-dim continuous action space [0, 1]
    - QoS priority > Energy savings
    """
    
    metadata = {'render_modes': []}
    
    # Reward weights (QoS priority per mentor)
    ALPHA_ENERGY = 0.1        # Energy saving reward (small)
    BETA_DROP = 100.0         # Drop rate penalty (VERY large - QoS priority)
    GAMMA_LATENCY = 20.0      # Latency penalty (VERY large - must stay < 50ms)
    DELTA_STABILITY = 0.1     # Power change penalty (reduced)
    
    # Episode settings
    MAX_STEPS = 300  # 300 seconds simulation
    
    def __init__(self, seed: int = 42):
        """Initialize the multi-cell environment"""
        super().__init__()
        
        # Initialize simulator
        self.simulator = IndoorHotspotSimulatorV2(seed=seed)
        self.n_cells = self.simulator.TOTAL_CELLS
        
        # Action space: Continuous power ratio for each cell [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_cells,),
            dtype=np.float32
        )
        
        # Observation space: 173 dimensions (normalized to [0, 1])
        self.observation_dim = 173
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.prev_actions = np.ones(self.n_cells, dtype=np.float32)
        self.state = None
        
        # Metrics tracking
        self.total_reward = 0.0
        self.total_qos_violations = 0
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset simulator
        if seed is not None:
            self.state = self.simulator.reset(seed=seed)
        else:
            self.state = self.simulator.reset()
        
        # Reset tracking
        self.current_step = 0
        self.prev_actions = np.ones(self.n_cells, dtype=np.float32)
        self.total_reward = 0.0
        self.total_qos_violations = 0
        
        # Get observation
        observation = self.simulator.get_flat_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Array of 12 power ratios [0, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip actions to valid range
        action = np.clip(action, 0.0, 1.0)
        
        # Execute simulation step
        self.state = self.simulator.step(action)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Track QoS violations
        if self.state['qos_violation']:
            self.total_qos_violations += 1
        
        # Update previous actions
        self.prev_actions = action.copy()
        
        # Check termination
        terminated = self.current_step >= self.MAX_STEPS
        truncated = False
        
        # Get observation and info
        observation = self.simulator.get_flat_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward with QoS priority
        
        R = R_energy - R_qos_drop - R_qos_latency - R_stability
        """
        # Energy savings reward (small weight)
        power_saved = self.state['power_saved_percent']
        R_energy = self.ALPHA_ENERGY * (power_saved / 100.0)
        
        # QoS Drop Rate penalty (large weight)
        drop_rate = self.state['avgDropRate']
        if drop_rate > self.simulator.DROP_CALL_THRESHOLD:
            excess = drop_rate - self.simulator.DROP_CALL_THRESHOLD
            R_drop = self.BETA_DROP * (excess ** 2)
        else:
            R_drop = 0.0
        
        # QoS Latency penalty (large weight)
        avg_latency = self.state['avgLatency']
        if avg_latency > self.simulator.LATENCY_THRESHOLD:
            excess = (avg_latency - self.simulator.LATENCY_THRESHOLD) / 100.0
            R_latency = self.GAMMA_LATENCY * (excess ** 2)
        else:
            R_latency = 0.0
        
        # Stability penalty (avoid ping-pong)
        power_change = np.mean(np.abs(action - self.prev_actions))
        R_stability = self.DELTA_STABILITY * power_change
        
        # Total reward
        reward = R_energy - R_drop - R_latency - R_stability
        
        return float(reward)
    
    def _get_info(self) -> Dict:
        """Get additional info dictionary"""
        return {
            'step': self.current_step,
            'avgDropRate': self.state['avgDropRate'],
            'avgLatency': self.state['avgLatency'],
            'qos_violation': self.state['qos_violation'],
            'power_saved_percent': self.state['power_saved_percent'],
            'activeCells': self.state['activeCells'],
            'connectedUEs': self.state['connectedUEs'],
            'total_reward': self.total_reward,
            'total_qos_violations': self.total_qos_violations,
        }


# Test the environment
if __name__ == "__main__":
    print("Testing Multi-Cell Environment V2...")
    print("=" * 60)
    
    env = MultiCellEnvV2(seed=42)
    
    print(f"\nEnvironment specs:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Observation dim: {env.observation_dim}")
    print(f"  Max steps: {env.MAX_STEPS}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: Drop={info['avgDropRate']*100:.2f}%, Latency={info['avgLatency']:.1f}ms")
    
    # Test with full power
    print(f"\n--- Test: Full Power (10 steps) ---")
    env.reset()
    total_reward = 0
    for i in range(10):
        action = np.ones(12)  # All cells at max power
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg Drop: {info['avgDropRate']*100:.2f}%")
    print(f"  Avg Latency: {info['avgLatency']:.1f}ms")
    print(f"  Power saved: {info['power_saved_percent']:.1f}%")
    
    # Test with half power
    print(f"\n--- Test: Half Power (10 steps) ---")
    env.reset()
    total_reward = 0
    for i in range(10):
        action = np.ones(12) * 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg Drop: {info['avgDropRate']*100:.2f}%")
    print(f"  Avg Latency: {info['avgLatency']:.1f}ms")
    print(f"  Power saved: {info['power_saved_percent']:.1f}%")
    
    # Test with random actions
    print(f"\n--- Test: Random Actions (10 steps) ---")
    env.reset()
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  QoS violations: {info['total_qos_violations']}")
    
    print(f"\n✓ Environment V2 test completed!")
