"""
PPO Training Script for Multi-Cell 5G Environment V2
QoS priority over energy savings
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_cell_env_v2 import MultiCellEnvV2


class MetricsCallback(BaseCallback):
    """Callback to track training metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.qos_violations = []
        self.power_saved = []
        
    def _on_step(self) -> bool:
        # Get info from first environment
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'avgDropRate' in info:
                    self.rewards.append(self.locals['rewards'][0])
                    self.qos_violations.append(1 if info['qos_violation'] else 0)
                    self.power_saved.append(info.get('power_saved_percent', 0))
        return True


def train_ppo(total_timesteps: int = 50000, seed: int = 42):
    """Train PPO agent on Multi-Cell environment"""
    print("=" * 60)
    print("PPO Training for Multi-Cell 5G Power Control V2")
    print("=" * 60)
    
    # Create environment
    env = MultiCellEnvV2(seed=seed)
    
    print(f"\nEnvironment:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Max steps: {env.MAX_STEPS}")
    
    # Create PPO model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,  # Steps per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy for exploration
        verbose=1,
        seed=seed,
        # tensorboard_log removed - not installed
    )
    
    # Training callback
    callback = MetricsCallback()
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False  # Disabled - requires tqdm/rich
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Save model
    model_path = "ppo_multicell_v2"
    model.save(model_path)
    print(f"Model saved to: {model_path}.zip")
    
    return model, callback


def evaluate_model(model, env, n_episodes: int = 5):
    """Evaluate trained model"""
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    all_rewards = []
    all_drops = []
    all_latencies = []
    all_power_saved = []
    all_qos_violations = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0
        drops = []
        latencies = []
        power_saved = []
        qos_violations = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            drops.append(info['avgDropRate'])
            latencies.append(info['avgLatency'])
            power_saved.append(info['power_saved_percent'])
            if info['qos_violation']:
                qos_violations += 1
        
        all_rewards.append(episode_reward)
        all_drops.append(np.mean(drops))
        all_latencies.append(np.mean(latencies))
        all_power_saved.append(np.mean(power_saved))
        all_qos_violations.append(qos_violations)
        
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, "
              f"Drop={np.mean(drops)*100:.2f}%, Latency={np.mean(latencies):.1f}ms, "
              f"Power Saved={np.mean(power_saved):.1f}%")
    
    print(f"\nOverall Results:")
    print(f"  Avg Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Avg Drop Rate: {np.mean(all_drops)*100:.2f}%")
    print(f"  Avg Latency: {np.mean(all_latencies):.1f}ms")
    print(f"  Avg Power Saved: {np.mean(all_power_saved):.1f}%")
    print(f"  QoS Violations: {np.mean(all_qos_violations):.1f}/episode")
    
    return {
        'rewards': all_rewards,
        'drops': all_drops,
        'latencies': all_latencies,
        'power_saved': all_power_saved,
        'qos_violations': all_qos_violations
    }


def compare_baselines(env, model):
    """Compare PPO with baseline policies"""
    print("\n" + "=" * 60)
    print("Policy Comparison")
    print("=" * 60)
    
    policies = {
        'Full Power': lambda: np.ones(12),
        'Half Power': lambda: np.ones(12) * 0.5,
        'Random': lambda: env.action_space.sample(),
        'PPO Agent': lambda: model.predict(obs, deterministic=True)[0] if 'obs' in dir() else np.ones(12)
    }
    
    results = {}
    
    for name, policy_fn in policies.items():
        obs, info = env.reset(seed=42)
        total_reward = 0
        drops = []
        latencies = []
        power_saved = []
        
        for _ in range(env.MAX_STEPS):
            if name == 'PPO Agent':
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = policy_fn()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            drops.append(info['avgDropRate'])
            latencies.append(info['avgLatency'])
            power_saved.append(info['power_saved_percent'])
            
            if terminated:
                break
        
        results[name] = {
            'reward': total_reward,
            'avg_drop': np.mean(drops),
            'avg_latency': np.mean(latencies),
            'avg_power_saved': np.mean(power_saved)
        }
        
        print(f"{name:12}: Reward={total_reward:8.2f}, "
              f"Drop={np.mean(drops)*100:5.2f}%, Latency={np.mean(latencies):5.1f}ms, "
              f"PowerSaved={np.mean(power_saved):5.1f}%")
    
    return results


def plot_results(callback, results, save_path="training_results_v2.png"):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training reward curve
    if callback.rewards:
        window = min(100, len(callback.rewards) // 10)
        rewards_smoothed = np.convolve(callback.rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(rewards_smoothed)
        axes[0, 0].set_title('Training Reward (Smoothed)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
    
    # QoS violations
    if callback.qos_violations:
        window = min(100, len(callback.qos_violations) // 10)
        violations_smoothed = np.convolve(callback.qos_violations, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(violations_smoothed)
        axes[0, 1].set_title('QoS Violation Rate (Smoothed)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Violation Rate')
    
    # Policy comparison - Rewards
    names = list(results.keys())
    rewards = [results[n]['reward'] for n in names]
    colors = ['blue', 'orange', 'gray', 'green']
    axes[1, 0].bar(names, rewards, color=colors)
    axes[1, 0].set_title('Policy Comparison: Total Reward')
    axes[1, 0].set_ylabel('Reward')
    
    # Policy comparison - Power Saved vs Drop Rate
    for i, name in enumerate(names):
        axes[1, 1].scatter(
            results[name]['avg_power_saved'], 
            results[name]['avg_drop'] * 100,
            s=200, c=colors[i], label=name
        )
    axes[1, 1].axhline(y=5, color='r', linestyle='--', label='QoS Threshold (5%)')
    axes[1, 1].set_xlabel('Power Saved (%)')
    axes[1, 1].set_ylabel('Drop Rate (%)')
    axes[1, 1].set_title('Trade-off: Energy vs QoS')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Train model
    model, callback = train_ppo(total_timesteps=30000, seed=42)
    
    # Create fresh environment for evaluation
    env = MultiCellEnvV2(seed=42)
    
    # Evaluate
    eval_results = evaluate_model(model, env, n_episodes=3)
    
    # Compare with baselines
    baseline_results = compare_baselines(env, model)
    
    # Plot results
    plot_results(callback, baseline_results)
    
    print("\n✓ Training and evaluation completed!")
