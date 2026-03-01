"""
Test script for SAC (Soft Actor-Critic) Agent Performance

Tests the SAC agent on OpenAI Gymnasium's LunarLanderContinuous-v2 environment:
- Training convergence
- Action selection (stochastic vs deterministic)
- Network updates and loss tracking
- Model save/load functionality
- Performance metrics

Requirements:
    pip install gymnasium[box2d]

Run with: python tests/test_sac_agent.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from datetime import datetime

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed. Run: pip install gymnasium[box2d]")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Run: pip install tensorboard")

from src.agents.sac_agent import SACAgent


# Environment configuration - change this to test different environments
# Examples: "Pendulum-v1", "MountainCarContinuous-v0", "LunarLanderContinuous-v2" (requires Box2D), "BipedalWalker-v3" (requires Box2D)
ENV_NAME = "Pendulum-v1"


def get_env_dimensions(env_name: str):
    """
    Get state and action dimensions from the environment.
    
    Args:
        env_name: Name of the Gymnasium environment
        
    Returns:
        Tuple of (state_size, action_size)
    """
    if not GYM_AVAILABLE:
        raise RuntimeError("Gymnasium not available. Install with: pip install gymnasium[box2d]")
    
    env = gym.make(env_name)
    
    # Get observation space size
    if hasattr(env.observation_space, 'shape'):
        state_size = env.observation_space.shape[0]
    else:
        raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    # Get action space size
    if hasattr(env.action_space, 'shape'):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space: {env.action_space}")
    
    env.close()
    
    return state_size, action_size


def test_agent_initialization(state_size: int, action_size: int):
    """Test SAC agent initialization with config file."""
    print("\n" + "=" * 60)
    print("TEST 1: Agent Initialization")
    print("=" * 60)
    
    config_path = project_root / "sac_config" / "sac_config.yaml"
    
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return None
    
    try:
        agent = SACAgent(
            state_size=state_size,
            action_size=action_size,
            config_path=str(config_path)
        )
        print(f"  ✓ Agent initialized successfully")
        print(f"    State size: {state_size}")
        print(f"    Action size: {action_size}")
        print(f"    Device: {agent.device}")
        print(f"    Learning rate: {agent.learning_rate}")
        print(f"    Gamma: {agent.gamma}")
        print(f"    Tau: {agent.tau}")
        print(f"    Alpha: {agent.alpha}")
        print(f"    Batch size: {agent.batch_size}")
        print(f"    Auto entropy tuning: {agent.automatic_entropy_tuning}")
        return agent
    except Exception as e:
        print(f"  ✗ Failed to initialize agent: {e}")
        return None


def test_action_selection(agent: SACAgent, state_size: int):
    """Test action selection in both stochastic and deterministic modes."""
    print("\n" + "=" * 60)
    print("TEST 2: Action Selection")
    print("=" * 60)
    
    state = np.random.randn(state_size).astype(np.float32)
    
    # Test stochastic action (exploration)
    stochastic_actions = []
    for _ in range(5):
        action = agent.select_action(state, evaluate=False)
        stochastic_actions.append(action)
    
    print(f"  Stochastic actions (5 samples):")
    for i, a in enumerate(stochastic_actions):
        print(f"    {i+1}: {a}")
    
    # Check that stochastic actions vary
    actions_array = np.array(stochastic_actions)
    variance = np.var(actions_array, axis=0).mean()
    print(f"  Action variance: {variance:.6f}")
    
    if variance > 0.001:
        print(f"  ✓ Stochastic actions show exploration (variance > 0.001)")
    else:
        print(f"  ⚠ Low variance in stochastic actions")
    
    # Test deterministic action (evaluation)
    deterministic_actions = []
    for _ in range(5):
        action = agent.select_action(state, evaluate=True)
        deterministic_actions.append(action)
    
    det_array = np.array(deterministic_actions)
    det_variance = np.var(det_array, axis=0).mean()
    print(f"\n  Deterministic action variance: {det_variance:.10f}")
    
    if det_variance < 1e-6:
        print(f"  ✓ Deterministic actions are consistent")
    else:
        print(f"  ⚠ Deterministic actions show unexpected variance")
    
    # Check action bounds
    all_actions = np.concatenate([actions_array, det_array])
    if np.all(np.abs(all_actions) <= 1.0):
        print(f"  ✓ All actions within [-1, 1] bounds")
    else:
        print(f"  ✗ Actions exceed bounds!")


def test_training_loop(agent: SACAgent, env_name: str, num_episodes: int = 100, render: bool = False, 
                       use_tensorboard: bool = True):
    """Test training loop and convergence.
    
    Args:
        agent: SAC agent to train
        env_name: Name of the Gymnasium environment
        num_episodes: Number of training episodes
        render: If True, render the environment during training
        use_tensorboard: If True, log metrics to TensorBoard
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: Training on {env_name} ({num_episodes} episodes)")
    if render:
        print("  [Rendering enabled]")
    print("=" * 60)
    
    if not GYM_AVAILABLE:
        print("  ✗ Gymnasium not available. Install with: pip install gymnasium[box2d]")
        return [], []
    
    # Setup TensorBoard
    writer = None
    if use_tensorboard and TENSORBOARD_AVAILABLE:
        log_dir = project_root / "runs" / f"sac_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"  [TensorBoard logging to: {log_dir}]")
        print(f"  [Run: tensorboard --logdir={project_root / 'runs'}]")
    elif use_tensorboard and not TENSORBOARD_AVAILABLE:
        print("  [TensorBoard not available - install with: pip install tensorboard]")
    
    # Create environment with or without rendering
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = state.astype(np.float32)
        episode_reward = 0
        steps = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.astype(np.float32)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, float(done or truncated))
            
            # Update agent
            if len(agent.memory) >= agent.batch_size:
                agent.update_parameters()
            
            episode_reward += reward
            steps += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        total_steps += steps
        
        # Print every episode
        stats = agent.get_training_stats()
        print(f"  Episode {episode+1:3d}/{num_episodes}: Reward={episode_reward:7.2f}, "
              f"Steps={steps:3d}, Alpha={stats['alpha']:.4f}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Reward/episode', episode_reward, episode)
            writer.add_scalar('Reward/avg_last_10', np.mean(episode_rewards[-10:]), episode)
            writer.add_scalar('Episode/length', steps, episode)
            writer.add_scalar('Episode/total_steps', total_steps, episode)
            writer.add_scalar('SAC/alpha', stats['alpha'], episode)
            writer.add_scalar('SAC/actor_loss', stats['avg_actor_loss'], episode)
            writer.add_scalar('SAC/critic1_loss', stats['avg_critic1_loss'], episode)
            writer.add_scalar('SAC/critic2_loss', stats['avg_critic2_loss'], episode)
            writer.add_scalar('SAC/alpha_loss', stats['avg_alpha_loss'], episode)
            writer.add_scalar('Memory/size', stats['memory_size'], episode)
    
    env.close()
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"\n  TensorBoard logs saved. View with: tensorboard --logdir={project_root / 'runs'}")
    
    # Evaluate improvement
    early_rewards = np.mean(episode_rewards[:10])
    late_rewards = np.mean(episode_rewards[-10:])
    improvement = late_rewards - early_rewards
    
    print(f"\n  Training Summary:")
    print(f"    Early avg reward (first 10): {early_rewards:.2f}")
    print(f"    Late avg reward (last 10):   {late_rewards:.2f}")
    print(f"    Improvement: {improvement:+.2f}")
    
    # LunarLander: >200 is solved, >0 is landing
    if late_rewards > 0:
        print(f"  ✓ Agent learned to land (reward > 0)")
    if late_rewards > 200:
        print(f"  ✓ Agent solved the environment (reward > 200)!")
    if improvement > 0:
        print(f"  ✓ Agent showed improvement during training")
    else:
        print(f"  ⚠ Agent did not show clear improvement (may need more episodes)")
    
    return episode_rewards, episode_lengths


def test_evaluation_performance(agent: SACAgent, env_name: str, num_episodes: int = 10, render: bool = False):
    """Test agent performance in evaluation mode (deterministic).
    
    Args:
        agent: Trained SAC agent
        env_name: Name of the Gymnasium environment
        num_episodes: Number of evaluation episodes
        render: If True, render the environment visually
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Evaluation on {env_name} ({num_episodes} episodes)")
    if render:
        print("  [Rendering enabled - close window to continue]")
    print("=" * 60)
    
    if not GYM_AVAILABLE:
        print("  ✗ Gymnasium not available")
        return 0.0, 0
    
    # Create environment with or without rendering
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    eval_rewards = []
    eval_lengths = []
    landings = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = state.astype(np.float32)
        episode_reward = 0
        steps = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            state = next_state.astype(np.float32)
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(steps)
        
        # Check if landed successfully (reward > 100 typically means good landing)
        if episode_reward > 100:
            landings += 1
        
        print(f"    Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}")
    
    env.close()
    
    avg_reward = np.mean(eval_rewards)
    avg_length = np.mean(eval_lengths)
    std_reward = np.std(eval_rewards)
    
    print(f"\n  Evaluation Results:")
    print(f"    Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"    Average Episode Length: {avg_length:.0f} steps")
    print(f"    Best Episode Reward: {max(eval_rewards):.2f}")
    print(f"    Worst Episode Reward: {min(eval_rewards):.2f}")
    print(f"    Successful Landings: {landings}/{num_episodes}")
    
    return avg_reward, avg_length


def test_save_load(agent: SACAgent, state_size: int, action_size: int):
    """Test model save and load functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: Save/Load Functionality")
    print("=" * 60)
    
    save_path = project_root / "outputs" / "test_sac_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get current stats
    original_stats = agent.get_training_stats()
    print(f"  Original training steps: {original_stats['training_steps']}")
    
    # Save model
    agent.save(str(save_path))
    
    if save_path.exists():
        print(f"  ✓ Model saved to {save_path}")
        file_size = save_path.stat().st_size / 1024
        print(f"    File size: {file_size:.1f} KB")
    else:
        print(f"  ✗ Failed to save model")
        return
    
    # Create new agent and load
    config_path = project_root / "sac_config" / "sac_config.yaml"
    new_agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        config_path=str(config_path)
    )
    
    new_agent.load(str(save_path))
    loaded_stats = new_agent.get_training_stats()
    
    print(f"  Loaded training steps: {loaded_stats['training_steps']}")
    
    if loaded_stats['training_steps'] == original_stats['training_steps']:
        print(f"  ✓ Training state preserved after load")
    else:
        print(f"  ✗ Training state mismatch")
    
    # Compare actions
    test_state = np.random.randn(state_size).astype(np.float32)
    original_action = agent.select_action(test_state, evaluate=True)
    loaded_action = new_agent.select_action(test_state, evaluate=True)
    
    if np.allclose(original_action, loaded_action, atol=1e-5):
        print(f"  ✓ Loaded model produces same actions")
    else:
        print(f"  ✗ Action mismatch after loading")
        print(f"    Original: {original_action}")
        print(f"    Loaded:   {loaded_action}")
    
    # Cleanup
    if save_path.exists():
        save_path.unlink()
        print(f"  ✓ Cleaned up test file")


def test_training_stats(agent: SACAgent):
    """Test training statistics tracking."""
    print("\n" + "=" * 60)
    print("TEST 6: Training Statistics")
    print("=" * 60)
    
    stats = agent.get_training_stats()
    
    print(f"  Training Steps: {stats['training_steps']}")
    print(f"  Alpha (temperature): {stats['alpha']:.4f}")
    print(f"  Avg Actor Loss: {stats['avg_actor_loss']:.6f}")
    print(f"  Avg Critic1 Loss: {stats['avg_critic1_loss']:.6f}")
    print(f"  Avg Critic2 Loss: {stats['avg_critic2_loss']:.6f}")
    print(f"  Avg Alpha Loss: {stats['avg_alpha_loss']:.6f}")
    print(f"  Memory Size: {stats['memory_size']}")
    
    if stats['training_steps'] > 0:
        print(f"  ✓ Training statistics are being tracked")
    else:
        print(f"  ⚠ No training steps recorded")


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n  Total: {passed}/{total} tests passed")


def main(env_name: str = None, render: bool = False, num_episodes: int = 100):
    """Run all SAC agent tests.
    
    Args:
        env_name: Optional environment name. If None, uses ENV_NAME constant.
        render: If True, render the environment during evaluation.
        num_episodes: Number of training episodes.
    """
    if env_name is None:
        env_name = ENV_NAME
    
    print("=" * 60)
    print("SAC AGENT PERFORMANCE TESTS")
    print(f"Environment: {env_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # Get environment dimensions dynamically
    try:
        state_size, action_size = get_env_dimensions(env_name)
        print(f"\nEnvironment dimensions:")
        print(f"  State size: {state_size}")
        print(f"  Action size: {action_size}")
        results['Environment Setup'] = True
    except Exception as e:
        print(f"\n✗ Failed to get environment dimensions: {e}")
        results['Environment Setup'] = False
        print_summary(results)
        return
    
    # Test 1: Initialization
    agent = test_agent_initialization(state_size, action_size)
    results['Initialization'] = agent is not None
    
    if agent is None:
        print("\n✗ Cannot continue without agent initialization")
        print_summary(results)
        return
    
    # Test 2: Action Selection
    try:
        test_action_selection(agent, state_size)
        results['Action Selection'] = True
    except Exception as e:
        print(f"  ✗ Action selection test failed: {e}")
        results['Action Selection'] = False
    
    # Test 3: Training Loop
    try:
        rewards, lengths = test_training_loop(agent, env_name=env_name, num_episodes=num_episodes, 
                                               render=render, use_tensorboard=True)
        if len(rewards) >= 20:
            improvement = np.mean(rewards[-10:]) - np.mean(rewards[:10])
            results['Training Loop'] = improvement > 0
        else:
            results['Training Loop'] = False
    except Exception as e:
        print(f"  ✗ Training loop test failed: {e}")
        results['Training Loop'] = False
    
    # Test 4: Evaluation Performance
    try:
        avg_reward, avg_length = test_evaluation_performance(agent, env_name=env_name, render=render)
        results['Evaluation'] = avg_reward > -200  # Not crashing immediately
    except Exception as e:
        print(f"  ✗ Evaluation test failed: {e}")
        results['Evaluation'] = False
    
    # Test 5: Save/Load
    try:
        test_save_load(agent, state_size, action_size)
        results['Save/Load'] = True
    except Exception as e:
        print(f"  ✗ Save/Load test failed: {e}")
        results['Save/Load'] = False
    
    # Test 6: Training Stats
    try:
        test_training_stats(agent)
        results['Training Stats'] = True
    except Exception as e:
        print(f"  ✗ Training stats test failed: {e}")
        results['Training Stats'] = False
    
    # Print summary
    print_summary(results)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def demo_environment(env_name: str = None, num_episodes: int = 3):
    """
    Quick demo to visualize the environment with an untrained agent.
    Useful for testing rendering without waiting for training.
    
    Args:
        env_name: Environment name
        num_episodes: Number of episodes to run
    """
    if env_name is None:
        env_name = ENV_NAME
    
    if not GYM_AVAILABLE:
        print("Gymnasium not available. Install with: pip install gymnasium")
        return
    
    print("=" * 60)
    print(f"DEMO: {env_name}")
    print("=" * 60)
    
    # Get dimensions
    state_size, action_size = get_env_dimensions(env_name)
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create agent
    config_path = project_root / "sac_config" / "sac_config.yaml"
    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        config_path=str(config_path)
    )
    
    # Create environment with rendering
    env = gym.make(env_name, render_mode="human")
    
    print(f"\nRunning {num_episodes} episodes (untrained agent)...")
    print("Close the window or press Ctrl+C to stop.\n")
    
    try:
        for episode in range(num_episodes):
            state, info = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            steps = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state.astype(np.float32)
            
            print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    finally:
        env.close()
    
    print("\nDemo complete.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SAC agent on Gymnasium environments')
    parser.add_argument('--env', type=str, default=None,
                        help=f'Environment name (default: {ENV_NAME}). '
                             'Examples: LunarLanderContinuous-v2, BipedalWalker-v3, Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--demo', action='store_true',
                        help='Quick demo mode: skip training and just visualize the environment')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_environment(env_name=args.env)
    else:
        main(env_name=args.env, render=args.render, num_episodes=args.episodes)
