"""Quick test to verify collision detection setup works.

Tests both shared and independent reward configurations with minimal iterations.
"""

import sys
from pathlib import Path

# Add parent directory to path
case_study_dir = Path(__file__).parent.parent
sys.path.insert(0, str(case_study_dir))

from collision_case import create_collision_system, CollisionEnv

def test_shared_reward():
    """Test shared reward configuration."""
    print("\n" + "="*60)
    print("Testing SHARED REWARD Configuration")
    print("="*60)
    
    # Create system
    system = create_collision_system(
        share_reward=True,
        penalty=10.0,
        enable_async=False,
    )
    
    # Create environment
    dataset_path = Path(__file__).parent.parent / "powergrid" / "data.pkl"
    env = CollisionEnv(
        system_agent=system,
        dataset_path=str(dataset_path),
        episode_steps=24,
        share_reward=True,
        penalty=10.0,
        log_path=None,  # No logging for quick test
    )
    
    # Reset
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful. Observation keys: {list(obs.keys())}")
    
    # Run a few steps
    print("\nRunning 5 test steps...")
    for step in range(5):
        # Random actions (only for device agents)
        actions = {}
        for agent_id in obs.keys():
            if "_" in agent_id:  # Device agents have underscore in ID
                agent = env.registered_agents.get(agent_id)
                if agent and hasattr(agent, 'action'):
                    actions[agent_id] = agent.action.sample()
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Compute total reward and collision count
        total_reward = sum(rewards.values())
        total_collisions = 0
        
        # Check collision stats
        for mg_id in ["MG1", "MG2", "MG3"]:
            if mg_id in env.registered_agents:
                mg_agent = env.registered_agents[mg_id]
                if hasattr(mg_agent, "get_collision_stats"):
                    stats = mg_agent.get_collision_stats()
                    total_collisions += stats.get("collision_flag", 0)
        
        print(f"  Step {step+1}: Reward={total_reward:6.2f}, Collisions={int(total_collisions)}")
        
        if terminated.get("__all__", False):
            print(f"  Episode ended at step {step+1}")
            break
    
    print("✓ Shared reward test completed!")
    return True


def test_independent_reward():
    """Test independent reward configuration."""
    print("\n" + "="*60)
    print("Testing INDEPENDENT REWARD Configuration")
    print("="*60)
    
    # Create system
    system = create_collision_system(
        share_reward=False,
        penalty=10.0,
        enable_async=False,
    )
    
    # Create environment
    dataset_path = Path(__file__).parent.parent / "powergrid" / "data.pkl"
    env = CollisionEnv(
        system_agent=system,
        dataset_path=str(dataset_path),
        episode_steps=24,
        share_reward=False,
        penalty=10.0,
        log_path=None,
    )
    
    # Reset
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful. Observation keys: {list(obs.keys())}")
    
    # Run a few steps
    print("\nRunning 5 test steps...")
    for step in range(5):
        # Random actions (only for device agents)
        actions = {}
        for agent_id in obs.keys():
            if "_" in agent_id:  # Device agents have underscore in ID
                agent = env.registered_agents.get(agent_id)
                if agent and hasattr(agent, 'action'):
                    actions[agent_id] = agent.action.sample()
        
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Show per-microgrid rewards
        mg_rewards = {k: v for k, v in rewards.items() if "MG" in k and "_" not in k}
        
        print(f"  Step {step+1}: Rewards={mg_rewards}")
        
        if terminated.get("__all__", False):
            print(f"  Episode ended at step {step+1}")
            break
    
    print("✓ Independent reward test completed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("COLLISION DETECTION QUICK TEST")
    print("="*60)
    
    try:
        # Test shared reward
        test_shared_reward()
        
        # Test independent reward
        test_independent_reward()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour collision detection setup is working correctly!")
        print("You can now run full training with:")
        print("  python -m collision_case.train_collision --num-iterations 10")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
