"""Debug script to check collision detection."""

import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from collision_case.system_builder import create_collision_system
from collision_case.collision_env import CollisionEnv

# Build system
print("Building collision system...")
system_agent = create_collision_system()
dataset_path = os.path.join(os.path.dirname(__file__), '../powergrid/data.pkl')

# Create environment
print("Creating environment...")
env = CollisionEnv(
    system_agent=system_agent,
    dataset_path=dataset_path,
    episode_steps=24,
    share_reward=True,
    penalty=10.0,
)

# Reset
print("\nResetting...")
obs, info = env.reset()
print(f"✓ Reset complete")

# Take one step with random actions
print("\nTaking step 1...")
actions = {}
# Use the system agent's action space dict
if hasattr(env, '_system_agent'):
    for agent_id, agent in env._system_agent.get_all_agents().items():
        if agent_id not in ['system_agent'] and hasattr(agent, 'action'):
            # Sample random action for each agent
            action_dim = agent.action.dim_c if hasattr(agent.action, 'dim_c') else 1
            actions[agent_id] = [0.0] * action_dim  # Zero action for safety

obs, rewards, terminated, truncated, infos = env.step(actions)

# Check collision in each microgrid
print("\n=== COLLISION DEBUG ===")
for mg_id in ['MG1', 'MG2', 'MG3']:
    mg_agent = env.registered_agents.get(mg_id)
    if mg_agent and hasattr(mg_agent, 'get_collision_stats'):
        stats = mg_agent.get_collision_stats()
        print(f"\n{mg_id}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    
    # Check if bus_voltages were passed
    if hasattr(mg_agent, 'state') and hasattr(mg_agent.state, '_last_kwargs'):
        print(f"  Last kwargs keys: {mg_agent.state._last_kwargs.keys() if hasattr(mg_agent.state, '_last_kwargs') else 'N/A'}")

# Check power flow results
print(f"\n=== POWER FLOW ===")
if env.net is not None:
    print(f"Converged: {env.net.get('converged', 'N/A')}")
    if hasattr(env.net, 'res_bus') and len(env.net.res_bus) > 0:
        print(f"Voltage range: {env.net.res_bus.vm_pu.min():.3f} - {env.net.res_bus.vm_pu.max():.3f}")
        print(f"MG1 Bus 645 voltage: {env.net.res_bus[env.net.bus.name == 'MG1 Bus 645'].vm_pu.values}")
        print(f"MG2 Bus 645 voltage: {env.net.res_bus[env.net.bus.name == 'MG2 Bus 645'].vm_pu.values}")
        print(f"MG3 Bus 645 voltage: {env.net.res_bus[env.net.bus.name == 'MG3 Bus 645'].vm_pu.values}")
    if hasattr(env.net, 'res_line') and len(env.net.res_line) > 0:
        print(f"Line loading range: {env.net.res_line.loading_percent.min():.1f}% - {env.net.res_line.loading_percent.max():.1f}%")

print("\n✓ Debug complete")
