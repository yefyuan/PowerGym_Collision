"""System builder for collision detection case study.

Creates the agent hierarchy matching the original collision experiment:
- 3 microgrids (MG1, MG2, MG3)
- Each with ESS, DG, PV, WT devices
- Connected to IEEE 34-bus main grid
"""

from typing import Dict, Optional

from heron.agents.system_agent import SystemAgent
from heron.protocols.vertical import VerticalProtocol
from heron.scheduling.schedule_config import JitterType, ScheduleConfig

from powergrid.agents import Generator, ESS
from collision_case.collision_grid_agent import CollisionGridAgent


def create_collision_system(
    # Experiment configuration
    share_reward: bool = True,
    penalty: float = 10.0,
    # Scheduling configuration for async experiments
    enable_async: bool = False,
    field_tick_s: float = 5.0,
    coord_tick_s: float = 10.0,
    system_tick_s: float = 30.0,
    jitter_ratio: float = 0.1,
    # Training vs evaluation
    train: bool = True,
) -> SystemAgent:
    """Create system agent with 3 microgrids for collision experiments.

    Recreates the setup from the original collision platform:
    - MG1: ESS(2MWh), DG(0.66MW), PV(0.1MW), WT(0.1MW)
    - MG2: ESS(2MWh), DG(0.60MW), PV(0.1MW), WT(0.1MW)
    - MG3: ESS(2MWh), DG(0.50MW), PV(0.1MW), WT(0.1MW)

    Args:
        share_reward: Whether to share rewards across microgrids
        penalty: Safety penalty multiplier
        enable_async: Enable asynchronous updates with different tick rates
        field_tick_s: Field agent tick interval (seconds)
        coord_tick_s: Coordinator agent tick interval (seconds)
        system_tick_s: System agent tick interval (seconds)
        jitter_ratio: Timing jitter ratio for async mode
        train: Training mode flag

    Returns:
        SystemAgent with complete hierarchy
    """
    # Create scheduling configs for event-driven mode
    if enable_async:
        field_schedule = ScheduleConfig.with_jitter(
            tick_interval=field_tick_s,
            obs_delay=0.1,
            act_delay=0.2,
            msg_delay=0.1,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=jitter_ratio,
            seed=42,
        )
        coord_schedule = ScheduleConfig.with_jitter(
            tick_interval=coord_tick_s,
            obs_delay=0.2,
            act_delay=0.3,
            msg_delay=0.15,
            reward_delay=0.6,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=jitter_ratio,
            seed=43,
        )
        system_schedule = ScheduleConfig.with_jitter(
            tick_interval=system_tick_s,
            obs_delay=0.3,
            act_delay=0.5,
            msg_delay=0.2,
            reward_delay=1.0,
            jitter_type=JitterType.GAUSSIAN,
            jitter_ratio=jitter_ratio,
            seed=41,
        )
    else:
        field_schedule = None
        coord_schedule = None
        system_schedule = None

    # Build MG1
    mg1_devices = _create_mg1_devices(field_schedule)
    mg1 = CollisionGridAgent(
        agent_id="MG1",
        subordinates=mg1_devices,
        protocol=VerticalProtocol(),
        schedule_config=coord_schedule,
        penalty_weight=penalty,
    )

    # Build MG2
    mg2_devices = _create_mg2_devices(field_schedule)
    mg2 = CollisionGridAgent(
        agent_id="MG2",
        subordinates=mg2_devices,
        protocol=VerticalProtocol(),
        schedule_config=coord_schedule,
        penalty_weight=penalty,
    )

    # Build MG3
    mg3_devices = _create_mg3_devices(field_schedule)
    mg3 = CollisionGridAgent(
        agent_id="MG3",
        subordinates=mg3_devices,
        protocol=VerticalProtocol(),
        schedule_config=coord_schedule,
        penalty_weight=penalty,
    )

    # Build system agent
    system = SystemAgent(
        agent_id="system_agent",  # Must match HERON's expected ID
        subordinates={"MG1": mg1, "MG2": mg2, "MG3": mg3},
        schedule_config=system_schedule,
    )

    return system


def _create_mg1_devices(schedule_config: Optional[ScheduleConfig]) -> Dict:
    """Create devices for MG1.

    Devices:
    - ESS1: 0.5MW, 2MWh capacity
    - DG1: 0-0.66MW with quadratic cost
    - PV1: 0.1MW solar
    - WT1: 0.1MW wind

    Args:
        schedule_config: Scheduling configuration for devices

    Returns:
        Dict of {device_id: device_agent}
    """
    # Energy Storage
    ess1 = ESS(
        agent_id="MG1_ESS1",
        bus="MG1_bus",
        p_min_MW=-0.5,
        p_max_MW=0.5,
        capacity_MWh=2.0,
        soc_min=0.1,
        soc_max=0.9,
        init_soc=0.5,
        degr_cost_per_MWh=0.1,
        schedule_config=schedule_config,
    )

    # Diesel Generator with quadratic cost curve
    # Original: cost_curve_coefs=[100, 72.4, 0.5011]
    dg1 = Generator(
        agent_id="MG1_DG1",
        bus="MG1_bus",
        p_min_MW=0.0,
        p_max_MW=0.66,
        cost_curve_coefs=(0.5011, 72.4, 100.0),  # (a, b, c) for a*P^2 + b*P + c
        schedule_config=schedule_config,
    )

    # Solar PV
    pv1 = Generator(
        agent_id="MG1_PV1",
        bus="MG1_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="solar",
        cost_curve_coefs=(0.0, 0.0, 0.0),  # No fuel cost for renewables
        schedule_config=schedule_config,
    )

    # Wind Turbine
    wt1 = Generator(
        agent_id="MG1_WT1",
        bus="MG1_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="wind",
        cost_curve_coefs=(0.0, 0.0, 0.0),
        schedule_config=schedule_config,
    )

    return {
        "MG1_ESS1": ess1,
        "MG1_DG1": dg1,
        "MG1_PV1": pv1,
        "MG1_WT1": wt1,
    }


def _create_mg2_devices(schedule_config: Optional[ScheduleConfig]) -> Dict:
    """Create devices for MG2.

    Devices:
    - ESS1: 0.5MW, 2MWh capacity
    - DG1: 0-0.60MW with quadratic cost
    - PV1: 0.1MW solar
    - WT1: 0.1MW wind

    Args:
        schedule_config: Scheduling configuration for devices

    Returns:
        Dict of {device_id: device_agent}
    """
    # Energy Storage
    ess1 = ESS(
        agent_id="MG2_ESS1",
        bus="MG2_bus",
        p_min_MW=-0.5,
        p_max_MW=0.5,
        capacity_MWh=2.0,
        soc_min=0.1,
        soc_max=0.9,
        init_soc=0.5,
        degr_cost_per_MWh=0.1,
        schedule_config=schedule_config,
    )

    # Diesel Generator
    # Original: cost_curve_coefs=[100, 51.6, 0.4615]
    dg1 = Generator(
        agent_id="MG2_DG1",
        bus="MG2_bus",
        p_min_MW=0.0,
        p_max_MW=0.60,
        cost_curve_coefs=(0.4615, 51.6, 100.0),
        schedule_config=schedule_config,
    )

    # Solar PV
    pv1 = Generator(
        agent_id="MG2_PV1",
        bus="MG2_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="solar",
        cost_curve_coefs=(0.0, 0.0, 0.0),
        schedule_config=schedule_config,
    )

    # Wind Turbine
    wt1 = Generator(
        agent_id="MG2_WT1",
        bus="MG2_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="wind",
        cost_curve_coefs=(0.0, 0.0, 0.0),
        schedule_config=schedule_config,
    )

    return {
        "MG2_ESS1": ess1,
        "MG2_DG1": dg1,
        "MG2_PV1": pv1,
        "MG2_WT1": wt1,
    }


def _create_mg3_devices(schedule_config: Optional[ScheduleConfig]) -> Dict:
    """Create devices for MG3.

    Devices:
    - ESS1: 0.5MW, 2MWh capacity
    - DG1: 0-0.50MW with quadratic cost
    - PV1: 0.1MW solar
    - WT1: 0.1MW wind

    Args:
        schedule_config: Scheduling configuration for devices

    Returns:
        Dict of {device_id: device_agent}
    """
    # Energy Storage
    ess1 = ESS(
        agent_id="MG3_ESS1",
        bus="MG3_bus",
        p_min_MW=-0.5,
        p_max_MW=0.5,
        capacity_MWh=2.0,
        soc_min=0.1,
        soc_max=0.9,
        init_soc=0.5,
        degr_cost_per_MWh=0.1,
        schedule_config=schedule_config,
    )

    # Diesel Generator
    # Original: cost_curve_coefs=[100, 51.6, 0.4615]
    dg1 = Generator(
        agent_id="MG3_DG1",
        bus="MG3_bus",
        p_min_MW=0.0,
        p_max_MW=0.50,
        cost_curve_coefs=(0.4615, 51.6, 100.0),
        schedule_config=schedule_config,
    )

    # Solar PV
    pv1 = Generator(
        agent_id="MG3_PV1",
        bus="MG3_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="solar",
        cost_curve_coefs=(0.0, 0.0, 0.0),
        schedule_config=schedule_config,
    )

    # Wind Turbine
    wt1 = Generator(
        agent_id="MG3_WT1",
        bus="MG3_bus",
        p_min_MW=0.0,
        p_max_MW=0.1,
        gen_type="renewable",
        source="wind",
        cost_curve_coefs=(0.0, 0.0, 0.0),
        schedule_config=schedule_config,
    )

    return {
        "MG3_ESS1": ess1,
        "MG3_DG1": dg1,
        "MG3_PV1": pv1,
        "MG3_WT1": wt1,
    }
