"""Constants for HERON agent system.

This module centralizes all hard-coded constants used across the agent hierarchy
to avoid circular dependencies and provide a single source of truth.
"""

# =============================================================================
# Agent Hierarchy Levels
# =============================================================================
PROXY_LEVEL = 0      # Proxy is not part of the agent hierarchy (L1-L3)
FIELD_LEVEL = 1      # Level identifier for field-level agents
COORDINATOR_LEVEL = 2  # Level identifier for coordinator-level agents
SYSTEM_LEVEL = 3     # Level identifier for system-level agents


# =============================================================================
# Special Agent IDs
# =============================================================================
PROXY_AGENT_ID = "proxy_agent"
SYSTEM_AGENT_ID = "system_agent"


# =============================================================================
# Message Types (for proxy-agent communication)
# =============================================================================
MSG_GET_INFO = "get_info"
MSG_SET_STATE = "set_state"
MSG_SET_TICK_RESULT = "set_tick_result"
MSG_SET_STATE_COMPLETION = "set_state_completion"

# Response message types
MSG_GET_OBS_RESPONSE = "get_obs_response"
MSG_GET_GLOBAL_STATE_RESPONSE = "get_global_state_response"
MSG_GET_LOCAL_STATE_RESPONSE = "get_local_state_response"


# =============================================================================
# Info Request Types
# =============================================================================
INFO_TYPE_OBS = "obs"
INFO_TYPE_GLOBAL_STATE = "global_state"
INFO_TYPE_LOCAL_STATE = "local_state"


# =============================================================================
# State Types
# =============================================================================
STATE_TYPE_GLOBAL = "global"
STATE_TYPE_LOCAL = "local"


# =============================================================================
# Message Content Keys
# =============================================================================
MSG_KEY_BODY = "body"
MSG_KEY_PROTOCOL = "protocol"


# =============================================================================
# Default Configuration Values
# =============================================================================
DEFAULT_HISTORY_LENGTH = 100
DEFAULT_FIELD_TICK_INTERVAL = 1.0
DEFAULT_COORDINATOR_TICK_INTERVAL = 60.0
DEFAULT_SYSTEM_TICK_INTERVAL = 300.0


EMPTY_REWARD = 0.0