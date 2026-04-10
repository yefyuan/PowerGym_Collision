# Heterogeneous Event-Driven Execution in Heron


---

## 1. The Core Idea in 30 Seconds

In **step-based mode** (training), all agents observe, decide, and act atomically in one `env.step()`. In **event-driven mode** (evaluation), that atomic step is broken into separate events with independent timing:

```
Step-based:   obs + decide + act  →  all happen at t=k, for all agents simultaneously
Event-driven: obs@t₁  →  decide@t₂  →  act-takes-effect@t₃  (each agent independently)
```

The real world clock keeps evolving between t₁ and t₃:
- Other agents act in between. 
- Messages arrive late.
- Faults strike mid-computation. 
  
This is where trained policies break.

---

## 2. Event Taxonomy

### 2.1 Conceptual Model — CPS Deployment Event Classes

Real CPS deployments produce events that fire for fundamentally different reasons. We classify them by **what determines when the event fires** — its timing mechanism — since that is what creates the deployment gap.

**Five event classes:**

#### 1. Periodic / scheduled events
Events that fire on a predictable cadence, driven by a timer or calendar.

| Timing character | CPS examples |
|---|---|
| Fixed interval ± jitter | Agent sensor poll (every 1s), SCADA scan cycle (every 2–4s), coordinator dispatch (every 60s), physics solver tick (every 300s) |
| Calendar-anchored | Market clearing (every 5 min), regulatory reporting deadline, shift handover |

**Why it matters**: These are the only events that step-based training models correctly (as uniform ticks). But even here, jitter and heterogeneous rates across agents create phase drift that step-based training doesn't see.

#### 2. Consequence events
Events that fire as a **delayed result** of a prior event. The delay is the key characteristic — it's the gap between cause and effect.

| Timing character | CPS examples |
|---|---|
| Fixed or stochastic delay after cause | Action takes effect on actuator (act_delay), observation delivered after sensor processing (obs_delay), message arrives at recipient (msg_delay), reward computed after physics settles (reward_delay) |

**Why it matters**: In step-based training, all consequences are instantaneous — observe, decide, act, effect all in one tick. At deployment, each is a separate event with its own delay. This is the primary source of **physics drift / observation staleness**: the world changes between cause and effect.

**Causal chains**: Consequence events form chains. A single agent tick triggers: obs request → obs delivery → decision → action effect → state update → reward. Each link adds delay. The end-to-end latency (obs to effect) determines how stale the decision is when it lands.

#### 3. Condition-triggered events
Events that fire when a **state condition** is met — a threshold breach, a constraint violation, or a pattern match.

| Timing character | CPS examples |
|---|---|
| Unpredictable, bursty, often urgent | Voltage violation alarm, overcurrent relay trip, battery SOC threshold, congestion detection, collision proximity alert |

**Why it matters**: These events are **not on any schedule** — they are driven by the evolving state of the system. A policy trained under periodic ticks has never seen "act immediately because a threshold was crossed." At deployment, condition-triggered events create situations where the agent must respond to a state it has never encountered at that point in its normal cycle.

**Interaction with other classes**: Condition-triggered events are often *caused by* physics progression (class 5) or exogenous disturbances (class 4), and they *cause* consequence events (class 2) — e.g., a fault (class 4) triggers a relay alarm (class 3), which triggers an immediate agent action whose effect is delayed (class 2).

#### 4. Exogenous disturbance events
Events from **outside the system boundary** that change the environment state discontinuously.

| Timing character | CPS examples |
|---|---|
| Stochastic, may be correlated | Line fault (topology change), sudden load spike, cloud cover reducing solar generation, weather front, vehicle accident on road, cyber-attack |

**Why it matters**: These are fundamentally different from smooth physics progression — they change the system **discontinuously**. A topology change invalidates any action computed for the pre-fault network. Step-based training can model disturbances (via domain randomization), but cannot model the *interaction* between a disturbance's timing and an agent's event-driven pipeline — e.g., a fault arriving during the agent's compute-to-effect delay.

#### 5. Physics progression events
Events representing the **continuous evolution** of the physical environment on its own schedule.

| Timing character | CPS examples |
|---|---|
| Continuous or periodic (solver-driven) | Power flow solving, traffic simulation step, robot dynamics integration, thermal evolution, battery charge/discharge |

**Why it matters**: Physics doesn't wait for agents. Between any two agent events, the environment state evolves. The physics tick rate relative to the agent tick rate determines how much the world changes between an agent's observation and its action effect. This is the engine behind **physics drift**.

---

**How the classes chain together in a real scenario:**

```
[Class 4] Line fault occurs at t=12.3s                   (exogenous disturbance)
    → [Class 5] Physics solver runs at t=15.0s            (scheduled physics — sees new topology)
    → [Class 3] Voltage violation detected at t=15.0s     (condition trigger — V < 0.95 pu)
    → [Class 1] Responsible agent's next tick at t=16.0s  (periodic — may be too late)
    → [Class 2] Agent's action effect at t=16.5s          (consequence — delayed by 0.5s)
```

In step-based training, all of this collapses to one tick. In event-driven evaluation, the 4.2-second gap (fault at 12.3 → effect at 16.5) is where performance degrades.

**Step-based training collapses all five classes into one synchronous tick.** Event-driven evaluation separates them, exposing the deployment gap.

### 2.2 Implementation — Heron Event Types

Heron's scheduler event types mapped to the 5-class model:

| Event Type | Class | What It Does | Priority | Status |
|---|---|---|---|---|
| `AGENT_TICK` | 1 — Periodic | Agent wakes on periodic timer, starts observe→act cycle | 0 | Implemented |
| `OBSERVATION_READY` | 2 — Consequence | Delayed observation becomes available to agent | 1 | Implemented |
| `ACTION_EFFECT` | 2 — Consequence | Agent's action is applied to environment after delay | 0 | Implemented |
| `MESSAGE_DELIVERY` | 2 — Consequence | Message arrives at recipient after network latency | 2 | Implemented |
| `SIMULATION` | 5 — Physics | Physics engine runs, updates global state | 1 | Implemented |
| `ENV_UPDATE` | 4 — Exogenous | Exogenous disturbance (fault, load change) | — | **Placeholder** — defined in enum, no handlers |
| `CUSTOM` | 3 — Condition | Intended for condition-triggered events | — | **Placeholder** — defined in enum, no handlers |

**Per-class assessment:**

#### Class 1 (Periodic) — Adequate

`AGENT_TICK` covers the core scenarios: periodic sensor polls, coordinator dispatch cycles, and heterogeneous rates across agent tiers. Jitter support (UNIFORM/GAUSSIAN) models realistic clock drift. Calendar-anchored events (market clearing, shift handover) can be approximated by aligning `tick_interval` to specific offsets. No changes needed.

#### Class 2 (Consequence) — Adequate with refinements

`OBSERVATION_READY`, `ACTION_EFFECT`, and `MESSAGE_DELIVERY` cover the main consequence chain (obs → decision → action effect → message delivery). Two refinements would improve fidelity:

- **Per-channel message delays**: `MESSAGE_DELIVERY` currently uses a single `msg_delay` for all message types — obs requests to the Proxy (local bus, ~10ms), coordination directives between agents (SCADA network, ~200ms), and state writes. In real CPS these traverse different infrastructure with different latencies. Fix: expand `ScheduleConfig` with per-channel delays (`obs_msg_delay`, `coord_msg_delay`) and make the delay lookup infer channel type from sender/recipient.

- **State write latency**: After `ACTION_EFFECT`, the agent writes its updated state to the Proxy synchronously (instant). In real SCADA, writes to a shared state store incur latency — other agents don't see the update until it's committed. Fix: route Proxy state writes through `MESSAGE_DELIVERY` with a `state_write_delay` parameter, using the same event mechanism as reads.

Neither refinement requires a new event type — they expand `ScheduleConfig` and make `MESSAGE_DELIVERY` delay-aware by channel.

#### Class 3 (Condition-triggered) — Not implemented

`CUSTOM` is a placeholder with no handlers, no condition evaluation mechanism, and no way to fire events based on state conditions. The entire class is missing. This is a **major gap** — condition-triggered events (voltage alarms, relay trips, threshold breaches) are the primary reactive mechanism in CPS and create evaluation scenarios that periodic ticks cannot approximate.

**Required changes**: See Section 2.3 Gap A.

#### Class 4 (Exogenous) — Not implemented

`ENV_UPDATE` is a placeholder with no handlers. The `pre_step()` hook provides a synchronous workaround for injecting disturbances, but only at simulation boundaries — disturbances cannot arrive between agent ticks, which is exactly when they matter most (a fault during an agent's compute-to-effect delay). This is a **major gap** — exogenous disturbances are the highest-stakes evaluation scenario for CPS.

**Required changes**: See Section 2.3 Gap B.

#### Class 5 (Physics) — Adequate

`SIMULATION` covers periodic physics solving across CPS domains (power flow, traffic sim, dynamics integration). The physics rate is independently configurable via SystemAgent's `tick_interval`. One interaction needs implementation: `ENV_UPDATE` (Class 4) should optionally trigger an immediate `SIMULATION` when a disturbance changes the physical state (e.g., a line fault requires re-solving power flow). This is addressed as part of Gap B.

**Priority rule**: When events share a timestamp, lower priority number fires first: action effects settle (0) → physics runs (1) → messages propagate (2).

---

### 2.3 Gaps — Required Changes

Two structural gaps remain. Both are in Classes 3 and 4, and they are coupled: exogenous disturbances (Class 4) feed condition triggers (Class 3). Closing them requires coordinated implementation.

#### Gap A: Condition-triggered events (Class 3)

**The problem**: Agents only wake on periodic timers. In real CPS, agents also wake on condition triggers — a voltage violation, a relay trip, a threshold breach. These are unpredictable, bursty, and often more urgent than periodic ticks. A policy trained under periodic ticks has never seen "act immediately because a condition was crossed."

**Required changes**:

1. **Rename `CUSTOM` → `CONDITION_TRIGGER`** in `EventType` enum. This gives the event proper semantics instead of being a generic catch-all.

2. **`ConditionMonitor` subsystem** — a lightweight mechanism that evaluates conditions and fires events:
   ```python
   @dataclass
   class ConditionMonitor:
       monitor_id: str                 # unique identifier
       agent_id: AgentID              # who gets woken up
       condition_fn: Callable          # (proxy_state) -> bool
       cooldown: float = 0.0          # min seconds between triggers (prevents spam)
       one_shot: bool = False          # fire once then deregister
       preempt_next_tick: bool = False # cancel agent's next scheduled AGENT_TICK
   ```
   Monitors are registered on the scheduler via `scheduler.register_condition(monitor)`.

3. **Evaluation hook**: After every `SIMULATION` event and every `ENV_UPDATE` event, the scheduler evaluates all registered conditions against the current Proxy state. If a condition is met (and cooldown has elapsed), it schedules a `CONDITION_TRIGGER` event for the target agent.

4. **Agent handler**: Each agent class implements `@Agent.handler("condition_trigger")` to define how it responds — run policy immediately (reactive response), flag for next tick (deferred), or custom logic.

5. **`preempt_next_tick` option**: When `True`, a fired `CONDITION_TRIGGER` cancels the agent's next scheduled `AGENT_TICK` (requires `cancel_event()` on the scheduler — see below). When `False`, the trigger fires *in addition to* the normal periodic tick. Both patterns exist in real CPS.

**Why evaluate after SIMULATION and ENV_UPDATE, not after every event**: Conditions depend on physical state (voltage, current, SOC). Physical state changes meaningfully only when physics runs (`SIMULATION`) or a disturbance arrives (`ENV_UPDATE`). Evaluating after every `MESSAGE_DELIVERY` would be wasteful and semantically wrong.

#### Gap B: Exogenous disturbance events (Class 4)

**The problem**: `ENV_UPDATE` exists in the enum but has no handlers. Real CPS environments experience discrete, exogenous state changes — line faults, load spikes, renewable drops, weather transitions — that change the system discontinuously and can arrive at any point on the timeline, including mid-cycle for an agent.

**Required changes**:

1. **Implement `ENV_UPDATE` with handler on SystemAgent** (or a dedicated `DisturbanceHandler`):
   ```python
   scheduler.schedule_env_update(
       timestamp=12.3,
       payload={
           "type": "line_fault",
           "element": "line_7_8",
           "action": "disconnect",
       }
   )
   ```
   The handler modifies environment state (topology, loads, generation) directly via the physics backend.

2. **`DisturbanceSchedule`** — a configurable schedule of exogenous events, injected at episode start:
   ```python
   # Deterministic (scripted scenario)
   DisturbanceSchedule([
       Disturbance(t=12.3, type="line_fault", element="line_7_8"),
       Disturbance(t=45.0, type="load_spike", bus="bus_5", delta_kw=500),
   ])

   # Stochastic (sampled per episode)
   DisturbanceSchedule.poisson(rate=0.1, types=["line_fault", "load_spike"])
   ```
   The schedule is consumed by the scheduler at `reset()`, which enqueues all `ENV_UPDATE` events onto the timeline.

3. **Optional immediate `SIMULATION` trigger**: Some disturbances change physical state in ways that require the physics solver to re-run (e.g., a topology change requires re-solving power flow). The `Disturbance` dataclass includes a `requires_physics: bool` flag. When `True`, the `ENV_UPDATE` handler schedules an immediate `SIMULATION` event after applying the disturbance.

4. **Condition re-evaluation after `ENV_UPDATE`**: After an `ENV_UPDATE` fires (and optional `SIMULATION` completes), the `ConditionMonitor` subsystem (Gap A) re-evaluates all registered conditions. This creates the causal chain:
   ```
   ENV_UPDATE (fault) → SIMULATION (re-solve) → conditions evaluated → CONDITION_TRIGGER (alarm) → agent responds
   ```

#### Supporting change: `cancel_event()` on scheduler

Both gaps benefit from the ability to cancel pending events:
- Gap A: `preempt_next_tick` needs to cancel a scheduled `AGENT_TICK`
- Gap B: A disturbance might invalidate a pending `ACTION_EFFECT` (e.g., actuator destroyed by fault)

**Required change**: Add `cancel_event(event_id)` or `cancel_events(agent_id, event_type)` to `EventScheduler`. Events get a unique `event_id` assigned at scheduling time. Cancellation marks the event as void (lazy deletion — event stays in heap but is skipped when popped).

#### Remaining minor gaps (approximate with current tools)

These do not require new event types — they can be approximated with current mechanisms or addressed as future work:

| Gap | Workaround | Good enough? |
|---|---|---|
| Per-channel message delays | Use single `msg_delay` tuned to dominant channel | Acceptable for initial case studies; expand `ScheduleConfig` later |
| State write latency | Instant writes (optimistic) | Acceptable; real write latency is small relative to other delays |
| Sensor/actuator failure | Static masking or manual feature zeroing | Rough; can model via `ENV_UPDATE` + condition that disables features |
| Batched communication | Set `msg_delay` to batch period | Acceptable; doesn't aggregate but captures timing |
| Calendar-anchored schedules | Align `tick_interval` to calendar period | Acceptable |

---

### 2.4 Summary: What You Can Ablate Now vs. After Closing Gaps

| Deployment phenomenon | Event classes | Failure mode | Status | Controlled by |
|---|---|---|---|---|
| Action delay / physics drift | 1 × 2 × 5 | **5a** Chain breakage | **Now** | `act_delay` |
| Observation staleness | 1 × 2 × 5 | **5a** Chain breakage | **Now** (emergent) | `msg_delay`, `tick_interval` ratios |
| Agent/physics rate mismatch | 1 × 5 | **5a** Chain breakage | **Now** | Agent vs SystemAgent `tick_interval` |
| Scheduling jitter | 1 | **5a** Chain breakage | **Now** | `jitter_ratio`, `jitter_type` |
| Message latency | 2 | **5a** Chain breakage | **Now** | `msg_delay` |
| Disturbance during pipeline | 4 × 2 | **5b** Disturbance collision | **After Gap B** | `DisturbanceSchedule` + `ENV_UPDATE` |
| Fault + delay interaction | 4 × 2 × 5 | **5b** Disturbance collision | **After Gap B** | `ENV_UPDATE` timing relative to agent cycle |
| Disturbance-triggered physics | 4 → 5 | **5b** Disturbance collision | **After Gap B** | `requires_physics` flag on `Disturbance` |
| Condition-triggered response | 3 vs 1 | **5c** Reactive vs periodic | **After Gap A** | `ConditionMonitor` + `CONDITION_TRIGGER` |
| Preemption of periodic tick | 3 vs 1 | **5c** Reactive vs periodic | **After Gap A** | `preempt_next_tick` + `cancel_event()` |
| Tier-dependent response gap | 3 vs 1 | **5c** Reactive vs periodic | **After Gap A** | Compare across agent tiers |

> Implementation plan for closing these gaps: see [`impl_todo_2026_03_20.md`](impl_todo_2026_03_20.md)

---

## 3. ScheduleConfig — The Deployment Parameters

Each agent gets its own `ScheduleConfig`. These are the **independent variables** for ablation — all Class 1 (periodic) and Class 2 (consequence) parameters.

```python
ScheduleConfig(
    tick_interval = 1.0,       # seconds between periodic wakeups (Class 1)
    obs_delay     = 0.0,       # observation round-trip latency (Class 2)
    act_delay     = 0.15,      # time from decision to action effect (Class 2)
    msg_delay     = 0.05,      # inter-agent message latency, one-way (Class 2)
    reward_delay  = 0.0,       # delay before reward computation (Class 2)
    jitter_type   = "gaussian",  # NONE | UNIFORM | GAUSSIAN
    jitter_ratio  = 0.1,      # magnitude as fraction of base (±10%)
)
```

**Typical values by agent tier** (power systems / SCADA):

| Parameter | FieldAgent | CoordinatorAgent | SystemAgent |
|---|---|---|---|
| `tick_interval` | 1.0 s | 60.0 s | 300.0 s |
| `obs_delay` | 0.0 s | 0.0 s | 0.0 s |
| `act_delay` | 0.1–0.5 s | 0.0 s (no actuator) | 0.0 s |
| `msg_delay` | 0.05 s | 0.1 s | 0.2 s |
| `jitter_ratio` | 0.1 (10%) | 0.05 (5%) | 0.02 (2%) |

> **Key insight**: Observation staleness is *emergent*, not configured directly. It arises from: (a) the Proxy serving cached state that was last written at some earlier time, and (b) the message round-trip delay (2 × `msg_delay`). A CoordinatorAgent observing a FieldAgent that last updated 5s ago sees state that is 5s stale — plus the round-trip.

**What ScheduleConfig does NOT control** (yet): Class 3 (condition triggers) and Class 4 (exogenous disturbances) are configured separately — via `ConditionMonitor` registration and `DisturbanceSchedule` injection. See [`impl_todo_2026_03_20.md`](impl_todo_2026_03_20.md).

---

## 4. Execution Flow

### 4.1 Normal cycle (Classes 1, 2, 5 only)

The 8-phase sequence that repeats each episode under current implementation. Understanding this is critical for knowing **where** degradation enters.

```
Phase 1: SystemAgent TICK (Class 1)
    └─ Schedules CoordinatorAgent ticks
    └─ Sends obs request to Proxy (MESSAGE_DELIVERY, delay = msg_delay_sys)

Phase 2: SystemAgent gets observation back (Class 2)
    └─ Proxy responds with visibility-filtered state (delay = msg_delay_sys)
    └─ SystemAgent runs policy → produces action
    └─ Decomposes action via Protocol → publishes to MessageBroker

Phase 3: CoordinatorAgent TICK (Class 1)
    └─ Consumes upstream action from broker
    └─ Schedules FieldAgent ticks
    └─ Sends obs request to Proxy

Phase 4: FieldAgent TICK (Class 1)
    └─ Consumes upstream action from broker
    └─ Sends obs request to Proxy
    └─ Gets obs back → runs policy → schedules ACTION_EFFECT

Phase 5: FieldAgent ACTION_EFFECT (Class 2)       ← action delay hits here
    └─ Action applied to local state
    └─ Updated state written to Proxy cache

Phase 6: SIMULATION (Class 5)                      ← physics runs here
    └─ SystemAgent requests global state from Proxy
    └─ Runs physics engine (e.g., pandapower)
    └─ Converts results → writes back to Proxy

Phase 7: Reward cascade (Class 2)
    └─ Proxy signals state update complete
    └─ Each agent requests local state (with reward_delay)
    └─ Computes reward → sends tick result to EpisodeAnalyzer

Phase 8: Next cycle
    └─ SystemAgent schedules its next AGENT_TICK
    └─ Repeat
```

### 4.2 Disrupted cycle (after Gaps A+B are closed)

When Classes 3 and 4 are implemented, the normal cycle can be interrupted at any point:

```
Phase 1–4: Normal cycle in progress...

  ⚡ ENV_UPDATE fires (Class 4) — e.g., line fault at t=12.3
    └─ SystemAgent.env_update_handler applies disturbance
    └─ requires_physics=True → immediate SIMULATION (Class 5)
    └─ Physics re-solves with new topology
    └─ ConditionMonitor evaluates: V_bus5 = 0.91 < 0.95 (Class 3)
    └─ CONDITION_TRIGGER scheduled for field_agent_bus_5
    └─ field_agent_bus_5 wakes immediately, requests obs
    └─ Obs arrives (Class 2, delay=msg_delay) → runs policy
    └─ ACTION_EFFECT scheduled (Class 2, delay=act_delay)

Phase 5–8: Normal cycle resumes (may overlap with reactive response)
```

The key difference: the agent responds to a *condition* rather than its *timer*. The observation it sees is post-fault, the action it computes is for the disrupted system, and the delay between condition trigger and action effect is where the new class of degradation lives.

---

## 5. Failure Modes to Measure

The paper claims **heterogeneous** event-driven execution matters — not just asynchrony, but the interaction between fundamentally different event types with different timing mechanisms. Each failure mode below involves a distinct combination of event classes. Together they span the full five-class taxonomy.

**Litmus test**: Would this failure mode occur in a system where all events are the same type, just at different times? If yes → it's an asynchrony problem, not a heterogeneity problem. All three modes below pass this test.

Mode 5a is measurable now (Classes 1, 2, 5). Modes 5b and 5c require Gap A/B implementation.

### 5a. Consequence Chain Breakage (Class 1 × 2 × 5)

**What happens**: The obs→decide→act→effect chain spans *multiple event types*: the agent wakes on a periodic tick (Class 1), the observation and action effect are consequence events with fixed delays (Class 2), and physics progresses on its own solver-driven schedule (Class 5). Between any two links in this chain, events of *other types* can intervene — most critically, physics advancing the state the action was computed for.

**The heterogeneity**: Physics events and action-effect events are different types on independent schedules. Physics doesn't wait for the agent's consequence chain to complete. The failure isn't just "delay" — it's that the chain crosses event-type boundaries, and at each crossing a different class of event can invalidate the premise.

**Where in the flow**: Between Phase 4 (obs delivered) and Phase 5 (action effect). The total staleness is `2 × msg_delay + act_delay`. During this window, physics (Class 5) may run, other agents' action effects (Class 2) may land, and the state the action was computed for drifts.

**How to measure**:
- **Metric**: `staleness = t_action_effect - t_observation`; correlate with reward degradation
- **Ablation**: Sweep `act_delay` ∈ {0, 0.1, 0.25, 0.5, 1.0} with other params fixed. Also sweep the agent/physics tick ratio (agent `tick_interval` vs SystemAgent `tick_interval`) to test how physics rate amplifies the chain breakage.
- **Expected**: Monotonic degradation. Steeper when physics rate is high relative to agent rate (more physics updates intervene in the chain).

### 5b. Disturbance–Pipeline Collision (Class 4 × 2)

**What happens**: An exogenous disturbance (Class 4) — a line fault, load spike, topology change — arrives while an agent is mid-pipeline, between observation and action effect. The agent's pending action was computed for the pre-disturbance world. Unlike 5a (gradual drift), the state change is *discontinuous*: the action doesn't land on a "slightly older" state, it lands on a *qualitatively different system* (e.g., post-fault topology where the pre-fault action may be harmful).

**The heterogeneity**: Exogenous events (Class 4) have a fundamentally different timing mechanism from consequence events (Class 2) — they are external and unpredictable, while consequence events are cause-driven with fixed delays. The severity depends on WHERE in the consequence chain the disturbance hits: early (agent can potentially re-observe) vs. late (action is already committed, effect is pending). This interaction between unpredictable and structured event types doesn't exist in step-based execution, where disturbances are applied at step boundaries and agents always see the post-disturbance state before acting.

**Where in the flow**: `ENV_UPDATE` fires between Phase 4 (agent decided) and Phase 5 (action effect), or between Phase 5 (action effect) and Phase 6 (physics). The worst case: disturbance arrives just after an agent commits its action but before the effect fires.

**How to measure** (requires Gap B):
- **Metric**: Compare 4 conditions in a factorial design:

  | | No disturbance | With disturbance |
  |---|---|---|
  | **Step-based** | A (baseline) | C |
  | **Event-driven** | B | D |

  The **interaction effect** = D − (B + C − A). If positive, disturbances and event-driven timing *compound* — the combination is worse than the sum of parts.
- **Ablation**: Inject disturbances at controlled times relative to the agent's event chain. Vary `act_delay` to widen/narrow the vulnerability window.
- **Current workaround**: Use `pre_step()` for fault injection. Approximate — disturbance timing is synchronous, so it can't land between agent events.
- **Expected**: Compounding degradation. Larger `act_delay` = wider window = higher probability that a disturbance invalidates a pending action.

### 5c. Reactive vs. Periodic Response Gap (Class 3 vs. 1)

**What happens**: A condition (e.g., voltage violation) occurs. The critical question: how does the agent *discover* it and how fast can it *respond*? In step-based execution, both are trivial — the agent sees the condition at the next step and acts immediately. In heterogeneous event-driven execution, there are two fundamentally different event types competing:
- **Periodic response** (Class 1): The agent discovers the condition at its next scheduled tick, which could be up to a full `tick_interval` later. For a coordinator ticking every 60s, this is a 60-second blind spot.
- **Reactive response** (Class 3): A condition trigger fires immediately when the condition is met, waking the agent outside its periodic schedule. But the agent still incurs the consequence chain (Class 2) — obs request, delivery, compute, action effect — before its corrective action takes effect.

**The heterogeneity**: The failure mode is specifically about two different *discovery mechanisms* (state-driven vs. clock-driven) for the same condition. It doesn't exist in step-based execution (where discovery is always "at the next step"), and it doesn't exist in a system with only one event type. The gap between reactive and periodic response quantifies the value of condition-triggered events — a pure heterogeneity measure.

**Where in the flow**: A `CONDITION_TRIGGER` fires after a `SIMULATION` or `ENV_UPDATE` detects a threshold breach. The agent's response traverses the same consequence chain as a normal tick, but starts immediately rather than waiting for the next `AGENT_TICK`.

**How to measure** (requires Gap A):
- **Metric**: (1) `response_time = t_corrective_action_effect - t_condition_onset`; (2) `periodic_response_time = t_next_tick + consequence_chain - t_condition_onset`; (3) the gap between them
- **Ablation**: Compare `preempt_next_tick=True` (condition trigger replaces next periodic tick) vs `False` (condition trigger fires *in addition to* periodic tick). Also compare across agent tiers — the gap matters most for slow-ticking agents (coordinators at 60s).
- **Expected**: Reactive response is up to one `tick_interval` faster than periodic. The improvement matters most for high-stakes, time-sensitive conditions (voltage violations, relay trips) on slow-ticking agents.

---

### Summary: How the three failure modes span the taxonomy

| Failure mode | Event classes involved | Core mechanism | Measurable |
|---|---|---|---|
| 5a: Chain breakage | 1 × 2 × 5 | Gradual drift between predictable event types | **Now** |
| 5b: Disturbance collision | 4 × 2 | Discontinuous state change during committed pipeline | **After Gap B** |
| 5c: Reactive vs periodic | 3 vs 1 | Different discovery mechanisms for the same condition | **After Gap A** |

5a = the world *drifts* while you act (predictable events interleave).
5b = the world *breaks* while you act (unpredictable event hits structured pipeline).
5c = *how you discover* something happened depends on which event type alerts you.

---

## 6. Ablation Experiment Design

### Baseline
Train with MAPPO/QMIX in step-based mode. Record step-based evaluation score (= "idealized" performance).

### Tier 1: Consequence chain breakage — single-parameter sweeps (5a, available now)

Hold all other parameters at zero/default while sweeping one. These isolate individual links in the cross-type chain (Classes 1 × 2 × 5).

| Experiment | Parameter | Values | What it tests |
|---|---|---|---|
| E1: Action delay | `act_delay` | 0, 0.1, 0.25, 0.5, 1.0 s | Length of the Class 2 consequence chain; how much physics (Class 5) drifts during it |
| E2: Message latency | `msg_delay` | 0, 0.05, 0.1, 0.5, 1.0 s | Observation round-trip delay; widens the chain's exposure to physics intervention |
| E3: Agent/physics rate | agent `tick_interval` / SystemAgent `tick_interval` | 1:1, 1:10, 1:60, 1:300 | How many physics updates (Class 5) intervene in the agent's consequence chain (Class 2) |
| E4: Scheduling jitter | `jitter_ratio` | 0, 0.05, 0.1, 0.2, 0.3 | Unpredictability in when the chain starts; varies action ordering across agents |

### Tier 2: Cross-type interaction experiments (5a pairwise, available now)

Pairwise combinations to test whether cross-type interactions compound or substitute.

| Experiment | Parameters | Question |
|---|---|---|
| E5: Delay × physics rate | `act_delay` × agent/physics tick ratio | Does faster physics amplify the chain breakage from action delay? |
| E6: Latency × jitter | `msg_delay` × `jitter_ratio` | Does jitter make message-timing races worse? |

### Tier 3: Disturbance–pipeline collision (5b, after Gap B)

Tests the interaction between unpredictable exogenous events (Class 4) and the structured consequence chain (Class 2).

| Experiment | Setup | Question |
|---|---|---|
| E7: Fault × delay | `DisturbanceSchedule` × `act_delay` | Does action delay widen the vulnerability window for disturbances? |
| E8: Fault timing | Inject same fault at different points in the agent's event chain | Does severity depend on WHERE in the chain the disturbance hits? |
| E9: Factorial | No disturbance × With disturbance × Step-based × Event-driven | Is the disturbance × event-driven interaction super-additive? (report interaction term D − (B + C − A)) |

### Tier 4: Reactive vs. periodic response (5c, after Gap A)

Tests the value of condition-triggered events (Class 3) vs. periodic ticks (Class 1).

| Experiment | Setup | Question |
|---|---|---|
| E10: Response time | `ConditionMonitor` with reactive trigger vs. wait-for-next-tick | How much faster is reactive response? |
| E11: Preemption | `preempt_next_tick` True vs False | Does preempting the periodic tick improve outcomes? |
| E12: Tier sensitivity | Compare across agent tiers (field 1s, coordinator 60s, system 300s) | Where does reactive response matter most? (expected: slow-ticking agents) |

### Realistic deployment profile
Set all parameters to domain-realistic values simultaneously (SCADA timing for power, intersection timing for traffic), including disturbances and condition monitors. This gives the "deployment evaluation score" — the single number that quantifies the step-to-deployment gap.

### What to report
For each experiment:
1. **Step-based score** (baseline)
2. **Event-driven score** (per parameter value)
3. **Gap** = (step-based − event-driven) / step-based × 100%
4. **Per-parameter decomposition**: Which parameter contributes what fraction of the total gap
5. **Cross-type interaction effects**: Are pairwise gaps additive or compounding? (report the interaction term)
6. **Failure mode attribution**: Map observed degradation to specific failure modes (5a/5b/5c) based on which event class interactions caused it

---

## 7. Running the Two Modes

### Current (Classes 1, 2, 5)

```python
# ── Step-based (training) ──
obs, _ = env.reset()
for step in range(num_steps):
    actions = {aid: policy(obs[aid]) for aid in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

# ── Event-driven (evaluation) ──
obs, _ = env.reset()
env.set_agent_policies({aid: policy for aid in env.agents})
result = env.run_event_driven(t_end=30.0)
# result contains per-agent rewards, event logs, staleness metrics
```

No code changes to agents or policies between modes. The only difference is the `ScheduleConfig` attached to each agent.

### After Gaps A+B (adds Classes 3, 4)

```python
from heron.scheduling.disturbance import DisturbanceSchedule, Disturbance
from heron.scheduling.condition_monitor import ConditionMonitor

# ── Register condition monitors ──
scheduler.register_condition(ConditionMonitor(
    monitor_id="v_violation_bus5",
    agent_id="field_agent_bus_5",
    condition_fn=lambda state: state["bus_5"]["vm_pu"] < 0.95,
    cooldown=5.0,
    preempt_next_tick=False,
))

# ── Create disturbance schedule ──
schedule = DisturbanceSchedule([
    Disturbance(t=12.3, disturbance_type="line_fault",
                payload={"element": "line_7_8", "action": "disconnect"}),
])

# ── Event-driven evaluation with disturbances ──
obs, _ = env.reset(disturbance_schedule=schedule)
env.set_agent_policies({aid: policy for aid in env.agents})
result = env.run_event_driven(t_end=30.0)
```

Same agents, same policies, same physics — now with exogenous disturbances and condition-triggered responses on the timeline.

---

## 8. Key Code Locations

### Current

| What | File |
|---|---|
| Event types + Event dataclass | `heron/scheduling/event.py` |
| EventScheduler (heap, dispatch) | `heron/scheduling/scheduler.py` |
| ScheduleConfig (timing params) | `heron/scheduling/schedule_config.py` |
| Agent base + `@handler` decorator | `heron/agents/base.py` |
| FieldAgent handlers | `heron/agents/field_agent.py` |
| CoordinatorAgent handlers | `heron/agents/coordinator_agent.py` |
| SystemAgent + simulation driver | `heron/agents/system_agent.py` |
| Proxy (state cache, visibility) | `heron/agents/proxy_agent.py` |
| Protocols (vertical, horizontal) | `heron/protocols/vertical.py`, `horizontal.py` |
| MessageBroker | `heron/messaging/in_memory_broker.py` |
| Dual-mode env (step vs event-driven) | `heron/envs/base.py` |

### After Gaps A+B (new files)

| What | File |
|---|---|
| ConditionMonitor dataclass | `heron/scheduling/condition_monitor.py` |
| Disturbance + DisturbanceSchedule | `heron/scheduling/disturbance.py` |

---

## 9. Checklist for Case Study Implementers

### Setup
- [ ] **Physics backend**: Implement 3 bridge methods (reset, step, state conversion)
- [ ] **Agent definitions**: Define Features (with visibility scopes), reward logic, state update
- [ ] **ScheduleConfig**: Set domain-realistic timing values per agent tier (Section 3)
- [ ] **Protocol**: Choose vertical (hierarchical) or horizontal (peer-to-peer)
- [ ] **Disturbance scenarios** (after Gap B): Implement `apply_disturbance()` on your env, define `DisturbanceSchedule`
- [ ] **Condition monitors** (after Gap A): Register `ConditionMonitor` for domain-relevant thresholds

### Training
- [ ] **Step-based training**: Train with MAPPO/QMIX, verify convergence

### Evaluation — Failure mode 5a: Chain breakage (available now)
- [ ] **Step-based evaluation**: Record baseline score
- [ ] **Event-driven evaluation (realistic profile)**: All `ScheduleConfig` params at deployment values
- [ ] **Single-param sweeps E1–E4**: Action delay, message latency, agent/physics rate, jitter
- [ ] **Cross-type interaction E5–E6**: Delay × physics rate, latency × jitter

### Evaluation — Failure mode 5b: Disturbance collision (after Gap B)
- [ ] **Disturbance injection E7–E9**: Fault × delay, fault timing in chain, factorial design
- [ ] **Interaction term**: Report D − (B + C − A) for disturbance × event-driven

### Evaluation — Failure mode 5c: Reactive vs periodic (after Gap A)
- [ ] **Response time comparison E10**: Reactive trigger vs wait-for-next-tick
- [ ] **Preemption test E11**: `preempt_next_tick` True vs False
- [ ] **Tier sensitivity E12**: Compare across agent tiers (1s, 60s, 300s)

### Reporting
- [ ] **Gap magnitude**: Step-based vs. event-driven scores
- [ ] **Per-parameter decomposition**: Which parameter contributes what share of the gap
- [ ] **Cross-type interaction effects**: Additive or compounding? (report the interaction term)
- [ ] **Failure mode attribution**: Map degradation to 5a/5b/5c based on event class interactions
- [ ] **Limitations acknowledged**: Note which failure modes (5b, 5c) are not yet tested if gaps are open
