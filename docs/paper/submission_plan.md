# HERON: NeurIPS 2026 D&B Submission Plan

**Track:** NeurIPS 2026 Datasets & Benchmarks (9 pages + unlimited appendix, single-blind)
**Category:** Framework/Infrastructure Paper
**Deadline:** TBD (typically late May)

---

## 1. One-Sentence Pitch

> HERON shifts MARL simulation from environment-centric synchronous stepping to agent-paced event-driven execution, making execution model, information structure, and coordination protocols first-class experimental variables for CPS research.

---

## 2. Current Status (Honest Inventory)

### What's Built

| Component | Status | Details |
|-----------|--------|---------|
| Agent hierarchy (3 levels) | ✅ Done | FieldAgent, CoordinatorAgent, SystemAgent |
| Proxy (state mediation) | ✅ Done | Centralized state management, visibility filtering |
| EventScheduler | ✅ Done | Heap-based priority queue, 7 event types, configurable delays/jitter |
| Dual execution modes | ✅ Done | `step()` (sync) + `run_event_driven()` (event-driven) |
| MessageBroker | ✅ Done | InMemoryBroker, channel-based, 6 message types |
| Feature system | ✅ Done | FeatureMeta metaclass, 4 visibility scopes |
| Protocol base classes | ✅ Done | VerticalProtocol (VectorDecomposition), HorizontalProtocol (StateShare) |
| Power domain: 20 Features | ✅ Done | But only 6/20 have visibility labels implemented |
| Power domain: 6 agent types | ✅ Done | DeviceAgent, Generator, ESS, Transformer, PowerGridAgent, GridSystemAgent |
| Power domain: 6 test networks | ✅ Done | IEEE 13/34/123, CIGRE LV, Case34 3ph, Case LVMG |
| Power domain: 7 tutorials | ✅ Done | Jupyter notebooks |

### What's NOT Built

| Component | Status | Risk |
|-----------|--------|------|
| Visibility labels on 14/20 power Features | ❌ TODO | Low (straightforward) |
| Domain-specific protocols (power) | ❌ TODO | Medium (setpoint, price signal, P2P, consensus) |
| Traffic domain (entire) | ❌ TODO | **HIGH** (0% complete, 0 lines of code) |
| MAPPO/IPPO via RLlib | ❌ TODO | Medium (no RL integration yet) |
| QMIX / TarMAC | ❌ TODO | Medium |
| All 12 experiments | ❌ TODO | High (0/12 can run today) |
| Scalability benchmarks | ❌ TODO | Low |
| Framework comparison (PettingZoo/EPyMARL) | ❌ TODO | Low-Medium |
| Documentation / video | ❌ TODO | Low |
| Croissant metadata | ❌ TODO | Low |

### Honest Summary

The **framework core is solid** (agents, proxy, scheduler, broker, protocols, features). The **power domain has good coverage** (20 Features, 6 networks, 6 agent types). The major gaps are: (1) traffic domain is 0% complete, (2) no RL algorithm integration, (3) no experiments can run.

---

## 3. Paper Structure (9 Pages + Appendix)

### Main Body

| Section | Pages | Content |
|---------|-------|---------|
| 1. Introduction | 1.3 | Simulation mismatch problem, 3 dimensions (execution, info, protocols), key idea (Proxy-mediated hierarchy), 6 contributions |
| 2. Related Work | 0.5 | Multi-agent benchmarks (PettingZoo, EPyMARL, MARLlib), hierarchical RL, CPS-domain benchmarks. Comparison table (7 frameworks) |
| 3. Framework | 3.0 | 3.1 Overview, 3.2 Agent hierarchy + dual modes + Algorithm 1, 3.3 Proxy, 3.4 Features, 3.5 MessageBroker, 3.6 Protocols |
| 4. Power Case Study | 0.8 | 20 Features (categorized table), 3-level agent hierarchy, 6 test networks, benchmark questions |
| 5. Traffic Case Study | 0.5 | Agent mapping, cross-domain reuse evidence, 14 Features |
| 6. Experiments | 1.8 | 5 categories: observability ablation, sim-to-real gap, protocol comparison, timing sensitivity, algorithm comparison |
| 7. Conclusion | 0.1 | Brief summary |

### Appendix (~12 pages)

| Section | Content |
|---------|---------|
| A. API Reference | Full method signatures (BaseEnv, BaseAgent, Proxy, Protocol, EventScheduler) |
| B. Event-Driven Flow | Detailed sequence diagram of a full tick cycle |
| C. Feature Specs | All 20 power + 14 traffic providers with dimensions and visibility |
| D. Full Experiments | Complete results across all networks and domains |
| E. Framework Comparison | LOC comparison, PettingZoo/EPyMARL implementation attempts |
| F. Network Specs | All test network parameters |
| G. Hyperparameters | Training configs |
| H. Safety Metrics | Voltage/loading/SOC violation definitions |
| I. Reproducibility | Croissant metadata, dataset cards |
| J. Game Domain (stretch) | Cooperative RTS with imposed hierarchy, sync vs. event-driven comparison |

---

## 4. Five Contributions

| # | Contribution | What Makes It Novel | Key Evidence |
|---|-------------|--------------------|--------------|
| 1 | **Event-driven hierarchical execution** | Dual modes (sync training + event-driven validation) via heap-based EventScheduler. Cannot be achieved by wrapping PettingZoo. | Algorithm 1, timing sensitivity experiments |
| 2 | **Proxy for state mediation** | All state access goes through a single gatekeeper that enforces visibility. Prevents the common "global state leak" in MARL benchmarks. | Proxy API, sim-to-real gap experiments |
| 3 | **Features with visibility labels** | 4-level visibility (public/owner/upper_level/system) as first-class experimental variable, not a binary on/off. | Observability ablation experiments |
| 4 | **MessageBroker + channel isolation** | Explicit message-based communication with typed channels, environment isolation for parallel training. | Channel naming, message types |
| 5 | **Composable protocol system** | CommunicationProtocol + ActionProtocol composition. Swap coordination mechanisms without changing agents. | Protocol comparison experiments |

### Key Differentiator vs. PettingZoo

PettingZoo standardizes the **env-algorithm interface** (reset, step, observe, act). HERON standardizes **what happens inside the environment** (how agents access state, how they communicate, how coordination works). Event-driven execution cannot be achieved by wrapping---it requires changing the fundamental step loop.

### Explicit Non-Claims

- HERON is **not** "privacy-preserving" in the DP/cryptographic sense
- We don't claim to "discover" findings---we provide infrastructure for systematic study
- We complement PettingZoo (HERON envs can export to PettingZoo API), not replace it

---

## 5. Experiments Plan (12 Total)

### Paired Cross-Domain Experiments (1-8)

These must show **consistent relative ordering** across both domains to validate domain-agnostic design.

| # | Experiment | Power | Traffic | Key Question |
|---|-----------|-------|---------|-------------|
| 1-2 | Visibility ablation | IEEE 34-bus, 3 microgrids | 5x5 grid, 25 intersections | Does system > upper > owner hold in both domains? |
| 3-4 | Protocol comparison | Vertical vs. horizontal | Vertical vs. horizontal | Does protocol ranking hold across domains? |
| 5-6 | Timing sensitivity | SCADA-calibrated (IEEE 2030) | NTCIP-calibrated | How much does sync-to-event gap vary by domain? |
| 7-8 | Algorithm comparison | MAPPO/IPPO/QMIX/TarMAC | MAPPO/IPPO/QMIX/TarMAC | Is visibility ordering algorithm-agnostic? |

### Single Experiments (9-12)

| # | Experiment | Details |
|---|-----------|---------|
| 9 | Scalability | 10 to 2000 agents, log-log plots, component breakdown |
| 10 | Broker overhead | Abstraction cost < 5% vs. direct calls |
| 11 | ~~User study~~ | Not needed for D&B (no precedent: PettingZoo, RLlib, SMAC omit this) |
| 12 | Framework comparison | Implement microgrid in PettingZoo + EPyMARL, measure LOC and feature gaps |

### Algorithm Integration

| Algorithm | Category | Integration Path | Status |
|-----------|----------|-----------------|--------|
| MAPPO | Policy gradient | RLlib MultiAgentEnv | ❌ TODO |
| IPPO | Independent | RLlib (independent policies) | ❌ TODO |
| QMIX | Value decomposition | RLlib or custom | ❌ TODO |
| TarMAC | Communication | Custom (learned comm) | ❌ TODO |

### Timing Configurations (CPS-Calibrated)

| Config | Distribution | Source | Parameters |
|--------|-------------|--------|------------|
| Synchronous | -- | Training baseline | All agents tick simultaneously |
| Uniform | U(0, tau_max) | Sensitivity sweep | tau_max in {0.5s, 1s, 2s, 4s} |
| SCADA | LogNormal(mu, sigma) | IEEE Std 2030-2011 | mu=2s, sigma=0.8s |
| Jitter | N(0, sigma) + base | Communication noise | sigma in {0.1s, 0.5s, 1s} |
| Heterogeneous | Per-agent different tau | Realistic deployment | Fast/slow agent mix |

---

## 6. Implementation Roadmap

### Phase 1: Complete Power Domain (Week 1-2)

- [ ] Add visibility labels to remaining 14/20 Features
- [ ] Implement 4 domain-specific protocols:
  - [ ] SetpointProtocol (vertical, centralized dispatch)
  - [ ] PriceSignalProtocol (vertical, decentralized response)
  - [ ] P2PTradingProtocol (horizontal, market clearing)
  - [ ] ConsensusProtocol (horizontal, gossip-based averaging)
- [ ] Integrate MAPPO + IPPO via RLlib MultiAgentEnv
- [ ] Run first experiments: visibility ablation + protocol comparison on power

### Phase 2: Traffic Domain (Week 2-4) --- CRITICAL PATH

This is the **highest risk item**. Entire domain is 0% complete.

- [ ] Choose physics backend (SUMO via TraCI, or simplified model)
- [ ] Implement 14 Features (5 owner, 3 upper_level, 4 system, 2 public)
- [ ] Implement 4 traffic protocols (FixedTiming, AdaptiveOffset, GreenWave, ConsensusTiming)
- [ ] Create 5x5 grid network (25 intersections)
- [ ] Implement 3 agent types: SignalAgent, CorridorAgent, NetworkAgent
- [ ] Verify identical base classes work (no traffic-specific hacks)
- [ ] **Parity checkpoint**: side-by-side table showing same structure as power

### Phase 3: Experiments (Week 4-6)

- [ ] Run all 8 paired experiments (power + traffic)
- [ ] Integrate QMIX + TarMAC
- [ ] Run all 4 algorithms on both domains
- [ ] Scalability benchmarks (10 to 2000 agents)
- [ ] Framework comparison (PettingZoo/EPyMARL implementation attempts)
- [ ] Event-driven timing sensitivity (5 configs x 2 domains)

### Phase 4: Paper + Polish (Week 6-8)

- [ ] Write paper with real results (replace all placeholder values)
- [ ] Create appendix content (API ref, sequence diagrams, full tables)
- [ ] Documentation: API reference, 2 quick-start tutorials, 5-min video
- [ ] Clean GitHub repo
- [ ] Croissant metadata for D&B compliance

**Total estimated timeline: 8 weeks**

---

## 7. Go/No-Go Checklist

### Must Have (ALL required)

- [ ] Traffic domain parity: 14 Features, 4 protocols, 25 agents
- [ ] 4 algorithms tested (MAPPO, IPPO, QMIX, TarMAC)
- [ ] Consistent visibility ordering (system > upper > owner) across both domains
- [ ] Event-driven timing calibrated to IEEE 2030 / NTCIP 1202
- [ ] Scalability benchmarked to 1000+ agents
- [ ] Framework comparison includes EPyMARL or MARLlib (not just PettingZoo)
- [ ] No "privacy-preserving" claims anywhere
- [ ] Documentation + tutorials + working examples in repo

### Do NOT Submit If

- Traffic domain has fewer Features or protocols than power
- Only 2 algorithms tested
- Event-driven not calibrated to real CPS timing standards
- Scalability only tested up to 500 agents
- No documentation or working examples

---

## 8. Acceptance Probability Estimate

| Milestone | P(Accept) | Remaining Risk |
|-----------|-----------|----------------|
| Current state (framework only, no experiments) | 20-30% | Everything |
| + Power experiments complete | 40-50% | No cross-domain evidence |
| + Traffic parity achieved | 60-70% | Scalability, algorithm diversity |
| + All 4 algorithms, scalability to 1000+ | 75-80% | Novelty perception |
| **+ Framework comparison, documentation, video** | **80-85%** | Residual: reviewer taste |

---

## 9. Ecosystem Comparison Table (for paper)

| Framework | Abstraction Level | Event-Driven | Visibility | Protocols | State Mediation | CPS Focus |
|-----------|-------------------|-------------|-----------|-----------|-----------------|-----------|
| **HERON** | Internal env structure | Native | 4-level | Composable | Proxy | Yes |
| PettingZoo | Env-algorithm interface | No | Manual | No | No | No |
| EPyMARL | Algorithm benchmarking | No | No | No | No | No |
| MARLlib | Algorithm library | No | No | No | No | No |
| SMAC/SMACv2 | Domain-specific | No | Partial | No | No | No |
| PowerGridworld | Power-specific | No | Binary | No | No | Yes |
| CityLearn | Building-specific | No | Binary | No | No | Yes |
| Grid2Op | Power-specific | No | N/A | No | No | Yes |

---

## 10. Language Guidelines

### Use

- "partial observability", "information asymmetry"
- "agent-paced event-driven", "CPS timing constraints"
- "shifts from environment-centric to agent-centric"
- "complements PettingZoo" (different abstraction level)

### Avoid

- ~~"privacy-preserving"~~, ~~"privacy guarantees"~~
- ~~"we discover"~~, ~~"our finding"~~ (we provide infrastructure, not findings)
- ~~"better than PettingZoo"~~ (different level, complementary)

---

## 11. Required References

**MARL Frameworks:** PettingZoo (Terry et al., JMLR 2021), EPyMARL (Papoudakis et al., NeurIPS 2021), MARLlib (Hu et al., JMLR 2023), MAPPO (Yu et al., NeurIPS 2022)

**Communication:** DIAL/CommNet (Foerster et al., NeurIPS 2016; Sukhbaatar et al., NeurIPS 2016), TarMAC (Das et al., ICML 2019)

**CPS Standards:** IEEE Std 2030-2011, NASPI 2018, NTCIP 1202

**Power RL:** Grid2Op (Donnot et al., 2020), gym-anm (Henry et al., 2021), CityLearn (Vazquez-Canteli et al., 2019), PowerGridworld (Biagioni et al., 2022)

**Traffic:** Wei et al. (2019), Zheng et al. (CIKM 2019)

---

## 12. Stretch Goals (If Time Permits)

### Game Domain Appendix (Extensibility Demonstration)

**Purpose:** Show HERON generalizes beyond CPS. A ~2-3 page appendix section demonstrating that event-driven hierarchical execution produces meaningful behavioral differences in a game setting.

**Candidate:** Simplified cooperative RTS (or modified SMAC/MPE scenario).

**Agent hierarchy (imposed):**

| Level | Agent | Tick Rate | Observability | Role |
|-------|-------|-----------|---------------|------|
| 3 (System) | Strategist | Every ~30s | Full map (delayed) | Resource allocation, tech decisions |
| 2 (Coordinator) | Squad Leader | Every ~5s | Squad area + neighbor summary | Group positioning, engagement calls |
| 1 (Field) | Unit | Every ~0.5s | Local radius only | Movement, ability usage, combat micro |

**Information flow via Proxy:**
- Units see only local radius (owner visibility)
- Squad leaders see their squad's states + neighbor summaries (upper_level visibility)
- Strategist sees full map but with observation delay (system visibility)
- All communication goes through MessageBroker (squad leaders relay strategist orders to units)

**Protocols:**
- Vertical: OrderProtocol (strategist → squad leader: attack/defend/retreat directives), TacticProtocol (squad leader → units: formation/target assignments)
- Horizontal: ScoutShareProtocol (squad leaders share enemy sightings with neighbors)

**Key experiment (1 table):**

| Execution Mode | Win Rate | Avg. Reward | Coordination Score |
|---------------|----------|-------------|-------------------|
| Synchronous (all tick at 0.5s, full obs) | Baseline | -- | -- |
| Event-driven (heterogeneous rates, hierarchy-filtered obs) | ? | ? | ? |
| Event-driven + command latency (2s strategist delay) | ? | ? | ? |

**Hypothesis:** Event-driven execution with command latency forces emergent local autonomy --- units learn to act independently when orders are delayed, while synchronous mode produces brittle policies that depend on instant global coordination.

**Implementation scope:**
- [ ] Wrap an existing SMAC/MPE environment with HERON agents (reuse FieldAgent, CoordinatorAgent, SystemAgent)
- [ ] Implement 3-4 game Features (LocalVision, SquadState, MapOverview, ResourceStatus)
- [ ] Implement 2 protocols (OrderProtocol, ScoutShareProtocol)
- [ ] Run sync vs. event-driven comparison (1 experiment)
- [ ] Write 2-3 page appendix section

**Priority:** Do this AFTER traffic domain and all main experiments are complete. Estimated effort: ~1 week.

---

### Other Stretch Goals

- Hardware-in-the-loop validation (OPAL-RT or similar)
- Real SCADA timing traces (anonymized)
- SUMO-HERON integration for traffic
- 3-5 informal practitioner interviews
