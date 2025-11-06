# 🎁 Comprehensive Reward System Guide

> **Complete documentation of the optimized multi-component reward system for lunar landing RL training**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Design Philosophy](#design-philosophy)
5. [Tuning Guidelines](#tuning-guidelines)
6. [Debugging & Monitoring](#debugging--monitoring)
7. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

The reward system is designed to guide the RL agent from random initial behavior to optimal lunar landing performance through a **multi-component architecture** with carefully balanced scales.

### Key Characteristics

✅ **Terminal rewards dominate** (±1000) to clearly signal episode outcomes  
✅ **Progressive guidance** through altitude-gated shaping rewards  
✅ **Multi-dimensional success criteria** (velocity, position, attitude, fuel)  
✅ **Penalties scale with severity** (worse violations = worse penalties)  
✅ **Fuel efficiency rewarded ONLY on success** (prevents hoarding)  
✅ **Action smoothing** built-in for stable control  

### Total Reward Range

| Outcome | Reward Range | Frequency (trained agent) |
|---------|--------------|---------------------------|
| Perfect landing | 1200-1600 | 10-20% |
| Good landing | 900-1200 | 30-40% |
| Basic landing | 600-900 | 20-30% |
| Poor landing | 400-600 | 5-10% |
| Hard landing | -300 to -450 | 2-5% |
| Crash | -400 to -800 | 1-3% |
| Timeout | -200 to -400 | 1-2% |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TOTAL REWARD                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Terminal Rewards        Progress Tracking                 │
│  (±1000 scale)          (0-5 per step)                     │
│  ├─ Success: +1000       ├─ Descent profile: 0-1           │
│  ├─ Precision: +0-200    ├─ Approach angle: 0-0.5          │
│  ├─ Softness: +0-100     ├─ Proximity: 0-1                 │
│  ├─ Attitude: +0-100     ├─ Attitude stability: 0-0.5      │
│  ├─ Fuel eff: +0-150     └─ Final approach: 0-2            │
│  ├─ Smoothness: +0-50                                       │
│  ├─ Hard landing: -300-450                                  │
│  ├─ Crash: -400-800                                         │
│  └─ Failure: -200-400    Safety & Efficiency                │
│                          (±2 per step)                      │
│                          ├─ Speed danger: -0 to -1          │
│                          ├─ Tilt danger: -0 to -0.5         │
│                          ├─ Lateral danger: -0 to -0.5      │
│                          ├─ Fuel warning: -0 to -1          │
│                          └─ Loitering: -0.5                 │
│                                                             │
│                          Control Quality                     │
│                          (±1 per step)                      │
│                          ├─ Control effort: -0.001×effort   │
│                          ├─ Control jitter: -0.1×change     │
│                          └─ Spin rate: -0 to -0.5           │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Terminal Rewards (±1000 scale)

**Purpose:** Provide dominant signal for episode outcome, ensuring agent clearly understands success vs failure.

#### Success Landing Components

```python
# Base success (dominant component)
reward += 1000.0

# Precision bonus (0-200): Reward accurate landing
precision_score = 1.0 - (horizontal_distance / 20.0)
precision_bonus = 200.0 * max(0, precision_score)
reward += precision_bonus

# Softness bonus (0-100): Reward gentle touchdown
softness_score = 1.0 - (abs(vertical_vel) / 3.0)
softness_bonus = 100.0 * max(0, softness_score)
reward += softness_bonus

# Attitude bonus (0-100): Reward upright landing
attitude_score = 1.0 - (attitude_error_deg / 15.0)
attitude_bonus = 100.0 * max(0, attitude_score)
reward += attitude_bonus

# Fuel efficiency bonus (0-150): ONLY on success
# Quadratic curve: rewards high fuel remaining exponentially
fuel_efficiency = 150.0 * (fuel_fraction ** 1.5)
reward += fuel_efficiency

# Control smoothness bonus (0-50): Reward stable approach
control_smoothness = 50.0 * max(0, 1.0 - angular_rate / 0.1)
reward += control_smoothness
```

**Success Criteria:**
- Altitude: 0-5m (in landing zone)
- Vertical velocity: < 3 m/s
- Horizontal speed: < 2 m/s
- Position: < 20m from target
- Attitude: < 15° from upright

**Total possible success reward:** 1000 + 200 + 100 + 100 + 150 + 50 = **1600 points**

#### Failure Components

```python
# Hard landing (in landing zone but violates criteria)
total_violation = (vel_violation + horiz_violation + 
                   pos_violation + att_violation)
hard_landing_penalty = -(300.0 + 150.0 * total_violation)  # -300 to -450

# Crash (below surface)
impact_energy = sqrt(vertical_vel^2 + horizontal_speed^2)
crash_penalty = -(400.0 + 100.0 * min(impact_energy / 5.0, 4.0))  # -400 to -800

# High altitude failure (timeout, etc.)
altitude_factor = min(altitude / 1000.0, 2.0)
failure_penalty = -(200.0 + 100.0 * altitude_factor)  # -200 to -400
```

---

### 2. Progress Tracking Rewards (0-5 per step)

**Purpose:** Provide continuous guidance toward successful landing throughout the descent.

#### A. Descent Profile (0-1 per step)

Encourages proper descent rate proportional to altitude:

```python
if altitude > 10.0:
    target_descent_rate = -2.0 - (altitude / 200.0) * 8.0  # -2 to -10 m/s
    target_descent_rate = max(target_descent_rate, -10.0)
    descent_error = abs(vertical_vel - target_descent_rate)
    descent_reward = 1.0 * max(0, 1.0 - descent_error / 5.0)
```

**Optimal descent profile:**
- 1000m altitude → -6 m/s vertical velocity
- 500m altitude → -4 m/s
- 100m altitude → -2.5 m/s
- 10m altitude → -2 m/s

#### B. Approach Angle (0-0.5 per step)

Encourages vertical descent near ground (active < 100m altitude):

```python
velocity_ratio = horizontal_speed / max(abs(vertical_vel), 0.1)
approach_reward = 0.5 * max(0, 1.0 - velocity_ratio)
```

**Optimal:** Low horizontal velocity relative to vertical velocity (nearly vertical descent).

#### C. Proximity to Target (0-1 per step)

Progressive reward for approaching landing site (active < 200m altitude):

```python
proximity_score = 1.0 - min(horizontal_distance / 50.0, 1.0)
proximity_reward = 1.0 * proximity_score
```

#### D. Attitude Stability (0-0.5 per step)

Rewards upright orientation near ground (active < 100m altitude):

```python
attitude_stability = max(0, 1.0 - attitude_error_deg / 30.0)
stability_reward = 0.5 * attitude_stability
```

#### E. Final Approach Quality (0-2 per step)

High reward in last 50m for combined quality metrics:

```python
final_approach_score = (
    max(0, 1.0 - abs(vertical_vel) / 5.0) * 0.4 +      # Slow descent
    max(0, 1.0 - horizontal_speed / 3.0) * 0.3 +       # Low horizontal
    max(0, 1.0 - horizontal_distance / 30.0) * 0.2 +   # Near target
    max(0, 1.0 - attitude_error_deg / 20.0) * 0.1      # Upright
)
final_approach_reward = 2.0 * final_approach_score
```

---

### 3. Safety & Efficiency (±2 per step)

**Purpose:** Provide warnings for dangerous situations and encourage fuel efficiency.

#### Danger Zone Warnings (altitude < 50m)

```python
# Speed danger
if abs(vertical_vel) > 10.0:
    danger_penalty = -1.0 * (abs(vertical_vel) - 10.0) / 10.0

# Tilt danger
if attitude_error_deg > 30.0:
    tilt_penalty = -0.5 * (attitude_error_deg - 30.0) / 30.0

# Lateral danger
if horizontal_speed > 5.0:
    lateral_penalty = -0.5 * (horizontal_speed - 5.0) / 5.0
```

#### Fuel Management

```python
if fuel_fraction < 0.1:
    fuel_penalty = -1.0 * (0.1 - fuel_fraction) / 0.1
```

#### Loitering Penalty

```python
if altitude > 500.0 and abs(vertical_vel) < 2.0:
    loiter_penalty = -0.5
```

---

### 4. Control Quality (±1 per step)

**Purpose:** Encourage smooth, efficient control.

```python
# Control effort penalty
control_penalty = -0.001 * (throttle_effort + torque_effort)

# Control jitter penalty
action_change = norm(action - prev_action)
jitter_penalty = -0.1 * min(action_change, 2.0)

# Spin rate penalty
if angular_rate > 0.2:
    spin_penalty = -0.5 * (angular_rate - 0.2) / 0.2
```

---

## Design Philosophy

### 1. Terminal Rewards Dominate

Terminal rewards (±1000) are **10x larger** than per-step shaping rewards (0-5). This ensures:
- Clear signal for episode outcome
- Agent prioritizes landing success over micro-optimization
- Failure states are clearly distinguished from success

### 2. Progressive Difficulty

Shaping rewards are **altitude-gated**:
- High altitude (>500m): Focus on descent profile
- Medium altitude (100-500m): Add approach angle and proximity
- Low altitude (<100m): Add attitude stability
- Final approach (<50m): High rewards + danger warnings

This creates a natural learning progression.

### 3. Fuel Efficiency Paradox Solution

**Problem:** If fuel efficiency is rewarded during flight, agent learns to hoard fuel and crash.  
**Solution:** Fuel efficiency bonus (+150) is **ONLY awarded on successful landing**.

This teaches:
- During flight: Use fuel as needed to achieve good approach
- During landing: Land with remaining fuel for bonus

### 4. Multi-Dimensional Success

Success requires satisfying ALL criteria:
- Velocity (vertical AND horizontal)
- Position (distance from target)
- Attitude (upright orientation)
- Fuel (efficiency bonus)
- Control (smoothness bonus)

This prevents degenerate solutions (e.g., landing far from target but softly).

### 5. Severity-Scaled Penalties

Penalties increase with violation severity:
- Minor speed excess → small penalty
- Major crash → large penalty
- Intermediate violations → scaled penalty

This provides informative gradient for learning.

---

## Tuning Guidelines

### Symptom: Agent Not Landing

**Diagnosis:** Check TensorBoard `episode/success_rate_100` metric. If < 10%, agent hasn't learned landing basics.

**Solutions:**
1. **Lower initial altitude** in Stage 1 (try 30-80m instead of 50-100m)
2. **Increase terminal success reward** (try 1500 instead of 1000)
3. **Reduce success criteria** temporarily (e.g., altitude < 7m instead of 5m)
4. **Increase training time** for Stage 1 (try 150k steps instead of 100k)
5. **Check observation normalization** (ensure VecNormalize is applied)

### Symptom: Agent Hovers Without Descending

**Diagnosis:** Agent maximizes loitering instead of landing.

**Solutions:**
1. **Increase loitering penalty** (try -1.0 instead of -0.5)
2. **Add time penalty** (e.g., -0.01 per step above 500m)
3. **Reduce fuel efficiency bonus** (try 100 instead of 150)
4. **Increase descent profile reward** (try 2.0 instead of 1.0)

### Symptom: Agent Crashes Frequently

**Diagnosis:** Agent too aggressive, doesn't learn safe descent.

**Solutions:**
1. **Increase danger zone penalties** (multiply by 2x)
2. **Increase crash penalty** (try -1000 instead of -400 to -800)
3. **Add intermediate stages** to curriculum (more gradual difficulty)
4. **Reduce initial velocity range** in early stages
5. **Increase softness bonus weight** (try 150 instead of 100)

### Symptom: Agent Uses Too Much Fuel

**Diagnosis:** Agent doesn't learn fuel efficiency.

**Solutions:**
1. **Increase fuel efficiency bonus coefficient** (try 200 instead of 150)
2. **Change curve from linear to quadratic** (use `fuel_fraction^2` instead of `fuel_fraction^1.5`)
3. **Add fuel consumption penalty during flight** (e.g., -0.01 per kg/s flow rate)
4. **Increase weight on descent profile reward** (encourages efficient trajectory)

### Symptom: Agent Lands Far From Target

**Diagnosis:** Agent ignores proximity reward.

**Solutions:**
1. **Increase precision bonus weight** (try 300 instead of 200)
2. **Tighten success criteria** (require < 10m instead of < 20m)
3. **Increase proximity reward coefficient** (try 2.0 instead of 1.0)
4. **Add horizontal distance penalty in danger zone** (< 50m altitude)

### Symptom: Agent Has Rough Control

**Diagnosis:** Agent doesn't learn smooth control.

**Solutions:**
1. **Increase action smoothing alpha** (try 0.3 instead of 0.2)
2. **Increase control jitter penalty** (try -0.2 instead of -0.1)
3. **Add rate limit on actions** (clip action change to ±0.2)
4. **Increase control smoothness bonus** (try 100 instead of 50)

---

## Debugging & Monitoring

### TensorBoard Metrics

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir=./logs
```

**Key metrics to watch:**

| Metric | What to Look For |
|--------|------------------|
| `rollout/ep_rew_mean` | Should trend upward, reach 800-1200 |
| `episode/success_rate_100` | Should reach 60%+ for curriculum advancement |
| `reward_components/terminal_success` | Should increase over time (more successes) |
| `reward_components/precision_bonus` | Should be positive and increasing |
| `reward_components/fuel_efficiency` | Should be positive (indicates fuel remaining) |
| `reward_components/descent_profile` | Should be positive (proper descent) |
| `reward_components/crash` | Should decrease over time (fewer crashes) |

### Reward Component Breakdown

Each episode's `info` dict includes `reward_components` with detailed breakdown:

```python
info = {
    'altitude': 2.3,
    'velocity': [0.1, -0.2, -1.5],
    'fuel_fraction': 0.45,
    'attitude_error_deg': 3.2,
    'step': 543,
    'reward_components': {
        'terminal_success': 1000.0,
        'precision_bonus': 185.3,
        'softness_bonus': 92.1,
        'attitude_bonus': 98.4,
        'fuel_efficiency': 118.7,
        'control_smoothness': 41.2,
        'descent_profile': 0.8,
        'approach_angle': 0.4,
        # ... etc
    }
}
```

### Custom Logging

Add custom reward component logging:

```python
# In unified_training.py, modify RewardStatisticsCallback
def _on_step(self):
    # Log specific components you want to track
    if 'reward_components' in info:
        components = info['reward_components']
        
        # Log terminal rewards separately
        if 'terminal_success' in components:
            self.logger.record("terminal/success", components['terminal_success'])
        
        # Log shaping rewards
        shaping_total = sum(v for k, v in components.items() 
                          if k in ['descent_profile', 'approach_angle', ...])
        self.logger.record("shaping/total", shaping_total)
```

---

## Common Issues & Solutions

### Issue: Reward Not Increasing

**Possible causes:**
1. Observation space not normalized → Apply `VecNormalize`
2. Learning rate too high/low → Try 1e-4 to 5e-4
3. Curriculum too hard → Lower initial altitude in Stage 1
4. Not enough exploration → Increase `ent_coef` (PPO) or exploration noise (SAC/TD3)

**Debug steps:**
1. Run `python unified_training.py --mode test` to verify environment
2. Check `rollout/ep_len_mean` - should be < max_episode_steps
3. Check `rollout/entropy_loss` - should not be near 0 (indicates exploration)
4. Try training on Stage 1 only for 200k steps

### Issue: High Variance in Rewards

**Possible causes:**
1. Terrain randomization too high
2. Initial conditions too variable
3. Stochastic policy not converged

**Solutions:**
1. Reduce terrain complexity in early stages
2. Narrow initial_altitude_range and initial_velocity_range
3. Increase number of parallel environments (`--n-envs 8`)
4. Increase training time per stage

### Issue: Agent Learns Then Forgets

**Possible causes:**
1. Curriculum advancement too fast
2. Model hyperparameters unstable
3. Observation normalization issues

**Solutions:**
1. Increase `min_episodes` before advancement (try 400 instead of 200)
2. Add stage regression (if mean reward drops, go back one stage)
3. Save `VecNormalize` statistics and reload them
4. Reduce learning rate by 2x

### Issue: Different Algorithms Perform Differently

**Expected behavior:**

| Algorithm | Sample Efficiency | Stability | Final Performance |
|-----------|------------------|-----------|-------------------|
| PPO | Medium | High | Good |
| SAC | High | Medium | Best |
| TD3 | High | Medium | Good |

**Recommendations:**
- **PPO**: Best for initial development (stable, predictable)
- **SAC**: Best for sample efficiency (learns faster, uses fewer timesteps)
- **TD3**: Best for deterministic control (less exploration noise)

---

## Code Locations

**Reward function:** `lunar_lander_env.py::_compute_reward()`  
**Success criteria:** `lunar_lander_env.py::_check_termination()`  
**Curriculum stages:** `unified_training.py::_create_curriculum()`  
**Reward statistics callback:** `unified_training.py::RewardStatisticsCallback`  

---

## Summary

The reward system is designed as a **comprehensive multi-component architecture** that:

✅ Provides clear terminal signals (±1000) for success/failure  
✅ Guides learning through progressive shaping (0-5 per step)  
✅ Warns about dangerous situations (±2 per step)  
✅ Encourages smooth control (±1 per step)  
✅ Rewards fuel efficiency ONLY on success  
✅ Scales penalties with violation severity  

**Total possible reward range:** -800 (worst crash) to +1600 (perfect landing)

**Typical successful landing:** 900-1200 points

**Curriculum advancement threshold:** Stage 1-5 require 400-900 mean reward + 60% success rate

For questions or issues, check TensorBoard metrics and reward component breakdown in episode info.

---

**Ready to tune your reward function?** Start by monitoring `episode/success_rate_100` and `reward_components/*` in TensorBoard!
