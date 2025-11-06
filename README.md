# 🚀 AI Lunar Lander

> **Autonomous lunar landing system using reinforcement learning and high-fidelity spacecraft dynamics**

Train AI agents to perform precise Moon landings using Basilisk astrodynamics simulation and Stable Baselines3 reinforcement learning.

---

## ⚡ Quick Start

```bash
# 1. Test your setup (2 minutes)
python unified_training.py --mode test

# 2. Train with curriculum learning (4-8 hours) - Recommended
python unified_training.py --mode curriculum

# 3. Monitor training
tensorboard --logdir=./logs

# 4. Evaluate trained model
python unified_training.py --mode eval --model-path ./models/best_model/best_model
```

---

## 🎯 What This Does

This project trains AI agents to autonomously land spacecraft on the Moon, handling:
- **Realistic physics**: 6-DOF dynamics via Basilisk astrodynamics framework
- **Complex terrain**: Procedurally generated lunar craters and slopes
- **Multiple sensors**: IMU, LIDAR, altimeter, fuel gauges
- **Challenging conditions**: Variable altitude, velocity, terrain difficulty

**Training approach**: Progressive curriculum learning from simple hovering → precision landings on extreme terrain

---

## 📋 Prerequisites

```bash
# Python 3.8+
pip install stable-baselines3[extra] gymnasium numpy matplotlib

# Basilisk is included in ./basilisk/ directory
```

---

## 🎓 Training Modes

| Mode | Duration | Purpose |
|------|----------|---------|
| `test` | 2 min | Verify environment setup |
| `demo` | 15 min | Quick demonstration of curriculum learning |
| `standard` | 1-2 hrs | Direct RL training without curriculum |
| `curriculum` | 4-8 hrs | **Progressive difficulty training (best results)** |
| `eval` | 1-2 min | Evaluate trained models |

### Curriculum Stages
1. **Hover Training** → Learn altitude/attitude control
2. **Simple Descent** → Controlled descent from moderate altitude
3. **Precision Landing** → Land softly near target position
4. **Challenging Terrain** → Handle complex lunar terrain
5. **Extreme Conditions** → Master worst-case scenarios

---

## 🤖 Supported Algorithms

- **PPO** (Proximal Policy Optimization) - Default, stable, general-purpose
- **SAC** (Soft Actor-Critic) - Sample efficient, good exploration
- **TD3** (Twin Delayed DDPG) - Continuous control, deterministic

```bash
# Try different algorithms
python unified_training.py --mode curriculum --algorithm ppo
python unified_training.py --mode curriculum --algorithm sac
```

---

## 📊 Monitoring Progress

```bash
# Launch TensorBoard
tensorboard --logdir=./logs

# Open browser to http://localhost:6006
```

**Key metrics**:
- `rollout/ep_rew_mean` - Average episode reward (primary metric)
- `curriculum/current_stage` - Current training stage
- `rollout/ep_len_mean` - Episode length

---

## 📁 Project Structure

```
.
├── unified_training.py              # 🌟 Main training script (all modes)
├── lunar_lander_env.py              # Gymnasium environment
├── ScenarioLunarLanderStarter.py    # Basilisk simulation setup
├── generate_terrain.py              # Terrain generation utilities
│
├── UNIFIED_TRAINING_GUIDE.md        # 📖 Complete training documentation
├── SB3_QUICKSTART.md                # Quick reference for Stable Baselines3
├── TERRAIN_SYSTEM_README.md         # Terrain physics details
│
├── basilisk/                        # Astrodynamics simulation framework
├── generated_terrain/               # Generated terrain heightmaps
├── models/                          # Saved trained models
└── logs/                            # TensorBoard logs
```

---

## 🔧 Common Commands

```bash
# Quick test
python unified_training.py --mode test

# Demo training
python unified_training.py --mode demo

# Full curriculum training (recommended)
python unified_training.py --mode curriculum --n-envs 4

# Standard training (no curriculum)
python unified_training.py --mode standard --timesteps 1000000

# Resume training from checkpoint
python unified_training.py --mode standard --resume ./models/checkpoints/ppo_lunar_lander_500000_steps

# Evaluate model
python unified_training.py --mode eval --model-path ./models/best_model/best_model --eval-episodes 20

# Evaluate with visualization
python unified_training.py --mode eval --model-path ./models/best_model/best_model --render
```

---

## 🏆 Expected Performance

After full curriculum training:
- **Mean reward**: 800-1200 on extreme conditions (terminal 1000 + bonuses up to 400)
- **Success rate**: 60%+ successful landings (curriculum requires this for advancement)
- **Landing criteria**: Altitude < 5m, vertical velocity < 3 m/s, horizontal speed < 2 m/s, attitude < 15° from upright
- **Fuel efficiency**: Bonus up to +150 points for high fuel remaining (only awarded on successful landing)

---

## 🎁 Reward System Design

The reward system uses a **comprehensive multi-component architecture** designed to guide the agent from initial random actions to optimal landing performance.

### Reward Architecture

```
Total Reward = Terminal Rewards + Progress Tracking + Safety/Efficiency + Control Quality
               (±1000 scale)    (0-5 scale)        (±2 scale)          (±1 scale)
```

### 1. Terminal Rewards (±1000) - Dominant Episode Outcome

**Success Landing (1000-1600 points):**
- Base success: **+1000** (largest single component)
- Precision bonus: **+0 to +200** (scales with distance from target)
- Softness bonus: **+0 to +100** (scales with touchdown velocity)
- Attitude bonus: **+0 to +100** (scales with upright orientation)
- Fuel efficiency: **+0 to +150** (quadratic curve, ONLY on success)
- Control smoothness: **+0 to +50** (rewards stable final approach)

**Hard Landing (-300 to -450 points):**
- Base penalty scaled by violation severity (velocity, position, attitude errors)

**Crash (-400 to -800 points):**
- Penalty scales with impact energy (velocity squared)

**High Altitude Failure (-200 to -400 points):**
- Penalty scales with altitude (higher failure = worse penalty)

### 2. Progress Tracking Rewards (0-5 per step) - Continuous Guidance

**Descent Profile (0-1):** Encourages proper descent rate proportional to altitude (-2 to -10 m/s)

**Approach Angle (0-0.5):** Rewards vertical descent near ground (low horizontal/vertical velocity ratio)

**Proximity to Target (0-1):** Progressive reward for getting closer to landing site (active < 200m altitude)

**Attitude Stability (0-0.5):** Rewards upright orientation near ground (< 100m altitude)

**Final Approach Quality (0-2):** High reward in last 50m for being slow, upright, and on-target

### 3. Safety & Efficiency Penalties (±2 per step)

**Danger Zone Warnings (altitude < 50m):**
- Speed danger: **-0 to -1** (excessive vertical velocity)
- Tilt danger: **-0 to -0.5** (tilted orientation)
- Lateral danger: **-0 to -0.5** (high horizontal velocity)

**Fuel Management:** **-0 to -1** (progressive warning as fuel depletes below 10%)

**High Altitude Loitering:** **-0.5** (discourages hovering above 500m)

### 4. Control Quality Penalties (±1 per step)

**Control Effort:** **-0.001 × effort** (encourages efficient control)

**Control Jitter:** **-0.1 × change** (penalizes rapid control changes)

**Spin Rate:** **-0 to -0.5** (discourages uncontrolled rotation)

### Reward Design Philosophy

1. **Terminal rewards dominate** (10x larger than shaping) - clearly signals success/failure
2. **Progressive difficulty** - rewards increase as agent approaches landing
3. **Fuel efficiency rewarded ONLY on success** - prevents hoarding during flight
4. **Multi-dimensional success criteria** - velocity, position, attitude, fuel all matter
5. **Penalties scale with severity** - worse violations get worse penalties
6. **Action smoothing** - exponential moving average (80% old, 20% new) for stable control

### Expected Cumulative Rewards

| Outcome | Cumulative Reward | Description |
|---------|-------------------|-------------|
| **Perfect Landing** | 1200-1600 | All bonuses achieved (precision, fuel, smoothness) |
| **Good Landing** | 900-1200 | Most bonuses achieved |
| **Basic Landing** | 600-900 | Minimal bonuses, but successful |
| **Poor Landing** | 400-600 | Barely meets success criteria |
| **Hard Landing** | -300 to -450 | Crashes but in landing zone |
| **Crash** | -400 to -800 | Impact below surface |
| **Timeout/Abort** | -200 to -400 | Failure at high altitude |

### Monitoring Reward Components

The system tracks **individual reward components** for debugging and analysis:

```bash
# Launch TensorBoard to view reward component breakdown
tensorboard --logdir=./logs

# Navigate to "reward_components" section to see:
# - terminal_success, precision_bonus, fuel_efficiency, etc.
# - descent_profile, approach_angle, proximity, etc.
# - danger penalties, control quality metrics
```

Each episode's info dict includes `reward_components` with detailed breakdown of all reward sources.

### Tuning Guidelines

**If agent not landing:**
- Increase `success_threshold` in curriculum stages
- Increase `min_episodes` for better mastery
- Check TensorBoard for `episode/success_rate_100` metric

**If agent too cautious (hovers):**
- Increase loitering penalty
- Reduce fuel efficiency bonus coefficient
- Add progressive time penalty

**If agent crashes frequently:**
- Increase danger zone penalties
- Reduce initial velocity range in curriculum
- Add more stages to curriculum

**If agent uses too much fuel:**
- Increase fuel efficiency bonus coefficient
- Add fuel consumption penalty during flight
- Reward descent profile adherence more

---

## 📚 Documentation

- **[UNIFIED_TRAINING_GUIDE.md](UNIFIED_TRAINING_GUIDE.md)** - Complete training guide with all options
- **[REWARD_SYSTEM_GUIDE.md](REWARD_SYSTEM_GUIDE.md)** - 🎁 Comprehensive reward system documentation with tuning guide
- **[SB3_QUICKSTART.md](SB3_QUICKSTART.md)** - Quick reference for algorithms and parameters
- **[TERRAIN_SYSTEM_README.md](TERRAIN_SYSTEM_README.md)** - Terrain physics and generation
- **[CURRICULUM_TRAINING_GUIDE.md](CURRICULUM_TRAINING_GUIDE.md)** - Curriculum learning theory and stages

---

## 🚀 Example Workflow

```bash
# 1. First-time setup verification
python unified_training.py --mode test

# 2. Quick demo to understand the system
python unified_training.py --mode demo

# 3. Start full curriculum training
python unified_training.py --mode curriculum --n-envs 4 --algorithm ppo

# 4. Monitor progress (in separate terminal)
tensorboard --logdir=./logs

# 5. After training completes, evaluate
python unified_training.py --mode eval \
    --model-path ./models/curriculum_final \
    --eval-episodes 20
```

---

## 🛠️ Customization

### Custom Terrain
```bash
# Generate custom terrain
python generate_terrain.py \
    --output generated_terrain/custom_terrain.npy \
    --size 2000 \
    --craters 25 \
    --seed 42 \
    --visualize
```

### Modify Training Parameters
Edit `unified_training.py` to customize:
- Curriculum stages and difficulty progression
- Model hyperparameters (learning rate, network architecture)
- Environment configuration (sensors, terrain, initial conditions)
- Success thresholds and advancement criteria

---

## 🐛 Troubleshooting

### Training is slow
```bash
# Use more parallel environments
python unified_training.py --mode curriculum --n-envs 8

# Or try a faster algorithm
python unified_training.py --mode curriculum --algorithm sac
```

### Agent not learning
```bash
# Use curriculum learning (automatic difficulty progression)
python unified_training.py --mode curriculum

# Or train longer
python unified_training.py --mode standard --timesteps 2000000
```

### Environment errors
```bash
# Run diagnostic test
python unified_training.py --mode test

# If test fails, check dependencies
pip install --upgrade stable-baselines3[extra] gymnasium
```

---

## 🔬 Technical Details

### Simulation Framework
- **Basilisk**: High-fidelity spacecraft dynamics with 6-DOF rigid body simulation
- **Gravity**: Lunar gravitational field (μ = 4.9028×10¹² m³/s²)
- **Propulsion**: 3 Raptor Vacuum engines (2.5 MN thrust each, 40-100% throttle)
- **Sensors**: IMU (noisy), LIDAR (64-ray cone scan), altimeter, fuel gauges, attitude sensors
- **Terrain**: Analytical Bekker-Wong model with procedural crater generation and realistic regolith mechanics

### Reinforcement Learning
- **Framework**: Stable Baselines3 (PyTorch-based)
- **Observation**: 32-dimensional state vector (position, velocity, Euler angles, fuel flow rate, time-to-impact, LIDAR azimuthal bins, IMU)
- **Observation normalization**: VecNormalize for zero-mean, unit-variance observations (improves stability)
- **Action**: 4-dimensional continuous (main throttle + pitch/yaw/roll torque commands)
- **Action smoothing**: Exponential moving average filter (80% old, 20% new) for stable control
- **Reward**: Comprehensive multi-component system with:
  - Terminal rewards (±1000): Dominant signals for success/failure
  - Progress tracking (0-5): Continuous guidance toward landing
  - Safety penalties (±2): Danger zone warnings and efficiency
  - Control quality (±1): Smooth control and technique optimization
- **Curriculum Learning**: 5 progressive stages with advancement requiring both mean reward threshold AND 60%+ success rate

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🌟 Key Features

✅ **Curriculum learning** for robust policy development  
✅ **Multiple RL algorithms** (PPO, SAC, TD3)  
✅ **High-fidelity physics** via Basilisk  
✅ **Procedural terrain** generation  
✅ **Real-time monitoring** with TensorBoard  
✅ **Checkpoint system** for resuming training  
✅ **Comprehensive evaluation** tools  

---

**Ready to train an AI to land on the Moon?** 🌙

Start here:
```bash
python unified_training.py --mode test
```

Then read **[UNIFIED_TRAINING_GUIDE.md](UNIFIED_TRAINING_GUIDE.md)** for complete documentation.
