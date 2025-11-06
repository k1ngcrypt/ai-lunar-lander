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
- **Mean reward**: 400-700 on extreme conditions (includes 500 base + bonuses up to 200)
- **Success rate**: 60%+ successful landings (curriculum requires this for advancement)
- **Landing criteria**: Altitude < 5m, vertical velocity < 3 m/s, horizontal speed < 2 m/s, attitude < 15° from upright
- **Fuel efficiency**: Bonus up to +100 points for high fuel remaining (only awarded on successful landing)

---

## 📚 Documentation

- **[UNIFIED_TRAINING_GUIDE.md](UNIFIED_TRAINING_GUIDE.md)** - Complete training guide with all options
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
- **Reward**: Rebalanced composite function with terminal rewards ±500, gentle shaping rewards, fuel efficiency bonus on success only

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
