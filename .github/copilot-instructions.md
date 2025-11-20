# AI Lunar Lander - Copilot Instructions

## Project Overview
This is a reinforcement learning project for training AI agents to perform autonomous lunar landings using:
- **Basilisk**: High-fidelity astrodynamics simulation framework (6-DOF spacecraft dynamics)
- **Stable Baselines3**: PyTorch-based RL library (PPO, SAC, TD3 algorithms)
- **Gymnasium**: Environment interface for RL agents
- **Starship HLS**: Realistic lunar lander configuration (1.3M kg initial mass, 3 Raptor engines)

## Architecture & Component Boundaries

### Three-Layer Architecture
1. **Simulation Layer** (`ScenarioLunarLanderStarter.py`): Basilisk physics simulation
   - Spacecraft dynamics with fuel depletion (`FuelTank` effectors for CH4/LOX)
   - Lunar gravity field and analytical terrain model (`LunarRegolithModel`)
   - Sensor systems: IMU, LIDAR (64-ray cone scan), fuel gauges
   - Thruster controller mapping actions to 3 Raptor engines + RCS

2. **Environment Layer** (`lunar_lander_env.py`): Gymnasium wrapper
   - Translates Basilisk simulation into Gym API (`reset()`, `step()`, render)
   - **Observation space**: 32D compact (position, velocity, Euler angles, IMU, fuel flow rate, time-to-impact, LIDAR azimuthal bins)
   - **Action space**: 15D (primary throttles×3, gimbals×6, mid-body groups×3, RCS groups×3)
   - **Action smoothing**: Exponential moving average (80% old, 20% new) for stable control
   - **Reward shaping**: Comprehensive multi-component architecture:
     - Terminal rewards (±1000): Dominant signals for episode outcomes
     - Progress tracking (0-5): Continuous guidance toward landing
     - Safety penalties (±2): Danger zone warnings and efficiency
     - Control quality (±1): Smooth control and technique optimization
   - **Termination**: Success (altitude < 5m, velocity < 3 m/s, upright), crash, or timeout

3. **Training Layer** (`unified_training.py`): RL orchestration
   - **Curriculum learning**: 5 progressive stages (simple landing → extreme conditions)
   - **Advancement criteria**: Mean reward > threshold AND success rate > 60% (stage regression if repeated failures)
   - **Observation normalization**: `VecNormalize` wrapper for zero-mean, unit-variance inputs (critical for PPO/SAC/TD3)
   - Multi-algorithm support (PPO default, SAC for sample efficiency, TD3 for deterministic control)
   - Parallel environments (`SubprocVecEnv` with `--n-envs 4-16`, default 12)
   - Checkpointing every 100k steps, best model auto-save via `EvalCallback` every 10k steps

### Key Integration Points
- **Basilisk ↔ Gymnasium**: `LunarLanderEnv._create_simulation()` initializes Basilisk sim, updates via `scSim.ExecuteSimulation()` at 0.1s timestep
- **Optimized reset**: Uses Basilisk's state engine to directly update state values without re-initialization (eliminates warnings, 100x faster)
- **Observation normalization**: `VecNormalize` wrapper applied to training environments for zero-mean, unit-variance observations (improves stability)
- **Curriculum advancement**: Requires BOTH mean reward > threshold AND 60%+ success rate; supports stage regression on repeated failures
- **Terrain generation**: `generate_terrain.py` creates realistic lunar surfaces with craters, boulders, ejecta; loaded by `LunarRegolithModel.load_terrain_from_file()`

## Critical Developer Workflows

### Setup & Dependencies
```powershell
# Basilisk must be installed separately (NOT included in repository)
# Option 1: Install from PyPI (recommended)
pip install Basilisk

# Option 2: Build from source
# See: https://hanspeterschaub.info/basilisk/
# Ensure built dist3 directory is in your Python path

# Install Python dependencies
pip install stable-baselines3[extra] gymnasium numpy matplotlib tensorboard
```

### Training Commands (Primary Workflow)
```powershell
# 1. Test setup (2 min sanity check)
python unified_training.py --mode test

# 2. Curriculum training (OPTIMIZED for i7-14700K + RTX 4080, 2-4 hrs)
python unified_training.py --mode curriculum --n-envs 12

# 3. Monitor via TensorBoard (separate terminal)
tensorboard --logdir=./logs

# 4. Monitor GPU usage (separate terminal - if using GPU)
nvidia-smi -l 1

# 5. Evaluate trained model
python unified_training.py --mode eval --model-path ./models/curriculum_final --eval-episodes 20
```

### Quick Development Iteration
```powershell
# Fast demo for code changes (15 min)
python unified_training.py --mode demo

# Single-stage testing (avoid full curriculum)
python unified_training.py --mode standard --timesteps 100000
```

### Debugging Environment Issues
```python
# Standalone environment test (bottom of lunar_lander_env.py)
python lunar_lander_env.py

# Check Gymnasium API compliance
from stable_baselines3.common.env_checker import check_env
env = LunarLanderEnv()
check_env(env)
```

## Project-Specific Conventions

### Reward Function Philosophy (Critical for Tuning)
**Located in**: `lunar_lander_env.py::_compute_reward()`

**Comprehensive Multi-Component Architecture:**
- **Terminal rewards**: ±1000 (10x larger than shaping) to dominate episode outcome
  - Success +1000, precision +200, fuel efficiency +150 (quadratic, ONLY on success), softness +100, attitude +100, control smoothness +50
- **Progress tracking**: 0-5 per step (continuous guidance) - descent profile, approach angle, proximity, attitude stability, final approach quality
- **Safety penalties**: ±2 per step - danger zone warnings (speed/tilt/lateral), fuel management, loitering penalty
- **Control quality**: ±1 per step - effort, jitter, spin rate penalties
- **Success window**: 0-5m altitude (realistic), velocity < 3 m/s, horizontal < 2 m/s, attitude < 15°
- **Fuel efficiency bonus**: +150 points (quadratic curve) ONLY on successful landing (prevents hoarding during flight)
- **Crash penalty gradient**: Scales with impact severity (-400 to -800)

**Expected Cumulative Rewards:**
- Perfect landing: 1200-1600 (all bonuses)
- Good landing: 900-1200
- Basic landing: 600-900
- Poor landing: 400-600
- Crash: -400 to -800

**DO NOT** add generic "exploration bonuses" - terrain randomization provides sufficient diversity

**For detailed tuning guide, see**: `REWARD_SYSTEM_GUIDE.md`

### Curriculum Stage Design Pattern
**Located in**: `unified_training.py::_create_curriculum()`
Each `CurriculumStage` requires:
- `env_config`: Dict passed to `LunarLanderEnv.__init__()` (NOT arbitrary parameters)
- `success_threshold`: Mean reward over `min_episodes` to advance (POSITIVE values target successful landings)
- `min_episodes`: Minimum episodes before checking advancement (200-400 for proper mastery)
- `max_timesteps`: Hard cap per stage (300k-800k to prevent overfitting)

**Advancement logic**: Requires BOTH mean reward > threshold AND success rate > 60%. Supports stage regression (go back one stage) if repeated failures.

**Convention**: Stages progressively increase `initial_altitude_range` (30m → 22000m), `num_craters` (0 → 25), reduce velocity tolerance

### Basilisk Integration Quirks
1. **Path setup required**: All scripts must import Basilisk as a Python module (install via `pip install Basilisk` or build from source)
2. **Time units**: Basilisk uses nanoseconds (`macros.sec2nano(0.1)` for 0.1s timestep)
3. **Coordinate frames**: 
   - `_N`: Moon-centered inertial frame (North-East-Down)
   - `_B`: Body-fixed frame (+Z is nose/up, +X forward)
   - **Critical**: LIDAR scans in body frame, positions in inertial frame
4. **Fuel depletion**: `FuelTank` effectors auto-update spacecraft mass/inertia - DO NOT manually modify `lander.hub.mHub`
5. **Reset optimization**: Use state engine (`stateEngine.setState()`) to update states directly - avoids re-initialization warnings and is 100x faster

### Action Space Mapping
**Located in**: `lunar_lander_env.py::step()`
```python
# action = [primary_throttles (3), primary_gimbals (6), midbody_groups (3), rcs_groups (3)]
# Total: 15D action space
# - Primary throttles: [0.4-1.0] for 3 Raptor engines
# - Gimbals: [-8°, +8°] pitch/yaw per engine (6 total)
# - Mid-body groups: [0, 1] for rotation control (3)
# - RCS groups: [0, 1] for pitch/yaw/roll (3)
# Action smoothing applied: 80% old action + 20% new action (exponential moving average)
```
**Why comprehensive**: 15D action space provides full pilot-level control authority; action smoothing improves stability (configurable via `action_smooth_alpha`)

### File Output Structure (Auto-generated)
```
models/
├── best_model/           # Best model via EvalCallback (auto-saved every 10k eval steps)
├── checkpoints/          # Every 100k steps (ppo_lunar_lander_NNNNNN_steps.zip)
├── curriculum_final.zip  # Final curriculum model
└── stage*_final.zip      # Per-stage completions

logs/
└── PPO_1/                # TensorBoard logs (timestamped subdirs)
```

## Common Pitfalls & Solutions

### "Module not found: Basilisk"
**Cause**: Basilisk not installed
**Fix**: Install Basilisk:
```python
# Option 1: Install from PyPI
pip install Basilisk

# Option 2: Build from source
# See: https://hanspeterschaub.info/basilisk/
```

### Agent Not Learning (Reward < -200)
1. **Check curriculum advancement**: Use `--mode curriculum` (NOT `--mode standard` initially)
2. **Verify environment termination**: Ensure episodes end (check `max_episode_steps` in `LunarLanderEnv`)
3. **Inspect TensorBoard**: Look for `rollout/ep_rew_mean` trend over 100k+ steps AND `episode/success_rate_100`
4. **Verify observation normalization**: Ensure `VecNormalize` is applied (check `unified_training.py::_normalize_env()`)
5. **Increase training time**: Try longer per-stage training or more episodes
6. **Check success rate**: If reward positive but success rate low, stage may need more training time

### Training Crashes on Step
**Likely**: Basilisk simulation divergence due to extreme actions
**Debug**: Add reward clipping in `_compute_reward()` or increase action smoothing (not currently implemented)

### Slow Training (< 1000 steps/sec)
- **Reduce parallel envs**: `--n-envs 2` on low-memory systems
- **Use SAC/TD3**: More sample-efficient than PPO (`--algorithm sac`)
- **Disable logging**: Comment out verbose TensorBoard logging in callbacks

## Key Files for Common Tasks

### Modifying Reward Function
Edit: `lunar_lander_env.py::_compute_reward()`
Test: `python lunar_lander_env.py` (runs 5-step sanity check)

### Adding Curriculum Stages
Edit: `unified_training.py::_create_curriculum()`
Test: `python unified_training.py --mode curriculum --start-stage N` (skip to stage N)

### Changing Terrain Difficulty
1. Generate new terrain: `python generate_terrain.py --craters 50 --size 5000`
2. Update `terrain_config` in curriculum stages or env init

### Hyperparameter Tuning
Edit: `unified_training.py::_create_model()` (learning_rate, n_steps, batch_size, etc.)
Reference: SB3 docs for algorithm-specific params

## Testing Philosophy
- **No unit tests** (physics simulation hard to mock)
- **Integration testing**: `--mode test` runs 1 episode with random actions
- **Validation**: Curriculum auto-advances only when reward threshold met across min_episodes
- **Visual inspection**: Use `--mode eval --render` to watch agent behavior

## Performance Expectations
- **Curriculum training**: 2-4 hours on i7-14700K + RTX 4080 (10-20x faster than CPU-only)
- **Training speed**: 5,000-15,000 steps/sec with GPU acceleration (vs 500-1,500 on CPU)
- **Final success rate**: 60-70% successful landings on extreme terrain (curriculum requires 60% to advance)
- **Mean reward**: 800-1200 (includes terminal 1000 + bonuses up to 400)
- **Landing criteria**: Altitude < 5m, velocity < 3 m/s, horizontal speed < 2 m/s, attitude < 15°
- **Fuel efficiency**: Variable (bonus +150 awarded only on successful landing)

## Hardware Optimizations (Applied)
**For high-end systems (i7-14700K + RTX 4080 + 64GB RAM):**
- ✅ **GPU acceleration**: `device='cuda'` (3-5x speedup)
- ✅ **Parallel environments**: Default `n_envs=12` (2-3x speedup)
- ✅ **Batch sizes**: PPO 512, SAC 1024, TD3 512 (1.5-2x speedup)
- ✅ **PPO n_steps**: 4096 (2x from 2048)
- ✅ **Curriculum timesteps**: 2x increase per stage (200k-800k)
- ✅ **Checkpoint frequency**: 100k steps (reduced I/O)
- **Combined speedup**: 10-20x faster training

## When Asking Questions
1. **For training issues**: Include TensorBoard metrics (`rollout/ep_rew_mean` trend) and command used
2. **For environment errors**: Run `python lunar_lander_env.py` and paste full traceback
3. **For reward tuning**: Specify target behavior (e.g., "prioritize fuel over precision")
4. **For curriculum changes**: State initial/target altitude, terrain difficulty, expected stage duration
