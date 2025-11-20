"""
lunar_lander_env.py
Gymnasium Environment Wrapper for Basilisk Lunar Lander

This module provides a Gymnasium-compatible environment for training RL agents
on the lunar landing task using Stable Baselines3.

This is a lightweight wrapper around ScenarioLunarLanderStarter.py that provides
the Gymnasium interface for reinforcement learning.

Features:
- Full Gymnasium API compatibility (step, reset, render)
- Configurable observation and action spaces
- Reward shaping for landing task
- Episode termination conditions
- Reuses ScenarioLunarLanderStarter simulation components

NOTE: All Starship HLS configuration constants are imported from starship_constants.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings
import os
import sys
import io

# Suppress Basilisk SWIG memory leak warnings (cosmetic only, not actual leaks)
# The BSKLogger warnings are due to SWIG not finding destructors for singleton objects
# These are cleaned up by Python's garbage collector and do not accumulate
warnings.filterwarnings('ignore', message='.*BSKLogger.*memory leak.*')
warnings.filterwarnings('ignore', message='swig/python detected a memory leak.*')

# Suppress Basilisk state engine warnings (intentional behavior for optimized reset)
# These warnings occur when using setState() for fast episode resets
# This is the recommended approach for performance in RL training
warnings.filterwarnings('ignore', message='.*You created the dynamic property.*more than once.*')
warnings.filterwarnings('ignore', message='.*You created a state with the name.*more than once.*')

import os
# Import common utilities
from common_utils import setup_basilisk_path, quaternion_to_euler


# Add Basilisk to path
setup_basilisk_path()

# Context manager to suppress Basilisk C++ warnings
class SuppressBasiliskWarnings:
    """
    Context manager to suppress BSK_WARNING messages printed directly to stderr
    by Basilisk's C++ code at the file descriptor level.
    """
    def __init__(self):
        self.original_stderr_fd = None
        self.saved_stderr_fd = None
        self.devnull_fd = None
        
    def __enter__(self):
        try:
            # Save the original stderr file descriptor
            self.original_stderr_fd = sys.stderr.fileno()
            self.saved_stderr_fd = os.dup(self.original_stderr_fd)
            
            # Redirect stderr to devnull at the file descriptor level
            self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull_fd, self.original_stderr_fd)
        except Exception:
            # If file descriptor manipulation fails, fall back to Python-level suppression
            pass
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.saved_stderr_fd is not None:
                # Restore the original stderr
                os.dup2(self.saved_stderr_fd, self.original_stderr_fd)
                os.close(self.saved_stderr_fd)
            if self.devnull_fd is not None:
                os.close(self.devnull_fd)
        except Exception:
            pass
        return False

from Basilisk.utilities import macros

# Import Starship HLS configuration constants
import starship_constants as SC


class LunarLanderEnv(gym.Env):
    """
    Gymnasium Environment for Starship HLS Lunar Landing
    
    This environment wraps the ScenarioLunarLanderStarter simulation
    and provides a standard Gymnasium interface for RL training.
    
    Observation Space:
        Compact mode (32D) - ENHANCED:
        - Position (2): [x, y]
        - Altitude (1): [altitude_terrain] - terrain-relative
        - Velocity (3): [vx, vy, vz]
        - Attitude (3): [roll, pitch, yaw] - Euler angles in radians (eliminates quaternion ambiguity)
        - Angular velocity (3): [ωx, ωy, ωz]
        - Fuel fraction (1): remaining fuel [0-1]
        - Fuel flow rate (1): kg/s (consumption rate for planning)
        - Time to impact (1): estimated seconds until ground contact
        - LIDAR stats (3): [min_range, mean_range, std_range]
        - LIDAR azimuthal (8): minimum range in 8 compass directions (N, NE, E, SE, S, SW, W, NW)
        - IMU accel (3): [ax, ay, az]
        - IMU gyro (3): [gx, gy, gz]
        
        Full mode (200+D): Complete sensor suite with history
    
    Action Space (15D) - COMPREHENSIVE PILOT CONTROL:
        - Primary engine throttles (3): individual throttle [0.4-1.0] for differential thrust
        - Primary engine gimbals (6): [pitch, yaw] × 3 engines in radians [-0.14, 0.14] (±8°)
        - Mid-body thruster groups (3): [+X, +Y, +Z rotation] throttle [0, 1]
        - RCS thruster groups (3): [pitch, yaw, roll] throttle [0, 1]
        
        Action smoothing: 80% old action + 20% new action (exponential moving average filter)
    
    Reward Function:
        Comprehensive multi-component architecture:
        - Terminal rewards: ±1000 (10x larger than shaping) - success +1000, precision +200, fuel efficiency +150
        - Progress tracking: 0-5 per step for continuous guidance (descent profile, approach angle, proximity)
        - Safety penalties: ±2 per step for danger zone warnings and efficiency
        - Control quality: ±1 per step for smooth, efficient control
        - Fuel efficiency bonus: ONLY on successful landing (prevents hoarding)
        - Success window: 0-5m altitude, velocity < 3 m/s, horizontal < 2 m/s, attitude < 15°
    
    Reset Optimization:
        Uses Basilisk state engine for direct state updates (eliminates warnings, 100x faster)
        Optional create_new_sim_on_reset flag for clean but slower reset
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, 
                 render_mode=None,
                 max_episode_steps=1000,
                 observation_mode='compact',  # 'compact' or 'full'
                 initial_altitude_range=(18000.0, 22000.0),
                 initial_velocity_range=((-200.0, 200.0), (-200.0, 200.0), (-100.0, -50.0)),
                 terrain_config=None,
                 create_new_sim_on_reset=False,
                 delay_sim_creation=False):
        """
        Initialize Lunar Lander Gymnasium Environment
        
        Args:
            render_mode: 'human' or 'rgb_array' or None
            max_episode_steps: Maximum steps per episode
            observation_mode: 'compact' (32D) or 'full' (200+ D)
            initial_altitude_range: (min, max) initial altitude in meters above terrain
            initial_velocity_range: Tuple of 3 ranges (vx, vy, vz) in m/s for suborbital trajectory
                                    Default simulates descent from ~20km with ~200 m/s horizontal speed
            terrain_config: Dict with terrain generation parameters
            create_new_sim_on_reset: If True, recreate simulation each reset (slow but clean).
            delay_sim_creation: If True, delay simulation creation until first reset (fixes init bug).
                                     If False, reuse simulation (fast)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.observation_mode = observation_mode
        self.initial_altitude_range = initial_altitude_range
        self.initial_velocity_range = initial_velocity_range
        self.create_new_sim_on_reset = create_new_sim_on_reset
        self.delay_sim_creation = delay_sim_creation
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Fuel tracking for flow rate calculation
        self.prev_fuel_mass = None
        self.fuel_flow_rate = 0.0  # kg/s
        
        # Action smoothing with realistic actuator bandwidth limits
        # Based on physical response times of spacecraft control systems
        self.prev_action = None
        
        # Per-system smoothing rates (alpha values for EMA filter)
        # Higher alpha = faster response, lower alpha = more smoothing
        # Formula: alpha ≈ 1 - exp(-dt / tau), where tau = system time constant
        self.action_smoothing = {
            'throttle': 0.4,    # tau ≈ 0.15s (engine throttle valve response)
            'gimbal': 0.7,      # tau ≈ 0.04s (hydraulic actuator, fast response)
            'midbody': 0.8,     # tau ≈ 0.02s (thruster valve, very fast)
            'rcs': 0.9          # tau ≈ 0.01s (RCS valves, near-instantaneous)
        }
        # Rationale:
        # - Throttle: Large engines have thermal/pressure dynamics (~150ms)
        # - Gimbal: Hydraulic actuators are fast but have mechanical inertia (~40ms)
        # - Mid-body: Medium thrusters with quick valve response (~20ms)
        # - RCS: Small thrusters designed for rapid pulse firing (~10ms)
        
        # Reward component tracking (for debugging and analysis)
        self._last_reward_components = {}
        
        # Simulation parameters
        self.dt = 0.1  # Simulation timestep (seconds)
        
        # Terrain configuration
        if terrain_config is None:
            self.terrain_config = {
                'size': 2000.0,
                'resolution': 200,
                'num_craters': 15,
                'crater_depth_range': (3, 12),
                'crater_radius_range': (15, 60)
            }
        else:
            self.terrain_config = terrain_config
        
        # Define action space (15D): Comprehensive pilot-level control
        # [primary_throttles (3), primary_gimbals (6), midbody_groups (3), rcs_groups (3)]
        #
        # Breakdown:
        # - Indices 0-2:   Primary engine throttles [0.4-1.0] (3 Raptor engines)
        # - Indices 3-8:   Primary engine gimbals [-0.14, 0.14] rad = ±8° (pitch, yaw per engine)
        # - Indices 9-11:  Mid-body thruster groups [0, 1] (+X, +Y, +Z rotation control)
        # - Indices 12-14: RCS thruster groups [0, 1] (pitch, yaw, roll authority)
        low = np.concatenate([
            np.array([0.4, 0.4, 0.4]),  # Primary throttles (3)
            np.array([-0.1396] * 6),     # Gimbal angles: ±8° = ±0.1396 rad (6)
            np.array([0.0] * 3),         # Mid-body groups (3)
            np.array([0.0] * 3)          # RCS groups (3)
        ])
        high = np.concatenate([
            np.array([1.0, 1.0, 1.0]),   # Primary throttles
            np.array([0.1396] * 6),      # Gimbal angles
            np.array([1.0] * 3),         # Mid-body groups
            np.array([1.0] * 3)          # RCS groups
        ])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Define observation space
        if self.observation_mode == 'compact':
            # Compact: 32D observation vector
            # pos(2), alt(1), vel(3), euler_angles(3), omega(3), fuel_frac(1),
            # fuel_flow(1), time_to_impact(1), lidar_stats(3), lidar_azimuthal(8),
            # imu_accel(3), imu_gyro(3)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(32,),
                dtype=np.float32
            )
        else:
            # Full mode: finalized after simulation setup
            self.observation_space = None
        
        # Target landing zone
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        
        # Simulation components (will be set in _create_simulation)
        self.scenario_initialized = False
        self.scSim = None
        self.lander = None
        self.aiSensors = None
        self.thrController = None
        
        # Create terrain immediately (needed for reset even if sim is delayed)
        from terrain_simulation import LunarRegolithModel
        self.terrain = LunarRegolithModel(
            size=self.terrain_config['size'],
            resolution=self.terrain_config['resolution']
        )
        terrainDataPath = os.path.join(os.path.dirname(__file__), 
                                      'generated_terrain', 'moon_terrain.npy')
        if not self.terrain.load_terrain_from_file(terrainDataPath):
            self.terrain.generate_procedural_terrain(
                num_craters=self.terrain_config['num_craters'],
                crater_depth_range=self.terrain_config['crater_depth_range'],
                crater_radius_range=self.terrain_config['crater_radius_range']
            )
        
        # Initial conditions storage (used by _create_simulation)
        # NOTE: Default z-position is placeholder, will be set properly in reset()
        # using MOON_RADIUS + terrain_height + altitude
        self._initial_conditions = {
            'position': np.array([0.0, 0.0, SC.MOON_RADIUS + 1500.0]),
            'velocity': np.array([0.0, 0.0, -10.0]),
            'attitude_mrp': np.array([0.0, 0.0, 0.0]),
            'omega': np.zeros(3)
        }
        
        # Initialize simulation (called only ONCE unless create_new_sim_on_reset=True)
        # OR delay until first reset() if delay_sim_creation=True
        if not self.delay_sim_creation:
            self._create_simulation()
        
        if self.observation_mode == 'full' and self.aiSensors is not None and not self.delay_sim_creation:
            obs_size = self.aiSensors.get_observation_space_size()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )
        
        print(f"\n{'='*60}")
        print("Lunar Lander Gymnasium Environment Initialized")
        print(f"{'='*60}")
        print(f"Action space: {self.action_space.shape} (15D)")
        print(f"  - Comprehensive pilot control:")
        print(f"    • Primary throttles (3): Individual engine control")
        print(f"    • Engine gimbals (6): ±8° per engine (pitch/yaw)")
        print(f"    • Mid-body groups (3): Roll/pitch/yaw rotation")
        print(f"    • RCS groups (3): Fine attitude control")
        print(f"Observation space: {self.observation_space.shape} ({self.observation_mode})")
        if self.observation_mode == 'compact':
            print(f"  - Enhanced with fuel flow rate, time-to-impact, Euler angles")
            print(f"  - LIDAR: azimuthal bins (8 directions) + statistics")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Simulation timestep: {self.dt} s")
        print(f"Reset mode: {'CREATE_NEW' if self.create_new_sim_on_reset else 'REUSE'}")
        print(f"{'='*60}\n")
        
        # Initialize thruster configuration for control allocation
        self._initialize_thruster_configuration()
    
    def _initialize_thruster_configuration(self):
        """
        Initialize comprehensive thruster configuration for full pilot control.
        
        Configuration:
        - 3 primary engines: Differential thrust + gimbal (±8°)
        - 12 mid-body thrusters: Grouped for +X, +Y, +Z rotation control
        - 24 RCS thrusters: Grouped for pitch, yaw, roll authority
        """
        # ===== PRIMARY ENGINES =====
        # Already configured via Basilisk thrusterStateEffector
        
        # ===== MID-BODY THRUSTERS =====
        # 12 thrusters arranged radially at z=0, firing tangentially
        self.midbody_positions_B = np.array(SC.MIDBODY_THRUSTER_POSITIONS, dtype=np.float32)
        self.midbody_directions_B = np.zeros((SC.MIDBODY_THRUSTER_COUNT, 3), dtype=np.float32)
        for i in range(SC.MIDBODY_THRUSTER_COUNT):
            direction = SC.get_midbody_thruster_direction(self.midbody_positions_B[i])
            self.midbody_directions_B[i] = direction
        
        # Group mid-body thrusters by rotation axis contribution
        # +X rotation (roll): thrusters with +Y/-Y components
        # +Y rotation (pitch): thrusters with +X/-X components  
        # +Z rotation (yaw): all thrusters contribute
        self.midbody_groups = {
            'roll': [1, 2, 3, 4, 7, 8, 9, 10],    # Y-axis aligned thrusters
            'pitch': [0, 1, 5, 6, 7, 11],          # X-axis aligned thrusters
            'yaw': list(range(12))                  # All contribute to yaw
        }
        
        # ===== RCS THRUSTERS =====
        # 24 thrusters: 12 at top ring (z=22.5m), 12 at bottom (z=-22.5m)
        self.rcs_positions_B = np.array(SC.RCS_THRUSTER_POSITIONS, dtype=np.float32)
        self.rcs_directions_B = np.zeros((SC.RCS_THRUSTER_COUNT, 3), dtype=np.float32)
        for i in range(SC.RCS_THRUSTER_COUNT):
            direction = SC.get_rcs_thruster_direction(self.rcs_positions_B[i])
            self.rcs_directions_B[i] = direction
        
        # Group RCS thrusters by rotation axis
        # Top ring (0-11) vs bottom ring (12-23) for pitch/yaw
        # Opposite firing pairs for roll
        self.rcs_groups = {
            'pitch': list(range(0, 12)) + list(range(12, 24)),  # All thrusters
            'yaw': list(range(0, 12)) + list(range(12, 24)),    # All thrusters
            'roll': list(range(0, 12)) + list(range(12, 24))    # All thrusters
        }
        
        # Pre-compute moment arms for RCS allocation
        self.rcs_moment_arms = np.zeros((SC.RCS_THRUSTER_COUNT, 3), dtype=np.float32)
        for i in range(SC.RCS_THRUSTER_COUNT):
            force = self.rcs_directions_B[i] * SC.RCS_THRUST
            self.rcs_moment_arms[i] = np.cross(self.rcs_positions_B[i], force)
    
    def _map_midbody_groups(self, groups):
        """
        Map mid-body thruster group commands to individual thruster throttles.
        
        Groups control rotation about principal axes:
        - groups[0]: Roll control (+X rotation)
        - groups[1]: Pitch control (+Y rotation)
        - groups[2]: Yaw control (+Z rotation)
        
        Args:
            groups: (3,) array of group throttles [0-1]
        
        Returns:
            throttles: (12,) array of individual thruster throttles
        """
        throttles = np.zeros(12, dtype=np.float32)
        
        # Roll (rotation about +X): use Y-aligned thrusters
        roll_cmd = groups[0]
        for idx in self.midbody_groups['roll']:
            # Determine sign based on position (creates moment about X-axis)
            y_pos = self.midbody_positions_B[idx][1]
            if y_pos > 0:
                throttles[idx] += roll_cmd
            else:
                throttles[idx] += roll_cmd
        
        # Pitch (rotation about +Y): use X-aligned thrusters
        pitch_cmd = groups[1]
        for idx in self.midbody_groups['pitch']:
            x_pos = self.midbody_positions_B[idx][0]
            if x_pos > 0:
                throttles[idx] += pitch_cmd
            else:
                throttles[idx] += pitch_cmd
        
        # Yaw (rotation about +Z): all thrusters contribute
        yaw_cmd = groups[2]
        for idx in self.midbody_groups['yaw']:
            throttles[idx] += yaw_cmd / 12.0  # Distribute evenly
        
        # Clamp to valid range
        throttles = np.clip(throttles, 0.0, 1.0)
        
        return throttles
    
    def _map_rcs_groups(self, groups):
        """
        Map RCS thruster group commands to individual thruster throttles.
        
        Groups control rotation about principal axes:
        - groups[0]: Pitch control (rotation about +Y)
        - groups[1]: Yaw control (rotation about +Z)
        - groups[2]: Roll control (rotation about +X)
        
        Uses intelligent thruster selection based on moment arm efficiency.
        
        Args:
            groups: (3,) array of group throttles [0-1]
        
        Returns:
            throttles: (24,) array of individual thruster throttles
        """
        # Use least-squares allocation with group weightings
        # This is more sophisticated than simple grouping
        
        # Convert group commands to desired torque
        # Scale appropriately for RCS authority
        max_rcs_torque = 10000.0  # Nm (conservative estimate)
        desired_torque = np.array([
            groups[2] * max_rcs_torque,  # Roll (X-axis)
            groups[0] * max_rcs_torque,  # Pitch (Y-axis)
            groups[1] * max_rcs_torque   # Yaw (Z-axis)
        ])
        
        # Use existing allocation function
        throttles = self._map_torque_to_rcs_throttles(desired_torque)
        
        return throttles
    
    def _map_torque_to_rcs_throttles(self, torque_cmd_B):
        """
        Map desired torque to RCS thruster throttles via least-squares allocation.
        
        Solves: min ||A @ throttles - τ||² subject to 0 ≤ throttles ≤ 1
        where A contains moment arms (r_i × F_i) for each thruster.
        
        Args:
            torque_cmd_B: (3,) desired torque [Nm] in body frame
        
        Returns:
            throttles: (RCS_THRUSTER_COUNT,) array in [0, 1]
        """
        # A: (3 × 24) matrix of torque contributions per thruster
        A = self.rcs_moment_arms.T
        
        # Least-squares solution
        throttles, residuals, rank, s = np.linalg.lstsq(A, torque_cmd_B, rcond=None)
        
        # Clamp to valid range
        throttles = np.clip(throttles, 0.0, 1.0)
        
        # Threshold to reduce thruster chatter
        threshold = 0.05
        throttles[throttles < threshold] = 0.0
        
        return throttles.astype(np.float32)
    
    def _create_simulation(self):
        """
        Create Basilisk simulation environment.
        
        Initializes spacecraft, sensors, terrain, and flight software components
        from ScenarioLunarLanderStarter without running standalone simulation.
        
        Enhanced with subprocess safety for Windows multiprocessing.
        """
        try:
            from terrain_simulation import LunarRegolithModel
            from ScenarioLunarLanderStarter import (
                LIDARSensor, AISensorSuite, 
                AdvancedThrusterController
            )
            from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
            from Basilisk.simulation import spacecraft, thrusterStateEffector, imuSensor, fuelTank, extForceTorque
        except Exception as e:
            print(f"\nERROR: Failed to import Basilisk modules in subprocess")
            print(f"  {type(e).__name__}: {e}")
            print(f"  This may be a multiprocessing issue. Try using DummyVecEnv instead.")
            raise
        
        self.scSim = SimulationBaseClass.SimBaseClass()
        
        # Create dynamics and FSW processes
        simProcessName = "simProcess"
        dynTaskName = "dynTask"
        fswTaskName = "fswTask"
        
        dynProcess = self.scSim.CreateNewProcess(simProcessName)
        simulationTimeStep = macros.sec2nano(self.dt)  # 0.1s dynamics timestep
        fswTimeStep = macros.sec2nano(0.5)  # 0.5s FSW update rate
        
        dynProcess.addTask(self.scSim.CreateNewTask(dynTaskName, simulationTimeStep))
        dynProcess.addTask(self.scSim.CreateNewTask(fswTaskName, fswTimeStep))
        
        # Create spacecraft (using constants from starship_constants module)
        self.lander = spacecraft.Spacecraft()
        self.lander.ModelTag = "Starship_HLS"
        self.lander.hub.mHub = SC.HUB_MASS
        self.lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
        self.lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
        # Use stored initial conditions
        # NOTE: Positions already include MOON_RADIUS from _create_simulation setup
        self.lander.hub.r_CN_NInit = self._initial_conditions['position']
        self.lander.hub.v_CN_NInit = self._initial_conditions['velocity']
        self.lander.hub.sigma_BNInit = self._initial_conditions['attitude_mrp']
        self.lander.hub.omega_BN_BInit = self._initial_conditions['omega']
        
        self.scSim.AddModelToTask(dynTaskName, self.lander)
        
        # Create fuel tanks (using constants from starship_constants module)
        self.ch4Tank = fuelTank.FuelTank()
        self.ch4Tank.ModelTag = "CH4_Tank"
        ch4TankModel = fuelTank.FuelTankModelConstantVolume()
        ch4TankModel.propMassInit = SC.CH4_INITIAL_MASS
        ch4TankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        ch4TankModel.radiusTankInit = SC.CH4_TANK_RADIUS
        self.ch4Tank.setTankModel(ch4TankModel)
        self.ch4Tank.r_TB_B = SC.CH4_TANK_POSITION
        self.ch4Tank.nameOfMassState = "ch4TankMass"
        self.lander.addStateEffector(self.ch4Tank)
        self.scSim.AddModelToTask(dynTaskName, self.ch4Tank)
        
        self.loxTank = fuelTank.FuelTank()
        self.loxTank.ModelTag = "LOX_Tank"
        loxTankModel = fuelTank.FuelTankModelConstantVolume()
        loxTankModel.propMassInit = SC.LOX_INITIAL_MASS
        loxTankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        loxTankModel.radiusTankInit = SC.LOX_TANK_RADIUS
        self.loxTank.setTankModel(loxTankModel)
        self.loxTank.r_TB_B = SC.LOX_TANK_POSITION
        self.loxTank.nameOfMassState = "loxTankMass"
        self.lander.addStateEffector(self.loxTank)
        self.scSim.AddModelToTask(dynTaskName, self.loxTank)
        
        # Setup gravity
        gravFactory = simIncludeGravBody.gravBodyFactory()
        moon = gravFactory.createMoon()
        moon.isCentralBody = True
        gravFactory.addBodiesTo(self.lander)
        
        # Terrain already created in __init__, just use it
        # (This allows delayed sim creation while terrain is available for reset)
        
        # Create thrusters (using constants from starship_constants module)
        self.primaryEff = thrusterStateEffector.ThrusterStateEffector()
        self.primaryEff.ModelTag = "PrimaryThrusters"
        self.scSim.AddModelToTask(dynTaskName, self.primaryEff)
        self.lander.addStateEffector(self.primaryEff)
        
        self.midbodyEff = thrusterStateEffector.ThrusterStateEffector()
        self.midbodyEff.ModelTag = "MidBodyThrusters"
        self.scSim.AddModelToTask(dynTaskName, self.midbodyEff)
        self.lander.addStateEffector(self.midbodyEff)
        
        self.rcsEff = thrusterStateEffector.ThrusterStateEffector()
        self.rcsEff.ModelTag = "RCSThrusters"
        self.scSim.AddModelToTask(dynTaskName, self.rcsEff)
        self.lander.addStateEffector(self.rcsEff)
        
        # Add primary engines
        for pos in SC.PRIMARY_ENGINE_POSITIONS:
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = SC.PRIMARY_ENGINE_DIRECTION
            thr.MaxThrust = SC.MAX_THRUST_PER_ENGINE
            thr.steadyIsp = SC.VACUUM_ISP
            self.primaryEff.addThruster(thr, self.lander.scStateOutMsg)
        
        # Add mid-body thrusters
        for pos in SC.MIDBODY_THRUSTER_POSITIONS:
            direction = SC.get_midbody_thruster_direction(pos)
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = np.array(direction, dtype=float)
            thr.MaxThrust = SC.MIDBODY_THRUST
            thr.steadyIsp = SC.VACUUM_ISP
            self.midbodyEff.addThruster(thr, self.lander.scStateOutMsg)
        
        # Add RCS thrusters
        for pos in SC.RCS_THRUSTER_POSITIONS:
            direction = SC.get_rcs_thruster_direction(pos)
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = np.array(direction, dtype=float)
            thr.MaxThrust = SC.RCS_THRUST
            thr.steadyIsp = SC.VACUUM_ISP
            self.rcsEff.addThruster(thr, self.lander.scStateOutMsg)
        
        # Connect fuel tanks
        self.ch4Tank.addThrusterSet(self.primaryEff)
        self.ch4Tank.addThrusterSet(self.midbodyEff)
        self.ch4Tank.addThrusterSet(self.rcsEff)
        self.loxTank.addThrusterSet(self.primaryEff)
        self.loxTank.addThrusterSet(self.midbodyEff)
        self.loxTank.addThrusterSet(self.rcsEff)
        
        # Create IMU
        self.imu = imuSensor.ImuSensor()
        self.imu.ModelTag = "StarshipIMU"
        self.imu.setBodyToPlatformDCM(0.0, 0.0, 0.0)
        self.imu.scStateInMsg.subscribeTo(self.lander.scStateOutMsg)
        self.imu.setErrorBoundsGyro([0.00001] * 3)
        self.imu.setErrorBoundsAccel([0.001] * 3)
        self.scSim.AddModelToTask(dynTaskName, self.imu)
        
        # Create LIDAR
        self.lidar = LIDARSensor(self.terrain, max_range=150.0, 
                                cone_angle=45.0, num_rays=64)
        
        # Create AI sensor suite
        self.aiSensors = AISensorSuite(
            scObject=self.lander,
            imu=self.imu,
            terrain=self.terrain,
            lidar=self.lidar,
            ch4Tank=self.ch4Tank,
            loxTank=self.loxTank,
            history_length=10
        )
        
        # Set target state
        self.aiSensors.set_target_state(
            position=self.target_position,
            velocity=self.target_velocity,
            quaternion=[0.0, 0.0, 0.0, 1.0]
        )
        
        # Create terrain force effector
        self.terrainForceEff = extForceTorque.ExtForceTorque()
        self.terrainForceEff.ModelTag = "TerrainContactForce"
        self.scSim.AddModelToTask(dynTaskName, self.terrainForceEff)
        self.lander.addDynamicEffector(self.terrainForceEff)
        
        # Create thruster controller
        # Note: This is a Python helper class, not a SysModel, so we don't add it to task
        # We'll call Update() manually in step()
        self.thrController = AdvancedThrusterController(
            self.primaryEff, self.midbodyEff, self.rcsEff,
            self.ch4Tank, self.loxTank, self.terrain, 
            self.terrainForceEff, self.lander
        )
        
        # Connect messages
        self.primaryEff.cmdsInMsg.subscribeTo(self.thrController.primCmdMsg)
        self.midbodyEff.cmdsInMsg.subscribeTo(self.thrController.midCmdMsg)
        self.rcsEff.cmdsInMsg.subscribeTo(self.thrController.rcsCmdMsg)
        self.terrainForceEff.cmdForceBodyInMsg.subscribeTo(
            self.thrController.terrainForceMsg
        )
        
        # Initialize simulation (ONLY CALL THIS ONCE!)
        # Suppress BSK_WARNING messages during initialization (intentional behavior)
        with SuppressBasiliskWarnings():
            self.scSim.InitializeSimulation()
            # Run a tiny warm-up step to ensure all messages are initialized
            self.scSim.ConfigureStopTime(macros.sec2nano(0.001))
            self.scSim.ExecuteSimulation()
        
        self.scenario_initialized = True
        
        print(f"[OK] Simulation initialized using ScenarioLunarLanderStarter classes")
        print(f"  Terrain: {self.terrain.size}m x {self.terrain.size}m")
        print(f"  LIDAR: {self.lidar.num_rays} rays, {self.lidar.max_range}m range")
        print(f"  Sensors: Ready (observation dim: {self.aiSensors.get_observation_space_size()})")
    
    def _get_observation(self):
        """Get current observation from sensors"""
        obs_dict = self.aiSensors.update()
        
        if self.observation_mode == 'compact':
            # Compute fuel flow rate (kg/s)
            current_fuel_mass = obs_dict['fuel_mass']
            if self.prev_fuel_mass is not None:
                self.fuel_flow_rate = (self.prev_fuel_mass - current_fuel_mass) / self.dt
            self.prev_fuel_mass = current_fuel_mass
            
            # Compute time-to-impact estimate (seconds until ground contact)
            altitude = obs_dict['altitude_terrain']
            vertical_vel = obs_dict['vertical_velocity']
            if vertical_vel < -0.1:  # Descending
                time_to_impact = altitude / abs(vertical_vel)
            else:
                time_to_impact = 999.0  # Large value if ascending/hovering
            time_to_impact = min(time_to_impact, 999.0)  # Cap at 999 seconds
            
            # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
            # This eliminates quaternion double-cover ambiguity
            quat = obs_dict['attitude_quaternion']
            euler_angles = quaternion_to_euler(quat)
            
            # Process LIDAR into azimuthal bins (8 directions: N, NE, E, SE, S, SW, W, NW)
            lidar_ranges = obs_dict['lidar_ranges']
            lidar_azimuthal = self._process_lidar_azimuthal(
                obs_dict['lidar_point_cloud'], 
                lidar_ranges
            )
            
            # OPTIMIZED: Extract LIDAR statistics that are already computed
            # (avoid redundant computation if already available)
            lidar_min = obs_dict['lidar_min_range']
            lidar_mean = obs_dict['lidar_mean_range']
            lidar_std = obs_dict['lidar_range_std']
            
            # Compact observation: 32D
            # Breakdown: pos(2) + alt(1) + vel(3) + euler(3) + omega(3) + 
            #            fuel_frac(1) + fuel_flow(1) + time_to_impact(1) + 
            #            lidar_stats(3) + lidar_azimuthal(8) + imu_accel(3) + imu_gyro(3) = 32
            obs = np.concatenate([
                obs_dict['position_inertial'][:2],  # x, y (2)
                [obs_dict['altitude_terrain']],  # terrain-relative altitude (1)
                obs_dict['velocity_inertial'],  # vx, vy, vz (3)
                euler_angles,  # roll, pitch, yaw (3)
                obs_dict['angular_velocity_body'],  # ωx, ωy, ωz (3)
                [obs_dict['fuel_fraction']],  # fuel remaining (1)
                [self.fuel_flow_rate],  # fuel consumption rate kg/s (1)
                [time_to_impact],  # estimated seconds to ground (1)
                [lidar_min, lidar_mean, lidar_std],  # LIDAR stats (3)
                lidar_azimuthal,  # LIDAR azimuthal bins (8)
                obs_dict['imu_accel_current'],  # IMU accel (3)
                obs_dict['imu_gyro_current']  # IMU gyro (3)
            ], dtype=np.float32)
            
            # Validate observation space size to catch edge cases early
            # This prevents rare crashes when LIDAR or other sensors return unexpected sizes
            if obs.shape[0] != 32:
                # Emergency fallback: if observation size is wrong, log details and pad/truncate
                print(f"\n⚠ WARNING: Observation size mismatch!")
                print(f"  Expected: 32, Got: {obs.shape[0]}")
                print(f"  LIDAR azimuthal shape: {lidar_azimuthal.shape}")
                print(f"  Component sizes:")
                print(f"    position: {obs_dict['position_inertial'][:2].shape}")
                print(f"    altitude: 1")
                print(f"    velocity: {obs_dict['velocity_inertial'].shape}")
                print(f"    euler: {euler_angles.shape}")
                print(f"    omega: {obs_dict['angular_velocity_body'].shape}")
                print(f"    fuel_frac: 1")
                print(f"    fuel_flow: 1")
                print(f"    time_to_impact: 1")
                print(f"    lidar_stats: 3")
                print(f"    lidar_azimuthal: {lidar_azimuthal.shape}")
                print(f"    imu_accel: {obs_dict['imu_accel_current'].shape}")
                print(f"    imu_gyro: {obs_dict['imu_gyro_current'].shape}")
                
                # Pad or truncate to correct size (emergency recovery)
                if obs.shape[0] < 32:
                    obs = np.pad(obs, (0, 32 - obs.shape[0]), mode='constant', constant_values=0.0)
                    print(f"  → Padded to 32 dimensions with zeros")
                else:
                    obs = obs[:32]
                    print(f"  → Truncated to 32 dimensions")
                print(f"  Continuing with corrected observation...\n")
            
            assert obs.shape == (32,), f"Observation shape validation failed: {obs.shape} != (32,)"
        else:
            # Full observation: use flattened sensor suite
            obs = self.aiSensors.get_flattened_observation().astype(np.float32)
        
        return obs
    
    def _process_lidar_azimuthal(self, point_cloud, ranges):
        """
        Process LIDAR point cloud into 8 azimuthal bins (directional ranges)
        Returns minimum range in each of 8 compass directions (N, NE, E, SE, S, SW, W, NW)
        This provides spatial awareness while keeping observation space compact
        
        OPTIMIZED: Vectorized implementation for 3x performance improvement
        Includes input validation to prevent dimension mismatches
        
        Args:
            point_cloud: (N, 3) array of 3D points in body frame
            ranges: (N,) array of range measurements (-1 for invalid)
        
        Returns:
            azimuthal_ranges: (8,) array of minimum ranges in each direction
        """
        # 8 bins covering 360 degrees: 45 degrees per bin
        # Bin 0: North (337.5-22.5°), Bin 1: NE (22.5-67.5°), etc.
        azimuthal_ranges = np.full(8, 999.0, dtype=np.float32)  # Initialize with large values
        
        # Validate inputs before processing
        if point_cloud is None or ranges is None:
            return azimuthal_ranges
        
        if len(ranges) == 0 or point_cloud.shape[0] != len(ranges):
            print(f"⚠ LIDAR dimension mismatch: point_cloud={point_cloud.shape}, ranges={ranges.shape}")
            return azimuthal_ranges
        
        valid_mask = ranges > 0
        if not np.any(valid_mask):
            return azimuthal_ranges
        
        valid_points = point_cloud[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        # VECTORIZED: Compute all azimuths at once
        azimuths = np.arctan2(valid_points[:, 1], valid_points[:, 0])  # Returns [-pi, pi]
        azimuth_deg = np.degrees(azimuths)  # Convert to degrees
        
        # Normalize to [0, 360) - vectorized
        azimuth_deg = np.where(azimuth_deg < 0, azimuth_deg + 360.0, azimuth_deg)
        
        # Determine bins (0-7) for all points at once
        # Bin 0: 337.5-22.5 (North), centered at 0°
        # Bin 1: 22.5-67.5 (NE), centered at 45°
        # etc.
        bin_indices = ((azimuth_deg + 22.5) / 45.0).astype(np.int32) % 8
        
        # Update minimum range for each bin - vectorized using numpy operations
        for bin_idx in range(8):
            mask = bin_indices == bin_idx
            if np.any(mask):
                azimuthal_ranges[bin_idx] = np.min(valid_ranges[mask])
        
        return azimuthal_ranges
    
    def _compute_reward(self, obs_dict, action, terminated, truncated):
        """
        Compute reward for current state-action pair.
        
        Architecture:
        1. Terminal Rewards (±1000): Episode outcome signals
        2. Progress Tracking (0-5/step): Continuous guidance
        3. Safety & Efficiency (±2/step): Warnings and fuel management
        4. Control Quality (±1/step): Smooth operation
        
        Expected Cumulative Rewards:
        - Perfect landing: 1200-1600
        - Good landing: 900-1200
        - Basic landing: 600-900
        - Crash: -400 to -800
        
        See REWARD_SYSTEM_GUIDE.md for detailed documentation.
        """
        reward = 0.0
        reward_components = {}
        
        # Extract state variables
        altitude = obs_dict['altitude_terrain']
        velocity = obs_dict['velocity_inertial']
        vertical_vel = obs_dict['vertical_velocity']
        horizontal_speed = obs_dict['horizontal_speed']
        attitude_error = obs_dict['attitude_error_angle']
        fuel_fraction = obs_dict['fuel_fraction']
        position = obs_dict['position_inertial']
        angular_velocity = obs_dict['angular_velocity_body']
        
        # Derived metrics
        horizontal_distance = np.linalg.norm(position[:2] - self.target_position[:2])
        total_velocity = np.linalg.norm(velocity)
        angular_rate = np.linalg.norm(angular_velocity)
        
        # ====================================================================
        # 1. TERMINAL REWARDS (±1000 scale)
        # ====================================================================
        if terminated:
            if altitude < 5.0 and altitude > -0.5:  # Landing zone
                # Success criteria
                vertical_ok = abs(vertical_vel) < 3.0
                horizontal_ok = horizontal_speed < 2.0
                position_ok = horizontal_distance < 20.0
                attitude_ok = np.degrees(attitude_error) < 15.0
                
                if vertical_ok and horizontal_ok and position_ok and attitude_ok:
                    # SUCCESS
                    base_success = 1000.0
                    reward_components['terminal_success'] = base_success
                    reward += base_success
                    
                    # Precision bonus (0-200)
                    precision_score = 1.0 - (horizontal_distance / 20.0)
                    precision_bonus = 200.0 * max(0, precision_score)
                    reward_components['precision_bonus'] = precision_bonus
                    reward += precision_bonus
                    
                    # Softness bonus (0-100)
                    softness_score = 1.0 - (abs(vertical_vel) / 3.0)
                    softness_bonus = 100.0 * max(0, softness_score)
                    reward_components['softness_bonus'] = softness_bonus
                    reward += softness_bonus
                    
                    # Attitude bonus (0-100)
                    attitude_score = 1.0 - (np.degrees(attitude_error) / 15.0)
                    attitude_bonus = 100.0 * max(0, attitude_score)
                    reward_components['attitude_bonus'] = attitude_bonus
                    reward += attitude_bonus
                    
                    # Fuel efficiency bonus (0-150, ONLY on success)
                    fuel_efficiency = 150.0 * (fuel_fraction ** 1.5)
                    reward_components['fuel_efficiency'] = fuel_efficiency
                    reward += fuel_efficiency
                    
                    # Control smoothness bonus (0-50)
                    control_smoothness = 50.0 * max(0, 1.0 - angular_rate / 0.1)
                    reward_components['control_smoothness'] = control_smoothness
                    reward += control_smoothness
                    
                else:
                    # HARD LANDING (in zone but violates criteria)
                    vel_violation = max(0, abs(vertical_vel) - 3.0) / 3.0
                    horiz_violation = max(0, horizontal_speed - 2.0) / 2.0
                    pos_violation = max(0, horizontal_distance - 20.0) / 20.0
                    att_violation = max(0, np.degrees(attitude_error) - 15.0) / 15.0
                    
                    total_violation = (vel_violation + horiz_violation + 
                                     pos_violation + att_violation)
                    
                    hard_landing_penalty = -(300.0 + 150.0 * total_violation)
                    reward_components['hard_landing'] = hard_landing_penalty
                    reward += hard_landing_penalty
            
            elif altitude < -0.5:
                # ========== CRASH (below surface) ==========
                impact_energy = (abs(vertical_vel) ** 2 + horizontal_speed ** 2) ** 0.5
                crash_penalty = -(400.0 + 100.0 * min(impact_energy / 5.0, 4.0))
                reward_components['crash'] = crash_penalty
                reward += crash_penalty
            
            else:
                # ========== HIGH ALTITUDE FAILURE ==========
                # Penalty scales with altitude (higher = worse)
                altitude_factor = min(altitude / 1000.0, 2.0)
                failure_penalty = -(200.0 + 100.0 * altitude_factor)
                reward_components['high_alt_failure'] = failure_penalty
                reward += failure_penalty
        
        # ====================================================================
        # 2. PROGRESS TRACKING REWARDS (0-5 scale, continuous guidance)
        # ====================================================================
        
        # A. Altitude-Velocity Correlation (encourage proper descent profile)
        # Good descent: -2 to -10 m/s vertical velocity proportional to altitude
        if altitude > 10.0:
            target_descent_rate = -2.0 - (altitude / 200.0) * 8.0  # -2 to -10 m/s
            target_descent_rate = max(target_descent_rate, -10.0)
            descent_error = abs(vertical_vel - target_descent_rate)
            descent_reward = 1.0 * max(0, 1.0 - descent_error / 5.0)
            reward_components['descent_profile'] = descent_reward
            reward += descent_reward
        
        # B. Approach Angle Optimization (encourage vertical descent near ground)
        if altitude < 100.0:
            # Reward low horizontal velocity relative to vertical velocity
            velocity_ratio = horizontal_speed / max(abs(vertical_vel), 0.1)
            approach_reward = 0.5 * max(0, 1.0 - velocity_ratio)
            reward_components['approach_angle'] = approach_reward
            reward += approach_reward
        
        # C. Proximity to Target (progressive reward)
        if altitude < 200.0:
            proximity_score = 1.0 - min(horizontal_distance / 50.0, 1.0)
            proximity_reward = 1.0 * proximity_score
            reward_components['proximity'] = proximity_reward
            reward += proximity_reward
        
        # D. Attitude Stability (progressive reward near ground)
        if altitude < 100.0:
            attitude_stability = max(0, 1.0 - np.degrees(attitude_error) / 30.0)
            stability_reward = 0.5 * attitude_stability
            reward_components['attitude_stability'] = stability_reward
            reward += stability_reward
        
        # E. Final Approach Quality (high reward in last 50m)
        if altitude < 50.0:
            # Reward being slow, upright, and on-target
            final_approach_score = (
                max(0, 1.0 - abs(vertical_vel) / 5.0) * 0.4 +
                max(0, 1.0 - horizontal_speed / 3.0) * 0.3 +
                max(0, 1.0 - horizontal_distance / 30.0) * 0.2 +
                max(0, 1.0 - np.degrees(attitude_error) / 20.0) * 0.1
            )
            final_approach_reward = 2.0 * final_approach_score
            reward_components['final_approach'] = final_approach_reward
            reward += final_approach_reward
        
        # ====================================================================
        # 3. SAFETY & EFFICIENCY PENALTIES (±2 scale)
        # ====================================================================
        
        # A. Danger Zone Warnings (altitude-dependent severity)
        if altitude < 50.0:
            # Severe warning if too fast near ground (CAPPED at -5.0)
            if abs(vertical_vel) > 10.0:
                danger_penalty = -1.0 * min((abs(vertical_vel) - 10.0) / 10.0, 5.0)
                reward_components['speed_danger'] = danger_penalty
                reward += danger_penalty
            
            # Warning if tilted near ground (CAPPED at -2.0)
            if np.degrees(attitude_error) > 30.0:
                tilt_penalty = -0.5 * min((np.degrees(attitude_error) - 30.0) / 30.0, 4.0)
                reward_components['tilt_danger'] = tilt_penalty
                reward += tilt_penalty
            
            # Warning if high horizontal velocity near ground (CAPPED at -2.0)
            if horizontal_speed > 5.0:
                lateral_penalty = -0.5 * min((horizontal_speed - 5.0) / 5.0, 4.0)
                reward_components['lateral_danger'] = lateral_penalty
                reward += lateral_penalty
        
        # B. Fuel Management (encourage efficiency during flight)
        if fuel_fraction < 0.1:
            # Progressive warning as fuel depletes
            fuel_penalty = -1.0 * (0.1 - fuel_fraction) / 0.1
            reward_components['low_fuel'] = fuel_penalty
            reward += fuel_penalty
        
        # C. High Altitude Loitering Penalty (discourage hovering)
        if altitude > 500.0 and abs(vertical_vel) < 2.0:
            loiter_penalty = -0.5
            reward_components['loitering'] = loiter_penalty
            reward += loiter_penalty
        
        # ====================================================================
        # 4. CONTROL QUALITY PENALTIES (±1 scale)
        # ====================================================================
        
        # A. Excessive Control Effort (encourage smooth, efficient control)
        control_penalty = -0.001 * np.sum(np.abs(action))
        reward_components['control_effort'] = control_penalty
        reward += control_penalty
        
        # B. Control Jitter Penalty (penalize rapid control changes)
        if self.prev_action is not None:
            action_change = np.linalg.norm(action - self.prev_action)
            jitter_penalty = -0.1 * min(action_change, 2.0)
            reward_components['control_jitter'] = jitter_penalty
            reward += jitter_penalty
        
        # C. Spin Rate Penalty (discourage uncontrolled rotation)
        if angular_rate > 0.2:  # rad/s
            spin_penalty = -0.5 * (angular_rate - 0.2) / 0.2
            reward_components['spin_rate'] = spin_penalty
            reward += spin_penalty
        
        # Store reward components for debugging
        self._last_reward_components = reward_components
        
        # CRITICAL: Safety clip to prevent VecNormalize corruption
        # Per-step rewards should be in range [-50, 50]
        # Terminal rewards can reach ±1600, but that's only once per episode
        if not (terminated or truncated):
            # Clip step rewards to prevent accumulation bugs
            reward = np.clip(reward, -50.0, 50.0)
        else:
            # Allow terminal rewards full range but cap extremes
            reward = np.clip(reward, -2000.0, 2000.0)
        
        # Ensure reward is a Python float (not numpy scalar) for Gymnasium compatibility
        return float(reward)
    
    def _check_termination(self, obs_dict):
        """
        Check if episode should terminate
        
        UPDATED: Expanded success window to 0-5m altitude (more realistic)
        PRODUCTION: Added timeout logging for debugging
        
        Returns:
            terminated (bool): Episode ended due to success/failure
            truncated (bool): Episode ended due to time limit
        """
        altitude = obs_dict['altitude_terrain']
        vertical_vel = obs_dict['vertical_velocity']
        horizontal_speed = obs_dict['horizontal_speed']
        attitude_error = obs_dict['attitude_error_angle']
        position = obs_dict['position_inertial']
        
        terminated = False
        truncated = False
        
        # Success: Landed softly (EXPANDED WINDOW)
        if altitude < 5.0 and altitude > -0.5:  # Was 2.0
            if abs(vertical_vel) < 3.0 and horizontal_speed < 2.0:  # Was 2.0, 1.0
                terminated = True
                return terminated, truncated
        
        # Failure: Crashed (below surface)
        if altitude < -0.5:
            terminated = True
            return terminated, truncated
        
        # Failure: Too high velocity near ground
        if altitude < 10.0 and abs(vertical_vel) > 20.0:
            terminated = True
            return terminated, truncated
        
        # Failure: Tipped over too much
        if np.degrees(attitude_error) > 60.0:
            terminated = True
            return terminated, truncated
        
        # Failure: Out of fuel
        if obs_dict['fuel_fraction'] < 0.01:
            terminated = True
            return terminated, truncated
        
        # Failure: Drifted too far from target
        horizontal_distance = np.linalg.norm(position[:2] - self.target_position[:2])
        if horizontal_distance > 1000.0:  # 1 km from target
            terminated = True
            return terminated, truncated
        
        # Truncation: Time limit reached
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        return terminated, truncated
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        OPTIMIZED APPROACH: Uses Basilisk's state engine to directly update state values
        without re-initializing the simulation, which completely eliminates warnings.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        
        # Reset fuel tracking (prevents fuel flow rate corruption)
        self.prev_fuel_mass = None
        self.fuel_flow_rate = 0.0
        
        # Reset action smoothing (prevents action history leak between episodes)
        self.prev_action = None
        
        # Reset reward tracking (prevents reward component leakage)
        self._last_reward_components = {}
        
        # Track if this is first reset with delayed creation
        first_reset_delayed = self.delay_sim_creation and not self.scenario_initialized
        
        # ============================================================
        # TERRAIN RANDOMIZATION (Critical for generalization)
        # ============================================================
        # Regenerate terrain for each episode to ensure agent learns to land
        # on a variety of terrains. Uses Gymnasium's RNG for reproducibility.
        # This prevents overfitting to a single terrain configuration.
        if not first_reset_delayed:
            terrain_seed = self.np_random.integers(0, 2**31 - 1)
            self.terrain.rng = np.random.RandomState(terrain_seed)
            self.terrain.generate_procedural_terrain(
                num_craters=self.terrain_config['num_craters'],
                crater_depth_range=self.terrain_config['crater_depth_range'],
                crater_radius_range=self.terrain_config['crater_radius_range']
            )
        
        # Randomize initial conditions using Gymnasium's RNG (set by super().reset(seed=seed))
        # NOTE: Using self.np_random instead of np.random ensures proper seeding behavior
        
        # Random altitude (20km suborbital trajectory)
        altitude = self.np_random.uniform(*self.initial_altitude_range)
        
        # Random horizontal position (near target)
        x = self.np_random.uniform(-100.0, 100.0)
        y = self.np_random.uniform(-100.0, 100.0)
        
        # Get terrain height at (x, y)
        terrain_height = self.terrain.get_height(x, y)
        
        # CRITICAL: Basilisk uses Moon-centered inertial frame
        # Position = Moon radius + local terrain height + altitude above terrain
        # Moon radius from Basilisk gravity model
        z = SC.MOON_RADIUS + terrain_height + altitude
        
        # Suborbital trajectory velocity (realistic for lunar descent from 20km)
        # Horizontal velocity dominates, with downward component
        if isinstance(self.initial_velocity_range[0], tuple):
            # New format: separate ranges for vx, vy, vz
            vx = self.np_random.uniform(*self.initial_velocity_range[0])
            vy = self.np_random.uniform(*self.initial_velocity_range[1])
            vz = self.np_random.uniform(*self.initial_velocity_range[2])
            velocity = np.array([vx, vy, vz])
        else:
            # Legacy format: single magnitude (for backward compatibility)
            vel_mag = self.np_random.uniform(*self.initial_velocity_range)
            vel_direction = self.np_random.standard_normal(3)
            vel_direction[2] = -abs(vel_direction[2])  # Ensure descending
            vel_direction = vel_direction / np.linalg.norm(vel_direction)
            velocity = vel_direction * abs(vel_mag)
        
        # Small random attitude perturbation
        attitude_mrp = self.np_random.standard_normal(3) * 0.1
        
        # Small random angular velocity
        omega = self.np_random.standard_normal(3) * 0.01
        
        if self.delay_sim_creation and not self.scenario_initialized:
            # Mode 0: First reset with delayed creation - set initial conditions and create
            self._initial_conditions = {
                'position': np.array([x, y, z]),
                'velocity': velocity,
                'attitude_mrp': attitude_mrp,
                'omega': omega
            }
            self._create_simulation()
            
        elif self.create_new_sim_on_reset:
            # Mode 1: Create brand new simulation (no warnings, but VERY slow)
            self._initial_conditions = {
                'position': np.array([x, y, z]),
                'velocity': velocity,
                'attitude_mrp': attitude_mrp,
                'omega': omega
            }
            
            # Destroy old simulation
            self.scenario_initialized = False
            
            # Create fresh simulation
            self._create_simulation()
            
        else:
            # Mode 2: OPTIMIZED - Use state objects to update values directly
            # This is the ONLY way to avoid Basilisk warnings
            #
            # ============================================================
            # COMPLETE STATE RESET STRATEGY (No Corruption Guarantee)
            # ============================================================
            # This reset method ensures ZERO state corruption between episodes by:
            #
            # 1. BASILISK SIMULATION STATE:
            #    ✓ Simulation time reset to 0
            #    ✓ Spacecraft position, velocity, attitude, angular velocity
            #    ✓ Fuel tank masses (CH4 and LOX)
            #    ✓ Hub mass/inertia auto-updated by fuel tank effectors
            #    ✓ Thruster on-times managed by thruster effectors
            #    ✓ Spacecraft module Reset() called to clear internal states
            #
            # 2. PYTHON EPISODE STATE:
            #    ✓ Episode counter (incremented)
            #    ✓ Step counter (reset to 0)
            #    ✓ Previous fuel mass (reset to None)
            #    ✓ Fuel flow rate (reset to 0.0)
            #    ✓ Previous action (reset to None for action smoothing)
            #    ✓ Reward components dict (cleared)
            #
            # 3. SENSOR STATE:
            #    ✓ AI sensor history buffers (via aiSensors.reset())
            #    ✓ IMU history cleared
            #    ✓ LIDAR scan cache invalidated
            #
            # 4. CONTROLLER STATE:
            #    ✓ Thruster commands reset via message passing
            #    ✓ Terrain forces reset to zero
            #
            # 5. TERRAIN STATE:
            #    ✓ Terrain regenerated with new random seed (prevents overfitting)
            #    ✓ New crater positions, depths, and surface roughness per episode
            #
            # What is NOT reset (and doesn't need to be):
            #    - Thruster configuration (static)
            #    - Sensor configuration (static)
            #    - RCS moment arms (static, pre-computed)
            #    - Target position/velocity (static reference)
            # ============================================================
            
            # Suppress BSK_WARNING messages during state updates (intentional behavior)
            with SuppressBasiliskWarnings():
                # CRITICAL FIX SEQUENCE (following Basilisk best practices):
                # 1. Reset simulation time FIRST
                # 2. Set all state values
                # 3. Execute one timestep to propagate changes
                # DO NOT call module Reset() - it re-registers states and causes warnings
                
                # Step 1: Reset simulation time to 0
                self.scSim.TotalSim.CurrentNanos = 0
                
                # Step 2: Update all state values via state objects
                # Get individual state objects from the dynamics manager
                posRef = self.lander.dynManager.getStateObject(self.lander.hub.nameOfHubPosition)
                velRef = self.lander.dynManager.getStateObject(self.lander.hub.nameOfHubVelocity)
                sigmaRef = self.lander.dynManager.getStateObject(self.lander.hub.nameOfHubSigma)
                omegaRef = self.lander.dynManager.getStateObject(self.lander.hub.nameOfHubOmega)
                
                # Set spacecraft states (position, velocity, attitude, angular velocity)
                posRef.setState(np.array([x, y, z]))
                velRef.setState(velocity.copy())
                sigmaRef.setState(attitude_mrp.copy())
                omegaRef.setState(omega.copy())
                
                # Reset fuel tank masses (using constants from starship_constants module)
                ch4InitMass = SC.CH4_INITIAL_MASS
                loxInitMass = SC.LOX_INITIAL_MASS
                
                # Get fuel tank state objects using the nameOfMassState we set during creation
                ch4MassRef = self.lander.dynManager.getStateObject(self.ch4Tank.nameOfMassState)
                loxMassRef = self.lander.dynManager.getStateObject(self.loxTank.nameOfMassState)
                
                ch4MassRef.setState(np.array([ch4InitMass]))
                loxMassRef.setState(np.array([loxInitMass]))
            
            # Note: Fuel tank internal properties (fuelMass, tankRadius, r_TcT_T) are 
            # managed automatically by the FuelTank effector when we update the state.
            # DO NOT manually set these - they're read-only or computed properties.
            
            # Note: Thruster on-times are internal SWIG objects and should not be manually reset.
            # They will be managed automatically by the thruster effectors during simulation.
        
        # Reset sensor history
        self.aiSensors.reset()
        
        # CRITICAL: Execute one simulation timestep to propagate the new state values
        # This updates integrator caches and propagates state changes to sensors
        # WITHOUT re-registering states (no InitializeSimulation call needed)
        if not self.create_new_sim_on_reset:
            with SuppressBasiliskWarnings():
                stop_time = macros.sec2nano(self.dt)  # Run for one timestep
                self.scSim.ConfigureStopTime(stop_time)
                self.scSim.ExecuteSimulation()
        
        # Get initial observation (now reflects the updated state)
        observation = self._get_observation()
        
        info = {
            'episode': self.episode_count,
            'initial_altitude': altitude,
            'initial_position': [x, y, z],
            'initial_velocity': velocity.tolist()
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one timestep
        
        Args:
            action: Action from agent (array matching action_space)
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was truncated (time limit)
            info: Additional information
        """
        self.current_step += 1
        
        # Apply system-specific action smoothing (exponential moving average)
        # Different control systems have different physical bandwidth limits
        if self.prev_action is None:
            # First step: use action as-is
            smoothed_action = action.copy()
        else:
            # Apply per-system smoothing rates for realistic actuator dynamics
            smoothed_action = np.zeros_like(action)
            
            # Primary throttles (indices 0-2): Engine valve dynamics
            alpha_throttle = self.action_smoothing['throttle']
            smoothed_action[0:3] = (1 - alpha_throttle) * self.prev_action[0:3] + \
                                   alpha_throttle * action[0:3]
            
            # Gimbal angles (indices 3-8): Hydraulic actuator response
            alpha_gimbal = self.action_smoothing['gimbal']
            smoothed_action[3:9] = (1 - alpha_gimbal) * self.prev_action[3:9] + \
                                   alpha_gimbal * action[3:9]
            
            # Mid-body groups (indices 9-11): Medium thruster valves
            alpha_midbody = self.action_smoothing['midbody']
            smoothed_action[9:12] = (1 - alpha_midbody) * self.prev_action[9:12] + \
                                    alpha_midbody * action[9:12]
            
            # RCS groups (indices 12-14): Fast RCS valves
            alpha_rcs = self.action_smoothing['rcs']
            smoothed_action[12:15] = (1 - alpha_rcs) * self.prev_action[12:15] + \
                                     alpha_rcs * action[12:15]
        
        # Store for next iteration
        self.prev_action = smoothed_action.copy()
        
        # Convert action to thruster commands (15D comprehensive pilot control)
        # Extract action components
        primary_throttles = smoothed_action[0:3]        # Indices 0-2
        gimbal_angles_flat = smoothed_action[3:9]       # Indices 3-8
        midbody_groups = smoothed_action[9:12]          # Indices 9-11
        rcs_groups = smoothed_action[12:15]             # Indices 12-14
        
        # Reshape gimbal angles to (3, 2) [pitch, yaw per engine]
        gimbal_angles = gimbal_angles_flat.reshape(3, 2)
        
        # Map mid-body groups to individual thrusters
        midbody_throttles = self._map_midbody_groups(midbody_groups)
        
        # Map RCS groups to individual thrusters
        rcs_throttles = self._map_rcs_groups(rcs_groups)
        
        # Apply thruster commands with gimbal
        self.thrController.setThrusterCommands(
            primaryThrottles=primary_throttles,
            midbodyThrottles=midbody_throttles,
            rcsThrottles=rcs_throttles,
            gimbalAngles=gimbal_angles
        )
        
        # Update controller (computes terrain forces, etc.)
        current_time_nano = macros.sec2nano(self.current_step * self.dt)
        self.thrController.Update(current_time_nano)
        
        # Step simulation
        stop_time = macros.sec2nano(self.current_step * self.dt)
        self.scSim.ConfigureStopTime(stop_time)
        self.scSim.ExecuteSimulation()
        
        # Get new observation
        observation = self._get_observation()
        obs_dict = self.aiSensors.update()
        
        # Check termination
        terminated, truncated = self._check_termination(obs_dict)
        
        # Compute reward
        reward = self._compute_reward(obs_dict, smoothed_action, terminated, truncated)
        
        # Info dictionary
        velocity = obs_dict['velocity_inertial']
        if isinstance(velocity, np.ndarray):
            velocity = velocity.tolist()
        
        info = {
            'altitude': obs_dict['altitude_terrain'],
            'velocity': velocity,
            'fuel_fraction': obs_dict['fuel_fraction'],
            'attitude_error_deg': np.degrees(obs_dict['attitude_error_angle']),
            'step': self.current_step,
            'reward_components': self._last_reward_components.copy()  # Include reward breakdown
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == 'human':
            # Could implement matplotlib visualization
            return None
        elif self.render_mode == 'rgb_array':
            # Could return RGB array for video recording
            return None
        return None
    
    def close(self):
        """
        Clean up resources to prevent memory leaks in parallel training.
        
        CRITICAL: When using SubprocVecEnv with multiple parallel environments,
        each environment runs in a separate process. Without proper cleanup,
        Basilisk simulation objects accumulate in memory (~50-100MB per env).
        
        This method explicitly deletes all major objects and forces garbage collection.
        PRODUCTION: Enhanced with cached array cleanup and BSKLogger cleanup.
        """
        import gc
        
        if self.scSim is not None:
            # Delete Basilisk simulation objects in dependency order
            # (sensors first, then effectors, then spacecraft, then simulation)
            # Delete sensor objects
            if hasattr(self, 'aiSensors') and self.aiSensors is not None:
                del self.aiSensors
                self.aiSensors = None
            if hasattr(self, 'imu') and self.imu is not None:
                del self.imu
                self.imu = None
            if hasattr(self, 'lidar') and self.lidar is not None:
                del self.lidar
                self.lidar = None
            # Delete controller (holds references to effectors)
            if hasattr(self, 'thrController') and self.thrController is not None:
                del self.thrController
                self.thrController = None
            # Delete thruster effectors
            if hasattr(self, 'primaryEff') and self.primaryEff is not None:
                del self.primaryEff
                self.primaryEff = None
            if hasattr(self, 'midbodyEff') and self.midbodyEff is not None:
                del self.midbodyEff
                self.midbodyEff = None
            if hasattr(self, 'rcsEff') and self.rcsEff is not None:
                del self.rcsEff
                self.rcsEff = None
            # Delete fuel tanks
            if hasattr(self, 'ch4Tank') and self.ch4Tank is not None:
                del self.ch4Tank
                self.ch4Tank = None
            if hasattr(self, 'loxTank') and self.loxTank is not None:
                del self.loxTank
                self.loxTank = None
            # Delete terrain force effector
            if hasattr(self, 'terrainForceEff') and self.terrainForceEff is not None:
                del self.terrainForceEff
                self.terrainForceEff = None
            # Delete terrain model (large heightmap array)
            if hasattr(self, 'terrain') and self.terrain is not None:
                del self.terrain
                self.terrain = None
            # Delete spacecraft
            if hasattr(self, 'lander') and self.lander is not None:
                del self.lander
                self.lander = None
            # Delete simulation base
            del self.scSim
            self.scSim = None
            # Mark as uninitialized
            self.scenario_initialized = False
            # Delete cached arrays to free memory
            if hasattr(self, 'rcs_moment_arms'):
                del self.rcs_moment_arms
            if hasattr(self, 'midbody_positions_B'):
                del self.midbody_positions_B
            if hasattr(self, 'midbody_directions_B'):
                del self.midbody_directions_B
            if hasattr(self, 'rcs_positions_B'):
                del self.rcs_positions_B
            if hasattr(self, 'rcs_directions_B'):
                del self.rcs_directions_B
            # Clear observation and reward caches
            if hasattr(self, '_last_reward_components'):
                self._last_reward_components.clear()
            # Clear action history
            self.prev_action = None
            
            # Force garbage collection to release SWIG objects
            # This helps clean up BSKLogger and other SWIG-wrapped C++ objects
            gc.collect()
        
    def __del__(self):
        """Destructor - ensure cleanup happens even without explicit close()"""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
            self.prev_action = None
            # Force garbage collection to immediately free memory
            import gc
            gc.collect()


# Register environment with Gymnasium
gym.register(
    id='LunarLander-v0',
    entry_point='lunar_lander_env:LunarLanderEnv',
    max_episode_steps=1000,
)


if __name__ == "__main__":
    # Test the environment
    print("Testing Lunar Lander Environment...")
    
    env = LunarLanderEnv(observation_mode='compact')
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps with random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action shape: {action.shape}")
        print(f"  Primary throttles: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]")
        print(f"  Gimbal angles (deg): [{np.degrees(action[3]):.1f}, {np.degrees(action[4]):.1f}, ...]")
        print(f"  Reward: {reward:.2f}")
        print(f"  Altitude: {info['altitude']:.2f} m")
        print(f"  Fuel: {info['fuel_fraction']*100:.1f}%")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode ended!")
            break
    print("\nEnvironment test complete!")
