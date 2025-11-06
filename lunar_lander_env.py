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
import sys
import os

# Import common utilities
from common_utils import setup_basilisk_path, suppress_basilisk_warnings, quaternion_to_euler

# Add Basilisk to path
setup_basilisk_path()

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
    
    Action Space:
        Compact mode (4D):
        - Main throttle (1): average throttle [0.4-1.0]
        - Attitude control (3): [pitch, yaw, roll] torque commands [-1, 1]
        
        Full mode (9D):
        - Primary thrusters (3): throttle [0.4-1.0] for each engine
        - RCS thrusters (6): simplified control [-1, 1]
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, 
                 render_mode=None,
                 max_episode_steps=1000,
                 action_mode='compact',  # 'compact' or 'full'
                 observation_mode='compact',  # 'compact' or 'full'
                 initial_altitude_range=(1000.0, 2000.0),
                 initial_velocity_range=(-50.0, 50.0),
                 terrain_config=None,
                 create_new_sim_on_reset=False):  # NEW: Control reset behavior
        """
        Initialize Lunar Lander Gymnasium Environment
        
        Args:
            render_mode: 'human' or 'rgb_array' or None
            max_episode_steps: Maximum steps per episode
            action_mode: 'compact' (4D) or 'full' (9D) action space
            observation_mode: 'compact' (23D) or 'full' (200+ D)
            initial_altitude_range: (min, max) initial altitude in meters
            initial_velocity_range: (min, max) initial velocity magnitude
            terrain_config: Dict with terrain generation parameters
            create_new_sim_on_reset: If True, recreate simulation each reset (slow but clean).
                                     If False, reuse simulation (fast but causes Basilisk warnings)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.initial_altitude_range = initial_altitude_range
        self.initial_velocity_range = initial_velocity_range
        self.create_new_sim_on_reset = create_new_sim_on_reset
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Fuel tracking for flow rate calculation
        self.prev_fuel_mass = None
        self.fuel_flow_rate = 0.0  # kg/s
        
        # Action smoothing for stable control
        self.prev_action = None
        self.action_smooth_alpha = 0.2  # 20% new action, 80% old action
        
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
        
        # Define action space
        if self.action_mode == 'compact':
            # Compact: [main_throttle, pitch_torque, yaw_torque, roll_torque]
            self.action_space = spaces.Box(
                low=np.array([0.4, -1.0, -1.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
        elif self.action_mode == 'full':
            # Full: [thr0, thr1, thr2, rcs0, rcs1, rcs2, rcs3, rcs4, rcs5]
            low = np.concatenate([
                np.array([0.4, 0.4, 0.4]),  # Primary throttles
                np.array([-1.0] * 6)  # RCS controls
            ])
            high = np.concatenate([
                np.array([1.0, 1.0, 1.0]),
                np.array([1.0] * 6)
            ])
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
        
        # Define observation space (will be finalized after simulation setup)
        if self.observation_mode == 'compact':
            # Compact: 32D observation (upgraded from 23D)
            # Breakdown: pos(2) + alt(1) + vel(3) + euler(3) + omega(3) + 
            #            fuel_frac(1) + fuel_flow(1) + time_to_impact(1) + 
            #            lidar_stats(3) + lidar_azimuthal(8) + imu_accel(3) + imu_gyro(3) = 32
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(32,),
                dtype=np.float32
            )
        else:
            # Will be set in _create_simulation
            self.observation_space = None
        
        # Target landing zone
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        
        # Simulation components (will be set in _create_simulation)
        self.scenario_initialized = False
        self.scSim = None
        self.lander = None
        self.terrain = None
        self.aiSensors = None
        self.thrController = None
        
        # Initial conditions storage (used by _create_simulation)
        self._initial_conditions = {
            'position': np.array([0.0, 0.0, 1500.0]),
            'velocity': np.array([0.0, 0.0, -10.0]),
            'attitude_mrp': np.array([0.0, 0.0, 0.0]),
            'omega': np.zeros(3)
        }
        
        # Initialize simulation (called only ONCE unless create_new_sim_on_reset=True)
        self._create_simulation()
        
        if self.observation_mode == 'full' and self.aiSensors is not None:
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
        print(f"Action space: {self.action_space.shape} ({self.action_mode})")
        print(f"Observation space: {self.observation_space.shape} ({self.observation_mode})")
        if self.observation_mode == 'compact':
            print(f"  - Enhanced with fuel flow rate, time-to-impact, Euler angles")
            print(f"  - LIDAR: azimuthal bins (8 directions) + statistics")
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Simulation timestep: {self.dt} s")
        print(f"Reset mode: {'CREATE_NEW' if self.create_new_sim_on_reset else 'REUSE'}")
        print(f"{'='*60}\n")
        
        # Initialize RCS thruster configuration for moment arm calculations
        self._initialize_rcs_configuration()
    
    def _initialize_rcs_configuration(self):
        """
        Initialize RCS thruster positions and directions for proper torque-to-throttle mapping.
        Uses configuration from starship_constants module.
        
        RCS Layout:
        - 24 thrusters total: 12 at top ring (z=22.5m), 12 at bottom ring (z=-22.5m)
        - Each ring has 12 thrusters evenly spaced around a 4.2m radius circle
        - All thrusters fire radially outward (tangential to the circle)
        - Each thruster produces 2,000 N
        """
        # RCS thruster positions (from starship_constants module)
        self.rcs_positions_B = np.array(SC.RCS_THRUSTER_POSITIONS, dtype=np.float32)
        
        # RCS thrust directions (radially outward, normalized)
        self.rcs_directions_B = np.zeros((SC.RCS_THRUSTER_COUNT, 3), dtype=np.float32)
        for i in range(SC.RCS_THRUSTER_COUNT):
            # Direction: radially outward in x-y plane (no z component)
            direction = SC.get_rcs_thruster_direction(self.rcs_positions_B[i])
            self.rcs_directions_B[i] = direction
        
        # RCS thruster max force
        self.rcs_max_force = SC.RCS_THRUST  # Newtons
        
        # Pre-compute moment arms for torque calculation
        # Torque = r × F, where r is position and F is force
        self.rcs_moment_arms = np.zeros((SC.RCS_THRUSTER_COUNT, 3), dtype=np.float32)
        for i in range(SC.RCS_THRUSTER_COUNT):
            # Moment arm for each thruster when firing at max thrust
            force = self.rcs_directions_B[i] * self.rcs_max_force
            self.rcs_moment_arms[i] = np.cross(self.rcs_positions_B[i], force)
    
    def _map_torque_to_rcs_throttles(self, torque_cmd_B):
        """
        Map desired torque command to RCS thruster throttles using least-squares allocation.
        
        This solves the inverse problem: given desired torque τ, find thruster throttles f
        such that Σ(r_i × F_i) ≈ τ, where:
        - r_i is the position of thruster i
        - F_i = f_i * direction_i * max_force (f_i ∈ [0, 1])
        - τ is the desired torque vector in body frame
        
        Args:
            torque_cmd_B: (3,) desired torque [Nm] in body frame [pitch, yaw, roll]
                         (Basilisk convention: +X forward, +Y right, +Z down/nose)
        
        Returns:
            throttles: (RCS_THRUSTER_COUNT,) array of RCS throttle values [0, 1]
        """
        # Build the moment arm matrix A where A @ throttles ≈ torque_cmd_B
        # Each column i represents the torque contribution of thruster i at full throttle
        A = self.rcs_moment_arms.T  # Shape: (3, RCS_THRUSTER_COUNT)
        
        # Solve least-squares: min ||A @ throttles - torque_cmd_B||^2
        # Subject to: 0 <= throttles <= 1
        # np.linalg.lstsq solves A @ x = b, so we pass A (not A.T) and torque_cmd_B
        throttles, residuals, rank, s = np.linalg.lstsq(A, torque_cmd_B, rcond=None)
        
        # Clip to valid throttle range [0, 1]
        throttles = np.clip(throttles, 0.0, 1.0)
        
        # Optional: Apply threshold to avoid very small activations (reduces chatter)
        threshold = 0.05  # Only fire thrusters if throttle > 5%
        throttles[throttles < threshold] = 0.0
        
        return throttles.astype(np.float32)
    
    def _create_simulation(self):
        """
        Create Basilisk simulation environment using components from ScenarioLunarLanderStarter
        
        This method creates a fresh simulation using the classes defined in
        ScenarioLunarLanderStarter.py without running the scenario's main simulation.
        """
        # Import the classes we need (not the running scenario)
        from ScenarioLunarLanderStarter import (
            LunarRegolithModel, LIDARSensor, AISensorSuite, 
            AdvancedThrusterController
        )
        from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
        from Basilisk.simulation import spacecraft, thrusterStateEffector, imuSensor, fuelTank, extForceTorque
        
        # Create simulation base
        self.scSim = SimulationBaseClass.SimBaseClass()
        
        # Create processes
        simProcessName = "simProcess"
        dynTaskName = "dynTask"
        fswTaskName = "fswTask"
        
        dynProcess = self.scSim.CreateNewProcess(simProcessName)
        simulationTimeStep = macros.sec2nano(self.dt)
        fswTimeStep = macros.sec2nano(0.5)
        
        dynProcess.addTask(self.scSim.CreateNewTask(dynTaskName, simulationTimeStep))
        dynProcess.addTask(self.scSim.CreateNewTask(fswTaskName, fswTimeStep))
        
        # Create spacecraft (using constants from starship_constants module)
        self.lander = spacecraft.Spacecraft()
        self.lander.ModelTag = "Starship_HLS"
        self.lander.hub.mHub = SC.HUB_MASS
        self.lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
        self.lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
        # Use stored initial conditions
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
        
        # Create terrain
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
        # Suppress expected Basilisk warnings during initialization
        with suppress_basilisk_warnings():
            self.scSim.InitializeSimulation()
        
        # Run a tiny warm-up step to ensure all messages are initialized
        # This prevents "IMUSensorMsgPayload not properly initialized" warning
        with suppress_basilisk_warnings():
            self.scSim.ConfigureStopTime(macros.sec2nano(0.001))
            self.scSim.ExecuteSimulation()
        
        self.scenario_initialized = True
        
        # Store state names for efficient reset
        self._state_names = {
            'position': 'hubPosition',
            'velocity': 'hubVelocity', 
            'attitude': 'hubSigma',
            'omega': 'hubOmega',
            'ch4_mass': 'ch4TankMass',
            'lox_mass': 'loxTankMass'
        }
        
        print(f"✓ Simulation initialized using ScenarioLunarLanderStarter classes")
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
                [obs_dict['lidar_min_range'],  # LIDAR stats (3)
                 obs_dict['lidar_mean_range'],
                 obs_dict['lidar_range_std']],
                lidar_azimuthal,  # LIDAR azimuthal bins (8)
                obs_dict['imu_accel_current'],  # IMU accel (3)
                obs_dict['imu_gyro_current']  # IMU gyro (3)
            ], dtype=np.float32)
        else:
            # Full observation: use flattened sensor suite
            obs = self.aiSensors.get_flattened_observation().astype(np.float32)
        
        return obs
    
    def _process_lidar_azimuthal(self, point_cloud, ranges):
        """
        Process LIDAR point cloud into 8 azimuthal bins (directional ranges)
        Returns minimum range in each of 8 compass directions (N, NE, E, SE, S, SW, W, NW)
        This provides spatial awareness while keeping observation space compact
        
        Args:
            point_cloud: (N, 3) array of 3D points in body frame
            ranges: (N,) array of range measurements (-1 for invalid)
        
        Returns:
            azimuthal_ranges: (8,) array of minimum ranges in each direction
        """
        # 8 bins covering 360 degrees: 45 degrees per bin
        # Bin 0: North (337.5-22.5°), Bin 1: NE (22.5-67.5°), etc.
        azimuthal_ranges = np.full(8, 999.0, dtype=np.float32)  # Initialize with large values
        
        valid_mask = ranges > 0
        if not np.any(valid_mask):
            return azimuthal_ranges
        
        valid_points = point_cloud[valid_mask]
        valid_ranges = ranges[valid_mask]
        
        for i, (point, range_val) in enumerate(zip(valid_points, valid_ranges)):
            # Compute azimuth angle in body frame (x-y plane)
            azimuth = np.arctan2(point[1], point[0])  # Returns [-pi, pi]
            azimuth_deg = np.degrees(azimuth)  # Convert to degrees
            
            # Normalize to [0, 360)
            if azimuth_deg < 0:
                azimuth_deg += 360.0
            
            # Determine bin (0-7)
            # Bin 0: 337.5-22.5 (North), centered at 0°
            # Bin 1: 22.5-67.5 (NE), centered at 45°
            # etc.
            bin_idx = int((azimuth_deg + 22.5) / 45.0) % 8
            
            # Update minimum range for this bin
            azimuthal_ranges[bin_idx] = min(azimuthal_ranges[bin_idx], range_val)
        
        return azimuthal_ranges
    
    def _compute_reward(self, obs_dict, action, terminated, truncated):
        """
        Compute reward for current state-action pair
        
        REBALANCED REWARD DESIGN (Fixed Issues #1-2):
        - Terminal rewards scaled to ±500 (was ±100) to dominate episode outcome
        - Shaping rewards scaled by 0.1x (reduced) for gentler gradient
        - Exponential altitude penalty (worse at high altitudes)
        - Crash penalty scales with severity (worse crashes = worse penalty)
        - Fuel efficiency bonus ONLY on successful landing (avoids hoarding during flight)
        - Success window expanded to 0-5m altitude (more realistic)
        
        Target cumulative reward: 
        - Successful landing: 400-700 (base 500 + bonuses up to 200)
        - Failed landing: -600 to -100
        """
        reward = 0.0
        
        altitude = obs_dict['altitude_terrain']
        velocity = obs_dict['velocity_inertial']
        vertical_vel = obs_dict['vertical_velocity']
        horizontal_speed = obs_dict['horizontal_speed']
        attitude_error = obs_dict['attitude_error_angle']
        fuel_fraction = obs_dict['fuel_fraction']
        position = obs_dict['position_inertial']
        
        # Distance from target (horizontal)
        horizontal_distance = np.linalg.norm(position[:2] - self.target_position[:2])
        
        # 1. SHAPING REWARDS (scaled down by 10x for gentler gradient)
        
        # Exponential altitude penalty (worse at high altitudes, encourages descent)
        if altitude > 10.0:
            # Exponential penalty: -0.001 * exp(alt/500)
            # At 100m: -0.12, at 500m: -0.27, at 1000m: -0.74, at 2000m: -5.4
            altitude_reward = -0.001 * np.exp(altitude / 500.0)
        else:
            # Small reward for being in landing zone
            altitude_reward = 0.05
        
        # Penalize horizontal distance from target (scaled down)
        distance_penalty = -0.005 * min(horizontal_distance, 100.0)  # Cap at 100m
        
        # Penalize high velocities (scaled down)
        velocity_penalty = -0.01 * (abs(vertical_vel) + horizontal_speed)
        
        # Penalize attitude error (want to stay upright, scaled down)
        attitude_penalty = -0.005 * np.degrees(attitude_error)
        
        # Penalize excessive control effort (encourage smooth control, scaled down)
        if self.action_mode == 'compact':
            control_penalty = -0.001 * np.sum(np.abs(action))
        else:
            control_penalty = -0.001 * np.sum(np.abs(action))
        
        # Sum shaping rewards (now much smaller magnitude)
        reward = (altitude_reward + distance_penalty + velocity_penalty + 
                 attitude_penalty + control_penalty)
        
        # 2. TERMINAL REWARDS/PENALTIES (scaled to ±500, 5x larger)
        if terminated:
            # Expanded success window: 0-5m altitude (more realistic)
            if altitude < 5.0 and altitude > -0.5:  # Near surface
                # Successful landing conditions (relaxed)
                if (abs(vertical_vel) < 3.0 and  # Soft touchdown (was 2.0)
                    horizontal_speed < 2.0 and  # Low horizontal speed (was 1.0)
                    horizontal_distance < 20.0 and  # Near target (was 10.0)
                    np.degrees(attitude_error) < 15.0):  # Upright (was 10.0)
                    
                    # SUCCESS: Base reward (5x larger)
                    reward += 500.0
                    
                    # Bonus for precision landing (0-100 points)
                    precision_bonus = 100.0 * max(0, 1.0 - horizontal_distance / 20.0)
                    reward += precision_bonus
                    
                    # Bonus for soft touchdown (0-50 points)
                    softness_bonus = 50.0 * max(0, 1.0 - abs(vertical_vel) / 3.0)
                    reward += softness_bonus
                    
                    # Bonus for upright attitude (0-50 points)
                    attitude_bonus = 50.0 * max(0, 1.0 - np.degrees(attitude_error) / 15.0)
                    reward += attitude_bonus
                    
                    # FUEL EFFICIENCY BONUS (0-100 points) - ONLY for successful landings
                    # This encourages fuel-efficient landings WITHOUT causing hoarding during flight
                    # Scale: 0% fuel = 0 bonus, 50% fuel = +50, 100% fuel = +100
                    fuel_efficiency_bonus = 100.0 * fuel_fraction
                    reward += fuel_efficiency_bonus
                    
                else:
                    # HARD LANDING: penalty scales with impact severity (5x larger)
                    crash_severity = (abs(vertical_vel) / 3.0 + 
                                    horizontal_speed / 2.0 + 
                                    np.degrees(attitude_error) / 15.0)
                    reward -= (250.0 + 100.0 * crash_severity)  # -250 to -500
            
            elif altitude < -0.5:
                # CRASH: Below surface - severe penalty with gradient (5x larger)
                crash_severity = abs(vertical_vel) + horizontal_speed * 2.0
                reward -= (250.0 + 50.0 * min(crash_severity, 10.0))  # -250 to -750
            
            else:
                # FAILURE: Terminated at high altitude (timeout, etc.)
                # Penalty scales with how far from landing zone (5x larger)
                failure_penalty = 150.0 + 0.5 * altitude
                reward -= failure_penalty
        
        # 3. DANGER ZONE WARNINGS (small penalties to guide away from crashes, scaled down)
        if altitude < 50.0:
            if abs(vertical_vel) > 10.0:  # Too fast close to ground
                reward -= 0.2
            if np.degrees(attitude_error) > 30.0:  # Too tilted
                reward -= 0.3
        
        # 4. OUT OF FUEL WARNING (scaled down)
        if fuel_fraction < 0.05:  # Less than 5% fuel
            reward -= 0.5  # Penalty to encourage fuel management
        
        return reward
    
    def _check_termination(self, obs_dict):
        """
        Check if episode should terminate
        
        UPDATED: Expanded success window to 0-5m altitude (more realistic)
        
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
        
        # Reset fuel tracking
        self.prev_fuel_mass = None
        self.fuel_flow_rate = 0.0
        
        # Reset action smoothing
        self.prev_action = None
        
        # Randomize initial conditions
        if seed is not None:
            np.random.seed(seed)
        
        # Random altitude
        altitude = np.random.uniform(*self.initial_altitude_range)
        
        # Random horizontal position (near target)
        x = np.random.uniform(-100.0, 100.0)
        y = np.random.uniform(-100.0, 100.0)
        
        # Get terrain height at (x, y)
        terrain_height = self.terrain.get_height(x, y)
        z = terrain_height + altitude
        
        # Random initial velocity
        vel_mag = np.random.uniform(*self.initial_velocity_range)
        vel_direction = np.random.randn(3)
        vel_direction[2] = -abs(vel_direction[2])  # Ensure descending
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        velocity = vel_direction * abs(vel_mag)
        
        # Small random attitude perturbation
        attitude_mrp = np.random.randn(3) * 0.1
        
        # Small random angular velocity
        omega = np.random.randn(3) * 0.01
        
        if self.create_new_sim_on_reset:
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
            # Mode 2: OPTIMIZED - Use state engine to update values directly
            # This is the ONLY way to avoid Basilisk warnings
            
            # Get the state engine from the dynamics manager
            stateEngine = self.lander.dynManager.getStateObject()
            
            # Reset simulation time
            self.scSim.TotalSim.CurrentNanos = 0
            
            # Update spacecraft states using state engine (no re-registration!)
            stateEngine.setState(self._state_names['position'], np.array([x, y, z]))
            stateEngine.setState(self._state_names['velocity'], velocity.copy())
            stateEngine.setState(self._state_names['attitude'], attitude_mrp.copy())
            stateEngine.setState(self._state_names['omega'], omega.copy())
            
            # Reset fuel tank masses (using constants from starship_constants module)
            ch4InitMass = SC.CH4_INITIAL_MASS
            loxInitMass = SC.LOX_INITIAL_MASS
            stateEngine.setState(self._state_names['ch4_mass'], np.array([ch4InitMass]))
            stateEngine.setState(self._state_names['lox_mass'], np.array([loxInitMass]))
            
            # Reset spacecraft hub mass and inertia properties directly
            self.lander.hub.mHub = SC.HUB_MASS
            self.lander.hub.c_B = SC.CENTER_OF_MASS_OFFSET
            self.lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
            
            # Reset fuel tank internal states
            self.ch4Tank.fuelMass = ch4InitMass
            self.ch4Tank.tankRadius = SC.CH4_TANK_RADIUS
            self.ch4Tank.r_TcT_T = np.zeros(3)
            
            self.loxTank.fuelMass = loxInitMass
            self.loxTank.tankRadius = SC.LOX_TANK_RADIUS
            self.loxTank.r_TcT_T = np.zeros(3)
            
            # Reset thruster on-times
            for i in range(3):
                self.primaryEff.thrusterData[i].thrustOnTime = 0.0
            for i in range(12):
                self.midbodyEff.thrusterData[i].thrustOnTime = 0.0
            for i in range(24):
                self.rcsEff.thrusterData[i].thrustOnTime = 0.0
        
        # Reset sensor history
        self.aiSensors.reset()
        
        # Get initial observation (no need for extra simulation step now)
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
        
        # Apply action smoothing (exponential moving average filter)
        # This reduces control oscillations and provides more stable control
        if self.prev_action is None:
            # First step: use action as-is
            smoothed_action = action.copy()
        else:
            # Blend: 80% old action + 20% new action
            smoothed_action = (1.0 - self.action_smooth_alpha) * self.prev_action + \
                             self.action_smooth_alpha * action
        
        # Store for next iteration
        self.prev_action = smoothed_action.copy()
        
        # Convert action to thruster commands
        if self.action_mode == 'compact':
            # Compact: [main_throttle, pitch_torque, yaw_torque, roll_torque]
            main_throttle = smoothed_action[0]
            torque_cmd = smoothed_action[1:4]  # [pitch, yaw, roll] torques in range [-1, 1]
            
            # Set all primary engines to same throttle
            # NOTE: This means we can't use differential throttling for attitude control
            # All attitude control comes from RCS thrusters
            primary_throttles = np.array([main_throttle] * 3)
            
            # Convert normalized torque commands [-1, 1] to actual torque [Nm]
            # Scale by maximum expected torque (empirically chosen)
            # RCS at max can produce ~200 kNm (24 thrusters * 2000N * 4.2m lever arm)
            max_torque = 50000.0  # Nm (conservative to avoid saturation)
            torque_cmd_Nm = torque_cmd * max_torque
            
            # Use proper least-squares allocation to map torque to RCS throttles
            rcs_throttles = self._map_torque_to_rcs_throttles(torque_cmd_Nm)
            
            # Set mid-body to zero (not used in compact mode)
            midbody_throttles = np.zeros(12)
            
        else:
            # Full mode: direct thruster control (no smoothing on individual thrusters)
            primary_throttles = smoothed_action[0:3]
            rcs_throttles = np.zeros(24)
            if len(smoothed_action) > 3:
                rcs_throttles[:min(len(smoothed_action)-3, 24)] = smoothed_action[3:3+min(len(smoothed_action)-3, 24)]
            midbody_throttles = np.zeros(12)
        
        # Apply thruster commands
        self.thrController.setThrusterCommands(
            primaryThrottles=primary_throttles,
            midbodyThrottles=midbody_throttles,
            rcsThrottles=rcs_throttles
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
        reward = self._compute_reward(obs_dict, action, terminated, truncated)
        
        # Info dictionary
        velocity = obs_dict['velocity_inertial']
        if isinstance(velocity, np.ndarray):
            velocity = velocity.tolist()
        
        info = {
            'altitude': obs_dict['altitude_terrain'],
            'velocity': velocity,
            'fuel_fraction': obs_dict['fuel_fraction'],
            'attitude_error_deg': np.degrees(obs_dict['attitude_error_angle']),
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == 'human':
            # Could implement matplotlib visualization
            pass
        elif self.render_mode == 'rgb_array':
            # Could return RGB array for video recording
            pass
        return None
    
    def close(self):
        """Clean up resources"""
        if self.scSim is not None:
            # Clean up Basilisk simulation
            pass


# Register environment with Gymnasium
gym.register(
    id='LunarLander-v0',
    entry_point='lunar_lander_env:LunarLanderEnv',
    max_episode_steps=1000,
)


if __name__ == "__main__":
    # Test the environment
    print("Testing Lunar Lander Environment...")
    
    env = LunarLanderEnv(action_mode='compact', observation_mode='compact')
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps with random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Altitude: {info['altitude']:.2f} m")
        print(f"  Fuel: {info['fuel_fraction']*100:.1f}%")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print("\nEnvironment test complete!")
