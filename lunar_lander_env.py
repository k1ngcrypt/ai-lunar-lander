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
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add Basilisk to path
basiliskPath = os.path.join(os.path.dirname(__file__), 'basilisk', 'dist3')
sys.path.insert(0, basiliskPath)

from Basilisk.utilities import macros


class LunarLanderEnv(gym.Env):
    """
    Gymnasium Environment for Starship HLS Lunar Landing
    
    This environment wraps the ScenarioLunarLanderStarter simulation
    and provides a standard Gymnasium interface for RL training.
    
    Observation Space:
        Compact mode (23D):
        - Position (3): [x, y, altitude_terrain]
        - Velocity (3): [vx, vy, vz]
        - Attitude (4): quaternion [qx, qy, qz, qw]
        - Angular velocity (3): [ωx, ωy, ωz]
        - Fuel fraction (1): remaining fuel [0-1]
        - LIDAR stats (3): [min_range, mean_range, std_range]
        - IMU current (6): [ax, ay, az, gx, gy, gz]
        
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
                 terrain_config=None):
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
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.initial_altitude_range = initial_altitude_range
        self.initial_velocity_range = initial_velocity_range
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
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
            # Compact: 23D observation
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(23,),
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
        
        # Initialize simulation
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
        print(f"Max episode steps: {self.max_episode_steps}")
        print(f"Simulation timestep: {self.dt} s")
        print(f"{'='*60}\n")
    
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
        
        # Create spacecraft
        self.lander = spacecraft.Spacecraft()
        self.lander.ModelTag = "Starship_HLS"
        self.lander.hub.mHub = 105000.0
        self.lander.hub.r_BcB_B = np.zeros(3)
        self.lander.hub.IHubPntBc_B = np.array([[231513125.0, 0.0, 0.0],
                                                [0.0, 231513125.0, 0.0],
                                                [0.0, 0.0, 14276250.0]])
        self.lander.hub.r_CN_NInit = np.array([0., 0., 1500.0])
        self.lander.hub.v_CN_NInit = np.array([0., 0., -10.0])
        self.lander.hub.sigma_BNInit = np.array([0., 0., 0.])
        self.lander.hub.omega_BN_BInit = np.zeros(3)
        
        self.scSim.AddModelToTask(dynTaskName, self.lander)
        
        # Create fuel tanks
        self.ch4Tank = fuelTank.FuelTank()
        self.ch4Tank.ModelTag = "CH4_Tank"
        ch4TankModel = fuelTank.FuelTankModelConstantVolume()
        ch4TankModel.propMassInit = 260869.565
        ch4TankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        ch4TankRadius = (3.0 * 617.005 / (4.0 * np.pi)) ** (1.0/3.0)
        ch4TankModel.radiusTankInit = ch4TankRadius
        self.ch4Tank.setTankModel(ch4TankModel)
        self.ch4Tank.r_TB_B = [[0.0], [0.0], [-10.0]]
        self.ch4Tank.nameOfMassState = "ch4TankMass"
        self.lander.addStateEffector(self.ch4Tank)
        self.scSim.AddModelToTask(dynTaskName, self.ch4Tank)
        
        self.loxTank = fuelTank.FuelTank()
        self.loxTank.ModelTag = "LOX_Tank"
        loxTankModel = fuelTank.FuelTankModelConstantVolume()
        loxTankModel.propMassInit = 939130.435
        loxTankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        loxTankRadius = (3.0 * 823.077 / (4.0 * np.pi)) ** (1.0/3.0)
        loxTankModel.radiusTankInit = loxTankRadius
        self.loxTank.setTankModel(loxTankModel)
        self.loxTank.r_TB_B = [[0.0], [0.0], [-5.0]]
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
        
        # Create thrusters
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
        vacuumIsp = 375.0
        g0 = 9.80665
        maxThrustPerEngine = 2500000.0
        perEngineMassFlow = maxThrustPerEngine / (vacuumIsp * g0)
        mixtureRatio = 3.6
        ch4FlowPerEngine = perEngineMassFlow / (1.0 + mixtureRatio)
        loxFlowPerEngine = perEngineMassFlow * mixtureRatio / (1.0 + mixtureRatio)
        
        enginePositions = [
            np.array([3.500, 0.000, -24.500]),
            np.array([-1.750, 3.031, -24.500]),
            np.array([-1.750, -3.031, -24.500])
        ]
        
        for pos in enginePositions:
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = np.array([0., 0., 1.])
            thr.MaxThrust = maxThrustPerEngine
            thr.steadyIsp = vacuumIsp
            self.primaryEff.addThruster(thr, self.lander.scStateOutMsg)
        
        # Add mid-body thrusters
        midBodyPositions = [
            np.array([4.000, 0.000, 0.000]),
            np.array([3.464, 2.000, 0.000]),
            np.array([2.000, 3.464, 0.000]),
            np.array([0.000, 4.000, 0.000]),
            np.array([-2.000, 3.464, 0.000]),
            np.array([-3.464, 2.000, 0.000]),
            np.array([-4.000, 0.000, 0.000]),
            np.array([-3.464, -2.000, 0.000]),
            np.array([-2.000, -3.464, 0.000]),
            np.array([0.000, -4.000, 0.000]),
            np.array([2.000, -3.464, 0.000]),
            np.array([3.464, -2.000, 0.000])
        ]
        
        for pos in midBodyPositions:
            direction = np.array([pos[0], pos[1], 0.0])
            direction = direction / np.linalg.norm(direction)
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = np.array(direction, dtype=float)
            thr.MaxThrust = 20000.0
            thr.steadyIsp = vacuumIsp
            self.midbodyEff.addThruster(thr, self.lander.scStateOutMsg)
        
        # Add RCS thrusters
        rcsPositions = [
            np.array([4.200, 0.000, 22.500]),
            np.array([3.637, 2.100, 22.500]),
            np.array([2.100, 3.637, 22.500]),
            np.array([0.000, 4.200, 22.500]),
            np.array([-2.100, 3.637, 22.500]),
            np.array([-3.637, 2.100, 22.500]),
            np.array([-4.200, 0.000, 22.500]),
            np.array([-3.637, -2.100, 22.500]),
            np.array([-2.100, -3.637, 22.500]),
            np.array([0.000, -4.200, 22.500]),
            np.array([2.100, -3.637, 22.500]),
            np.array([3.637, -2.100, 22.500]),
            np.array([4.200, 0.000, -22.500]),
            np.array([3.637, 2.100, -22.500]),
            np.array([2.100, 3.637, -22.500]),
            np.array([0.000, 4.200, -22.500]),
            np.array([-2.100, 3.637, -22.500]),
            np.array([-3.637, 2.100, -22.500]),
            np.array([-4.200, 0.000, -22.500]),
            np.array([-3.637, -2.100, -22.500]),
            np.array([-2.100, -3.637, -22.500]),
            np.array([0.000, -4.200, -22.500]),
            np.array([2.100, -3.637, -22.500]),
            np.array([3.637, -2.100, -22.500])
        ]
        
        for pos in rcsPositions:
            direction = np.array([pos[0], pos[1], 0.0])
            direction = direction / np.linalg.norm(direction)
            thr = thrusterStateEffector.THRSimConfig()
            thr.thrLoc_B = np.array(pos, dtype=float)
            thr.thrDir_B = np.array(direction, dtype=float)
            thr.MaxThrust = 2000.0
            thr.steadyIsp = vacuumIsp
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
        
        # Initialize simulation
        self.scSim.InitializeSimulation()
        
        self.scenario_initialized = True
        
        print(f"✓ Simulation initialized using ScenarioLunarLanderStarter classes")
        print(f"  Terrain: {self.terrain.size}m x {self.terrain.size}m")
        print(f"  LIDAR: {self.lidar.num_rays} rays, {self.lidar.max_range}m range")
        print(f"  Sensors: Ready (observation dim: {self.aiSensors.get_observation_space_size()})")
    
    def _get_observation(self):
        """Get current observation from sensors"""
        obs_dict = self.aiSensors.update()
        
        if self.observation_mode == 'compact':
            # Compact observation: 23D
            obs = np.concatenate([
                obs_dict['position_inertial'][:2],  # x, y (2)
                [obs_dict['altitude_terrain']],  # terrain-relative altitude (1)
                obs_dict['velocity_inertial'],  # vx, vy, vz (3)
                obs_dict['attitude_quaternion'],  # qx, qy, qz, qw (4)
                obs_dict['angular_velocity_body'],  # ωx, ωy, ωz (3)
                [obs_dict['fuel_fraction']],  # fuel remaining (1)
                [obs_dict['lidar_min_range'],  # LIDAR stats (3)
                 obs_dict['lidar_mean_range'],
                 obs_dict['lidar_range_std']],
                obs_dict['imu_accel_current'],  # IMU (6)
                obs_dict['imu_gyro_current']
            ], dtype=np.float32)
        else:
            # Full observation: use flattened sensor suite
            obs = self.aiSensors.get_flattened_observation().astype(np.float32)
        
        return obs
    
    def _compute_reward(self, obs_dict, action, terminated, truncated):
        """
        Compute reward for current state-action pair
        
        Reward components:
        1. Altitude penalty: penalize being too high (fuel waste)
        2. Velocity penalty: penalize high velocities (crash risk)
        3. Attitude penalty: penalize tilted orientation
        4. Fuel efficiency: reward fuel conservation
        5. Landing bonus: large reward for successful landing
        6. Crash penalty: large penalty for hard landing
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
        
        # 1. Shaping reward: guide towards target and safe descent
        # Penalize altitude (want to descend, but not too fast)
        altitude_reward = -0.001 * altitude
        
        # Penalize horizontal distance from target
        distance_penalty = -0.01 * horizontal_distance
        
        # Penalize high velocities (especially vertical)
        velocity_penalty = -0.01 * (abs(vertical_vel) + 0.5 * horizontal_speed)
        
        # Penalize attitude error (want to stay upright)
        attitude_penalty = -0.1 * np.degrees(attitude_error)
        
        # Small reward for fuel conservation
        fuel_reward = 0.001 * fuel_fraction
        
        # Penalize control effort (smooth control)
        if self.action_mode == 'compact':
            control_penalty = -0.001 * np.sum(np.abs(action))
        else:
            control_penalty = -0.001 * np.sum(np.abs(action))
        
        # Sum shaping rewards
        reward = (altitude_reward + distance_penalty + velocity_penalty + 
                 attitude_penalty + fuel_reward + control_penalty)
        
        # 2. Terminal rewards/penalties
        if terminated:
            # Check if successful landing or crash
            if altitude < 2.0 and altitude > -0.5:  # Near surface
                # Successful landing conditions
                if (abs(vertical_vel) < 2.0 and  # Soft touchdown
                    horizontal_speed < 1.0 and  # Low horizontal speed
                    horizontal_distance < 10.0 and  # Near target
                    np.degrees(attitude_error) < 10.0):  # Upright
                    
                    # BONUS: Successful landing!
                    reward += 1000.0
                    
                    # Extra bonus for precision
                    precision_bonus = 100.0 * (1.0 - horizontal_distance / 10.0)
                    reward += precision_bonus
                    
                    # Extra bonus for softness
                    softness_bonus = 100.0 * (1.0 - abs(vertical_vel) / 2.0)
                    reward += softness_bonus
                    
                else:
                    # Crash landing (too hard or wrong attitude)
                    reward -= 500.0
            
            elif altitude < -0.5:
                # Below surface = crash
                reward -= 500.0
        
        # 3. Danger zone penalties (approaching crash conditions)
        if altitude < 50.0:
            if abs(vertical_vel) > 10.0:  # Too fast close to ground
                reward -= 1.0
            if np.degrees(attitude_error) > 30.0:  # Too tilted
                reward -= 2.0
        
        return reward
    
    def _check_termination(self, obs_dict):
        """
        Check if episode should terminate
        
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
        
        # Success: Landed softly
        if altitude < 2.0 and altitude > -0.5:
            if abs(vertical_vel) < 2.0 and horizontal_speed < 1.0:
                terminated = True
                return terminated, truncated
        
        # Failure: Crashed
        if altitude < -0.5:  # Below surface
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
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        
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
        
        # Set spacecraft initial state
        self.lander.hub.r_CN_NInit = np.array([x, y, z])
        self.lander.hub.v_CN_NInit = velocity
        self.lander.hub.sigma_BNInit = attitude_mrp
        self.lander.hub.omega_BN_BInit = omega
        
        # Re-initialize simulation
        self.scSim.InitializeSimulation()
        
        # Reset sensor history
        self.aiSensors.reset()
        
        # Get initial observation
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
        
        # Convert action to thruster commands
        if self.action_mode == 'compact':
            # Compact: [main_throttle, pitch_torque, yaw_torque, roll_torque]
            main_throttle = action[0]
            torque_cmd = action[1:4]
            
            # Set all primary engines to same throttle
            primary_throttles = np.array([main_throttle] * 3)
            
            # Convert torque commands to RCS thruster activations
            # This is a simplified mapping
            rcs_throttles = np.zeros(24)  # We have 24 RCS thrusters
            
            # Pitch control (torque_cmd[0]): use top/bottom thrusters
            if torque_cmd[0] > 0:
                rcs_throttles[0] = abs(torque_cmd[0])
                rcs_throttles[14] = abs(torque_cmd[0])
            else:
                rcs_throttles[6] = abs(torque_cmd[0])
                rcs_throttles[20] = abs(torque_cmd[0])
            
            # Yaw control (torque_cmd[1]): use side thrusters
            if torque_cmd[1] > 0:
                rcs_throttles[3] = abs(torque_cmd[1])
                rcs_throttles[17] = abs(torque_cmd[1])
            else:
                rcs_throttles[9] = abs(torque_cmd[1])
                rcs_throttles[23] = abs(torque_cmd[1])
            
            # Roll control (torque_cmd[2]): use opposing thrusters
            if torque_cmd[2] > 0:
                rcs_throttles[1] = abs(torque_cmd[2])
                rcs_throttles[7] = abs(torque_cmd[2])
            else:
                rcs_throttles[4] = abs(torque_cmd[2])
                rcs_throttles[10] = abs(torque_cmd[2])
            
            # Set mid-body to zero (not used in compact mode)
            midbody_throttles = np.zeros(12)
            
        else:
            # Full mode: direct thruster control
            primary_throttles = action[0:3]
            rcs_throttles = np.zeros(24)
            if len(action) > 3:
                rcs_throttles[:min(len(action)-3, 24)] = action[3:3+min(len(action)-3, 24)]
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
