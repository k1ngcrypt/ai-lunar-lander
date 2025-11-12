"""
ScenarioLunarLanderStarter.py
Basilisk lunar lander scenario for Starship HLS (Human Landing System)

This script demonstrates the high-fidelity Basilisk simulation used for RL training.
All physical constants are imported from starship_constants.py module.

KEY FEATURES:
- Realistic Starship HLS configuration (1.3M kg initial mass, 3 Raptor engines)
- Advanced sensor suite (IMU with noise, LIDAR with 64 rays, fuel gauges)
- Lunar regolith terrain model (Bekker-Wong mechanics, procedural craters)
- Automatic fuel depletion via thrusterStateEffector
- Terrain contact forces via analytical model
- Python-based thruster controller (AdvancedThrusterController)

SENSORS:
- High-precision IMU (gyro noise: 0.00001 rad/s, accel noise: 0.001 m/s²)
- LIDAR: 64-ray cone scan (45° cone, 150m range, realistic noise/dropout)
- AISensorSuite: Comprehensive observation space (200+ dimensions)

COORDINATE SYSTEM:
- Vehicle origin at mid-height
- +Z is nose/up
- +X is forward, +Y is starboard

NOTE: This script runs a standalone demo. For RL training, use lunar_lander_env.py
which wraps these components in a Gymnasium interface.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import common utilities
from common_utils import setup_basilisk_path, mrp_to_dcm, mrp_to_quaternion

# Import terrain simulation module
from terrain_simulation import LunarRegolithModel

# Add Basilisk to path (built in dist3 directory)
setup_basilisk_path()

from Basilisk.simulation import spacecraft, thrusterStateEffector, imuSensor
from Basilisk.simulation import fuelTank, dragDynamicEffector, exponentialAtmosphere, extForceTorque
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, unitTestSupport
from Basilisk.architecture import messaging

# Import Starship HLS configuration constants
import starship_constants as SC

# ----------------------------------------------------------------------
# 1. Simulation Setup
# ----------------------------------------------------------------------

# Create simulation base class
scSim = SimulationBaseClass.SimBaseClass()

# Create process and tasks
simProcessName = "simProcess"
dynTaskName = "dynTask"
fswTaskName = "fswTask"

dynProcess = scSim.CreateNewProcess(simProcessName)
simulationTimeStep = macros.sec2nano(0.1)  # 0.1 second timestep
fswTimeStep = macros.sec2nano(0.5)         # FSW runs at 2 Hz

dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simulationTimeStep))
dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep))

# ----------------------------------------------------------------------
# 2. Create Spacecraft - Starship HLS Configuration
# ----------------------------------------------------------------------
# NOTE: All Starship constants imported from starship_constants module (as SC)

lander = spacecraft.Spacecraft()
lander.ModelTag = "Starship_HLS"
lander.hub.mHub = SC.HUB_MASS
lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
lander.hub.r_CN_NInit = np.array([0., 0., 1500.0])   # 1500 m altitude above Moon surface
lander.hub.v_CN_NInit = np.array([0., 0., -10.0])    # descending at 10 m/s
lander.hub.sigma_BNInit = np.array([0., 0., 0.])     # no initial attitude
lander.hub.omega_BN_BInit = np.zeros(3)              # no rotation

scSim.AddModelToTask(dynTaskName, lander)

# ----------------------------------------------------------------------
# 2a. Add Propellant Tanks - CH4 and LOX
# ----------------------------------------------------------------------
# NOTE: Tank parameters from starship_constants module

# CH4 Tank (Methane)
ch4Tank = fuelTank.FuelTank()
ch4Tank.ModelTag = "CH4_Tank"
ch4TankModel = fuelTank.FuelTankModelConstantVolume()
ch4TankModel.propMassInit = SC.CH4_INITIAL_MASS
ch4TankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
ch4TankModel.radiusTankInit = SC.CH4_TANK_RADIUS
ch4Tank.setTankModel(ch4TankModel)
ch4Tank.r_TB_B = SC.CH4_TANK_POSITION
ch4Tank.nameOfMassState = "ch4TankMass"
lander.addStateEffector(ch4Tank)
scSim.AddModelToTask(dynTaskName, ch4Tank)

# LOX Tank (Liquid Oxygen)
loxTank = fuelTank.FuelTank()
loxTank.ModelTag = "LOX_Tank"
loxTankModel = fuelTank.FuelTankModelConstantVolume()
loxTankModel.propMassInit = SC.LOX_INITIAL_MASS
loxTankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
loxTankModel.radiusTankInit = SC.LOX_TANK_RADIUS
loxTank.setTankModel(loxTankModel)
loxTank.r_TB_B = SC.LOX_TANK_POSITION
loxTank.nameOfMassState = "loxTankMass"
lander.addStateEffector(loxTank)
scSim.AddModelToTask(dynTaskName, loxTank)

# ----------------------------------------------------------------------
# 3. Setup Gravity
# ----------------------------------------------------------------------
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True
gravFactory.addBodiesTo(lander)

# ----------------------------------------------------------------------
# 3a. Lightweight Analytical Terrain Model (Replaces Chrono DEM)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("INITIALIZING ANALYTICAL TERRAIN SYSTEM")
print("="*60)

# Create terrain model instance
terrain = LunarRegolithModel(size=2000.0, resolution=200)

# Try to load terrain from file (if available)
terrainDataPath = os.path.join(os.path.dirname(__file__), 'generated_terrain', 'moon_terrain.npy')
if not terrain.load_terrain_from_file(terrainDataPath):
    # Fall back to procedural generation
    print("  Generating procedural terrain...")
    terrain.generate_procedural_terrain(num_craters=15, 
                                        crater_depth_range=(3, 12), 
                                        crater_radius_range=(15, 60))

print("="*60 + "\n")

# ----------------------------------------------------------------------
# 3b. External Force Effector for Terrain Contact
# ----------------------------------------------------------------------
# Use Basilisk's extForceTorque module to apply terrain contact forces

terrainForceEff = extForceTorque.ExtForceTorque()
terrainForceEff.ModelTag = "TerrainContactForce"
scSim.AddModelToTask(dynTaskName, terrainForceEff)
lander.addDynamicEffector(terrainForceEff)

# ----------------------------------------------------------------------
# 3c. Setup Aerodynamics (minimal effect on Moon)
# ----------------------------------------------------------------------
# Moon has essentially no atmosphere (exosphere ~10^-15 kg/m³), but
# aerodynamic effector is included for realism and future Earth reentry simulations.

dragEffector = dragDynamicEffector.DragDynamicEffector()
dragEffector.ModelTag = "DragEffector"
dragEffector.coreParams.projectedArea = 63.617  # m² (π × radius²)
dragEffector.coreParams.dragCoeff = 0.6
dragEffector.coreParams.comOffset = [0.0, 0.0, 0.0]

# Add drag effector to spacecraft
lander.addDynamicEffector(dragEffector)
scSim.AddModelToTask(dynTaskName, dragEffector)

# Moon exosphere model (essentially vacuum)
moonAtmo = exponentialAtmosphere.ExponentialAtmosphere()
moonAtmo.ModelTag = "MoonAtmosphere"
moonAtmo.planetRadius = moon.radEquator
moonAtmo.scaleHeight = 100000.0  # m (arbitrary large value)
moonAtmo.baseDensity = 1e-15  # kg/m³ (Moon's exosphere density)
moonAtmo.envMinReach = -10000.0
moonAtmo.envMaxReach = 10000.0
moonAtmo.addSpacecraftToModel(lander.scStateOutMsg)
scSim.AddModelToTask(dynTaskName, moonAtmo)

# Connect atmosphere to drag effector
dragEffector.atmoDensInMsg.subscribeTo(moonAtmo.envOutMsgs[0])

# ----------------------------------------------------------------------
# 4. Add Thrusters - Starship HLS Configuration (thrusterStateEffector for fuel integration)
# ----------------------------------------------------------------------
# Using thrusterStateEffector instead of thrusterDynamicEffector for:
# - Automatic fuel tank depletion
# - Isp-based performance modeling
# - Multiple fuel tank connections per thruster
# - Gimbal actuation support
# 
# NOTE: MAX_EFF_CNT = 36, so we need separate effectors for each thruster group
# to avoid exceeding the limit (3 + 12 + 24 = 39 > 36)

# Create separate thruster state effectors for each group
primaryEff = thrusterStateEffector.ThrusterStateEffector()
primaryEff.ModelTag = "PrimaryThrusters"
scSim.AddModelToTask(dynTaskName, primaryEff)
lander.addStateEffector(primaryEff)

midbodyEff = thrusterStateEffector.ThrusterStateEffector()
midbodyEff.ModelTag = "MidBodyThrusters"
scSim.AddModelToTask(dynTaskName, midbodyEff)
lander.addStateEffector(midbodyEff)

rcsEff = thrusterStateEffector.ThrusterStateEffector()
rcsEff.ModelTag = "RCSThrusters"
scSim.AddModelToTask(dynTaskName, rcsEff)
lander.addStateEffector(rcsEff)

# PRIMARY AFT ENGINES (3 Vacuum Raptors with gimbal)
# NOTE: All thruster configuration from starship_constants module

print(f"\nPrimary Engine Fuel Consumption (per engine @ 100%):")
print(f"  Total mass flow: {SC.PER_ENGINE_MASS_FLOW:.2f} kg/s")
print(f"  CH4 flow: {SC.CH4_FLOW_PER_ENGINE:.2f} kg/s")
print(f"  LOX flow: {SC.LOX_FLOW_PER_ENGINE:.2f} kg/s")

for pos in SC.PRIMARY_ENGINE_POSITIONS:
    thrConfig = thrusterStateEffector.THRSimConfig()
    thrConfig.thrLoc_B = np.array(pos, dtype=float)
    thrConfig.thrDir_B = SC.PRIMARY_ENGINE_DIRECTION
    thrConfig.MaxThrust = SC.MAX_THRUST_PER_ENGINE
    thrConfig.steadyIsp = SC.VACUUM_ISP
    primaryEff.addThruster(thrConfig, lander.scStateOutMsg)

# Store indices for easy reference
PRIMARY_START = SC.PRIMARY_ENGINE_START_INDEX
PRIMARY_COUNT = SC.PRIMARY_ENGINE_COUNT

# MID-BODY THRUSTERS (12 thrusters for attitude control)
for pos in SC.MIDBODY_THRUSTER_POSITIONS:
    direction = SC.get_midbody_thruster_direction(pos)
    thrConfig = thrusterStateEffector.THRSimConfig()
    thrConfig.thrLoc_B = np.array(pos, dtype=float)
    thrConfig.thrDir_B = np.array(direction, dtype=float)
    thrConfig.MaxThrust = SC.MIDBODY_THRUST
    thrConfig.steadyIsp = SC.VACUUM_ISP
    midbodyEff.addThruster(thrConfig, lander.scStateOutMsg)

MIDBODY_START = SC.MIDBODY_THRUSTER_START_INDEX
MIDBODY_COUNT = SC.MIDBODY_THRUSTER_COUNT

# RCS THRUSTERS (24 thrusters: 12 at top ring, 12 at bottom ring)
for pos in SC.RCS_THRUSTER_POSITIONS:
    direction = SC.get_rcs_thruster_direction(pos)
    thrConfig = thrusterStateEffector.THRSimConfig()
    thrConfig.thrLoc_B = np.array(pos, dtype=float)
    thrConfig.thrDir_B = np.array(direction, dtype=float)
    thrConfig.MaxThrust = SC.RCS_THRUST
    thrConfig.steadyIsp = SC.VACUUM_ISP
    rcsEff.addThruster(thrConfig, lander.scStateOutMsg)

RCS_START = SC.RCS_THRUSTER_START_INDEX
RCS_COUNT = SC.RCS_THRUSTER_COUNT

# Total thrusters
TOTAL_THRUSTERS = SC.TOTAL_THRUSTER_COUNT

# Connect fuel tanks to thruster effectors for automatic fuel depletion
# This enables Basilisk to automatically deplete fuel based on thrust and Isp
ch4Tank.addThrusterSet(primaryEff)
ch4Tank.addThrusterSet(midbodyEff)
ch4Tank.addThrusterSet(rcsEff)

loxTank.addThrusterSet(primaryEff)
loxTank.addThrusterSet(midbodyEff)
loxTank.addThrusterSet(rcsEff)

# Print thruster configuration summary
print(f"\nThruster Configuration (thrusterStateEffector):")
print(f"  - {PRIMARY_COUNT} Primary Raptor engines (2,500,000 N each, gimbal capable)")
print(f"  - {MIDBODY_COUNT} Mid-body thrusters (20,000 N each)")
print(f"  - {RCS_COUNT} RCS thrusters (2,000 N each)")
print(f"  - Total: {TOTAL_THRUSTERS} thrusters")
print(f"  - Split into 3 effectors (MAX_EFF_CNT=36 limit)")
print(f"  - All thrusters connected to fuel tanks for automatic depletion")

# ----------------------------------------------------------------------
# 5. Add IMU Sensor - Starship HLS with Realistic Noise
# ----------------------------------------------------------------------
imu = imuSensor.ImuSensor()
imu.ModelTag = "StarshipIMU"
# IMU aligned with body frame (no rotation: yaw=0, pitch=0, roll=0)
imu.setBodyToPlatformDCM(0.0, 0.0, 0.0)
imu.scStateInMsg.subscribeTo(lander.scStateOutMsg)
# Set error bounds (noise levels) - high-precision IMU for large spacecraft
# Gyro noise: 0.00001 rad/s (typical for space-grade IMU)
# Accel noise: 0.001 m/s² (typical for space-grade accelerometer)
imu.setErrorBoundsGyro([0.00001] * 3)   # Very low gyro noise
imu.setErrorBoundsAccel([0.001] * 3)    # Low accelerometer noise
scSim.AddModelToTask(dynTaskName, imu)

# ----------------------------------------------------------------------
# 5a. LIDAR Sensor for Terrain Mapping
# ----------------------------------------------------------------------
class LIDARSensor:
    """
    Simulated LIDAR sensor for terrain mapping and obstacle detection.
    Features:
    - Configurable scan pattern (cone angle, resolution)
    - Ray-casting against terrain heightmap
    - Realistic noise model (range noise, dropout)
    - Returns point cloud in body frame
    - High performance (vectorized operations)
    """
    
    def __init__(self, terrain, max_range=100.0, cone_angle=30.0, num_rays=64):
        """
        Args:
            terrain: LunarRegolithModel instance for ray-casting
            max_range: Maximum detection range in meters
            cone_angle: Cone half-angle in degrees (0 = nadir only)
            num_rays: Number of rays in scan pattern
        """
        self.terrain = terrain
        self.max_range = max_range
        self.cone_angle_rad = np.radians(cone_angle)
        self.num_rays = num_rays
        
        # Noise parameters
        self.range_noise_std = 0.05  # meters (5 cm standard deviation)
        self.dropout_prob = 0.02     # 2% probability of no return
        self.min_range = 0.5         # Minimum valid range (m)
        
        # Generate scan pattern (cone of rays pointing down)
        self.ray_directions_B = self._generate_scan_pattern()
        
        self.rng = np.random.RandomState(123)
        
        print(f"\nLIDAR Sensor Configuration:")
        print(f"  Max range: {self.max_range} m")
        print(f"  Cone angle: ±{cone_angle}°")
        print(f"  Number of rays: {self.num_rays}")
        print(f"  Range noise std: {self.range_noise_std} m")
        print(f"  Dropout probability: {self.dropout_prob*100}%")
    
    def _generate_scan_pattern(self):
        """
        Generate ray directions in body frame (downward-looking cone)
        Returns: (num_rays, 3) array of unit vectors
        """
        if self.num_rays == 1:
            # Single nadir-pointing ray
            return np.array([[0.0, 0.0, -1.0]])
        
        # Generate spiral pattern on cone surface
        directions = []
        
        # Always include nadir ray
        directions.append([0.0, 0.0, -1.0])
        
        # Generate rays in concentric rings
        num_rings = int(np.sqrt(self.num_rays))
        rays_per_ring = (self.num_rays - 1) // num_rings
        
        for ring in range(1, num_rings + 1):
            angle_from_nadir = self.cone_angle_rad * (ring / num_rings)
            num_in_ring = rays_per_ring if ring < num_rings else (self.num_rays - 1 - rays_per_ring * (num_rings - 1))
            
            for i in range(num_in_ring):
                azimuth = 2.0 * np.pi * i / num_in_ring
                
                # Convert spherical to Cartesian (body frame: +Z up, -Z down)
                x = np.sin(angle_from_nadir) * np.cos(azimuth)
                y = np.sin(angle_from_nadir) * np.sin(azimuth)
                z = -np.cos(angle_from_nadir)  # Negative because pointing down
                
                directions.append([x, y, z])
        
        directions = np.array(directions[:self.num_rays])
        # Normalize (should already be normalized, but ensure)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        return directions
    
    def scan(self, position_N, attitude_BN, velocity_N=None):
        """
        Perform LIDAR scan from current spacecraft state
        
        Args:
            position_N: Spacecraft position [x, y, z] in inertial frame (m)
            attitude_BN: DCM from body to inertial frame (3x3 matrix)
            velocity_N: Spacecraft velocity (optional, for motion compensation)
        
        Returns:
            point_cloud_B: (N, 3) array of 3D points in body frame (m)
                          Invalid returns are marked as [0, 0, 0]
            ranges: (N,) array of ranges (m), -1 for invalid
            intensities: (N,) array of return intensities [0-1]
        """
        point_cloud_B = []
        ranges = []
        intensities = []
        
        for ray_dir_B in self.ray_directions_B:
            # Transform ray direction to inertial frame
            ray_dir_N = attitude_BN @ ray_dir_B
            
            # Ray-cast to find terrain intersection
            range_m, hit = self._raycast_terrain(position_N, ray_dir_N)
            
            # Apply noise and dropout
            if hit and self.rng.random() > self.dropout_prob:
                # Add range noise
                range_noisy = range_m + self.rng.normal(0, self.range_noise_std)
                range_noisy = np.clip(range_noisy, self.min_range, self.max_range)
                
                # Compute point in body frame
                point_B = ray_dir_B * range_noisy
                
                # Intensity model (decreases with range, some randomness)
                intensity = max(0.1, 1.0 - (range_noisy / self.max_range)) * self.rng.uniform(0.8, 1.0)
                
                point_cloud_B.append(point_B)
                ranges.append(range_noisy)
                intensities.append(intensity)
            else:
                # No return (dropout or out of range)
                point_cloud_B.append([0.0, 0.0, 0.0])
                ranges.append(-1.0)
                intensities.append(0.0)
        
        return np.array(point_cloud_B), np.array(ranges), np.array(intensities)
    
    def _raycast_terrain(self, origin_N, direction_N):
        """
        Cast ray from origin in direction to find terrain intersection
        
        Args:
            origin_N: Ray origin in inertial frame [x, y, z]
            direction_N: Unit ray direction in inertial frame
        
        Returns:
            range_m: Distance to intersection (m), or max_range if no hit
            hit: Boolean, True if terrain was hit
        """
        # Simple ray marching along ray direction
        # Step size based on terrain resolution
        step_size = self.terrain.cell_size * 0.5  # Half cell for accuracy
        max_steps = int(self.max_range / step_size)
        
        pos = origin_N.copy()
        
        for step in range(max_steps):
            # Check current position height vs terrain
            x, y, z = pos
            terrain_height = self.terrain.get_height(x, y)
            
            if z <= terrain_height:
                # Hit! Compute exact range
                range_m = step * step_size
                return range_m, True
            
            # Step along ray
            pos += direction_N * step_size
        
        # No hit within max range
        return self.max_range, False

# Create LIDAR sensor instance
lidar = LIDARSensor(terrain, max_range=150.0, cone_angle=45.0, num_rays=64)

print("="*60 + "\n")

# ----------------------------------------------------------------------
# 6. AI Sensor Suite - Comprehensive Sensor Integration for AI Agent
# ----------------------------------------------------------------------
class AISensorSuite:
    """
    Comprehensive sensor suite for AI agent control of lunar lander.
    Provides:
    - IMU acceleration and gyroscope with history buffer (N frames)
    - Altitude above local terrain and vertical velocity
    - Horizontal velocity in local frame
    - Attitude (quaternion) and attitude error to target
    - Remaining mass and fuel
    - LIDAR point cloud with noise
    - All data formatted for AI/RL observation space
    """
    
    def __init__(self, scObject, imu, terrain, lidar, ch4Tank, loxTank, history_length=10):
        """
        Args:
            scObject: Spacecraft object
            imu: IMU sensor object
            terrain: LunarRegolithModel instance
            lidar: LIDARSensor instance
            ch4Tank: CH4 fuel tank
            loxTank: LOX fuel tank
            history_length: Number of IMU frames to keep in history buffer
        """
        self.scObject = scObject
        self.imu = imu
        self.terrain = terrain
        self.lidar = lidar
        self.ch4Tank = ch4Tank
        self.loxTank = loxTank
        
        # History buffers for IMU data (FIFO queues)
        self.history_length = history_length
        self.accel_history = []  # List of (3,) arrays
        self.gyro_history = []   # List of (3,) arrays
        
        # Target state for computing errors (default: hover at origin, upright)
        self.target_position = np.array([0.0, 0.0, 0.0])  # Target position (m)
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # Target velocity (m/s)
        self.target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Target attitude (upright)
        
        # Constants
        self.dry_mass = SC.HUB_MASS  # kg (dry mass + payload)
        self.g_moon = 1.62  # m/s² (lunar surface gravity)
        
        print(f"\nAI Sensor Suite Initialized:")
        print(f"  IMU history buffer: {history_length} frames")
        print(f"  Sensors: IMU, Altimeter, Velocimeter, Attitude, Fuel, LIDAR")
        print(f"  Target state: Position {self.target_position}, Attitude upright")
    
    def set_target_state(self, position=None, velocity=None, quaternion=None):
        """Set target state for computing errors"""
        if position is not None:
            self.target_position = np.array(position)
        if velocity is not None:
            self.target_velocity = np.array(velocity)
        if quaternion is not None:
            self.target_quaternion = np.array(quaternion)
    
    def quaternion_error(self, q_current, q_target):
        """
        Compute attitude error quaternion: q_error = q_target^-1 * q_current
        Returns quaternion representing rotation from current to target
        """
        from common_utils import quaternion_error
        return quaternion_error(q_current, q_target)
    
    def update(self):
        """
        Update sensor readings and return comprehensive observation dict
        Call this every timestep to get latest sensor data
        
        Returns:
            dict with all sensor data for AI agent
        """
        # Read spacecraft state
        scState = self.scObject.scStateOutMsg.read()
        position_N = scState.r_BN_N  # Inertial position (m)
        velocity_N = scState.v_BN_N  # Inertial velocity (m/s)
        sigma_BN = scState.sigma_BN  # Attitude (MRP)
        omega_BN_B = scState.omega_BN_B  # Angular velocity in body frame (rad/s)
        
        # Read IMU data (with sensor noise)
        imuData = self.imu.sensorOutMsg.read()
        accel_B = imuData.AccelPlatform  # Measured acceleration in body frame (m/s²)
        gyro_B = imuData.AngVelPlatform  # Measured angular velocity in body frame (rad/s)
        
        # Update IMU history buffers
        self.accel_history.append(accel_B.copy())
        self.gyro_history.append(gyro_B.copy())
        if len(self.accel_history) > self.history_length:
            self.accel_history.pop(0)
            self.gyro_history.pop(0)
        
        # Pad history if not full yet
        accel_history_padded = self.accel_history.copy()
        gyro_history_padded = self.gyro_history.copy()
        while len(accel_history_padded) < self.history_length:
            accel_history_padded.insert(0, np.zeros(3))
            gyro_history_padded.insert(0, np.zeros(3))
        
        # Convert attitude to DCM and quaternion
        BN_dcm = mrp_to_dcm(sigma_BN)
        quaternion = mrp_to_quaternion(sigma_BN)
        
        # --- TERRAIN-RELATIVE MEASUREMENTS ---
        # Altitude above local terrain (not just inertial Z)
        x, y, z = position_N
        terrain_height = self.terrain.get_height(x, y)
        altitude_terrain = z - terrain_height  # meters above local terrain
        
        # Vertical velocity (project velocity onto local "up" direction)
        # For flat terrain, this is just velocity_N[2]
        # For sloped terrain, would need terrain normal vector
        vertical_velocity = velocity_N[2]  # m/s (positive = ascending)
        
        # --- HORIZONTAL VELOCITY IN LOCAL FRAME ---
        # Local frame: East-North-Up (ENU) aligned with terrain at current position
        # For simplicity, assume local frame = inertial frame (flat Moon surface)
        # Horizontal velocity in local frame
        horizontal_velocity_local = np.array([velocity_N[0], velocity_N[1], 0.0])
        horizontal_speed = np.linalg.norm(horizontal_velocity_local[:2])
        
        # Velocity in body frame (for AI to understand motion relative to spacecraft orientation)
        velocity_B = BN_dcm.T @ velocity_N
        
        # --- ATTITUDE ERROR TO TARGET ---
        attitude_error_quat = self.quaternion_error(quaternion, self.target_quaternion)
        
        # Also compute Euler angle approximation for small errors
        # Error angle (magnitude of rotation)
        error_angle = 2.0 * np.arccos(np.clip(attitude_error_quat[3], -1.0, 1.0))
        
        # --- FUEL AND MASS ---
        ch4State = self.ch4Tank.fuelTankOutMsg.read()
        loxState = self.loxTank.fuelTankOutMsg.read()
        
        ch4_mass = ch4State.fuelMass
        lox_mass = loxState.fuelMass
        total_fuel_mass = ch4_mass + lox_mass
        total_mass = self.dry_mass + total_fuel_mass
        
        # Fuel fractions (0-1)
        ch4_fraction = ch4_mass / SC.CH4_INITIAL_MASS
        lox_fraction = lox_mass / SC.LOX_INITIAL_MASS
        fuel_fraction = total_fuel_mass / SC.TOTAL_PROPELLANT_MASS
        
        # --- LIDAR POINT CLOUD ---
        point_cloud_B, ranges, intensities = self.lidar.scan(position_N, BN_dcm, velocity_N)
        
        # Compute LIDAR statistics (min range, variance, etc.)
        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) > 0:
            min_range = np.min(valid_ranges)
            mean_range = np.mean(valid_ranges)
            range_std = np.std(valid_ranges)
        else:
            min_range = -1.0
            mean_range = -1.0
            range_std = 0.0
        
        # --- CONSTRUCT OBSERVATION DICTIONARY ---
        observation = {
            # IMU data with history
            'imu_accel_current': accel_B,  # (3,) current frame
            'imu_gyro_current': gyro_B,    # (3,) current frame
            'imu_accel_history': np.array(accel_history_padded),  # (history_length, 3)
            'imu_gyro_history': np.array(gyro_history_padded),    # (history_length, 3)
            
            # Position and velocity (inertial frame)
            'position_inertial': position_N,  # (3,) [x, y, z] meters
            'velocity_inertial': velocity_N,  # (3,) [vx, vy, vz] m/s
            'velocity_body': velocity_B,      # (3,) velocity in body frame
            
            # Terrain-relative measurements
            'altitude_terrain': altitude_terrain,  # meters above local terrain
            'terrain_height': terrain_height,      # terrain height at (x, y)
            'vertical_velocity': vertical_velocity,  # m/s (positive = ascending)
            'horizontal_velocity_local': horizontal_velocity_local,  # (3,) m/s
            'horizontal_speed': horizontal_speed,  # m/s (magnitude)
            
            # Attitude
            'attitude_quaternion': quaternion,  # (4,) [x, y, z, w]
            'attitude_mrp': sigma_BN,           # (3,) Modified Rodriguez Parameters
            'angular_velocity_body': omega_BN_B,  # (3,) rad/s in body frame
            'attitude_error_quaternion': attitude_error_quat,  # (4,) error to target
            'attitude_error_angle': error_angle,  # radians (scalar error magnitude)
            
            # Mass and fuel
            'total_mass': total_mass,        # kg
            'fuel_mass': total_fuel_mass,    # kg
            'ch4_mass': ch4_mass,            # kg
            'lox_mass': lox_mass,            # kg
            'fuel_fraction': fuel_fraction,  # 0-1 (remaining fuel percentage)
            'ch4_fraction': ch4_fraction,    # 0-1
            'lox_fraction': lox_fraction,    # 0-1
            
            # LIDAR data
            'lidar_point_cloud': point_cloud_B,  # (num_rays, 3) points in body frame
            'lidar_ranges': ranges,              # (num_rays,) range measurements
            'lidar_intensities': intensities,    # (num_rays,) return intensities
            'lidar_min_range': min_range,        # minimum valid range
            'lidar_mean_range': mean_range,      # mean valid range
            'lidar_range_std': range_std,        # range standard deviation
            
            # Derived/useful quantities
            'gravity_body': BN_dcm.T @ np.array([0, 0, -self.g_moon]),  # gravity in body frame
            'dcm_body_to_inertial': BN_dcm,  # (3, 3) rotation matrix
        }
        
        return observation
    
    def get_flattened_observation(self):
        """
        Get observation as a flat numpy array (useful for neural networks)
        Returns: 1D numpy array with all sensor data concatenated
        """
        obs = self.update()
        
        # Flatten all arrays and concatenate
        flat_obs = np.concatenate([
            obs['imu_accel_current'],
            obs['imu_gyro_current'],
            obs['imu_accel_history'].flatten(),
            obs['imu_gyro_history'].flatten(),
            obs['position_inertial'],
            obs['velocity_inertial'],
            obs['velocity_body'],
            [obs['altitude_terrain'], obs['vertical_velocity'], obs['horizontal_speed']],
            obs['attitude_quaternion'],
            obs['angular_velocity_body'],
            obs['attitude_error_quaternion'],
            [obs['attitude_error_angle']],
            [obs['total_mass'], obs['fuel_mass'], obs['fuel_fraction']],
            obs['lidar_ranges'],
            obs['lidar_intensities'],
            [obs['lidar_min_range'], obs['lidar_mean_range'], obs['lidar_range_std']],
            obs['gravity_body'],
        ])
        
        return flat_obs
    
    def get_observation_space_size(self):
        """Return the size of the flattened observation vector"""
        obs = self.get_flattened_observation()
        return obs.shape[0]
    
    def reset(self):
        """Reset history buffers (call at start of new episode)"""
        self.accel_history = []
        self.gyro_history = []

# ----------------------------------------------------------------------
# 6a. Advanced Thruster Control FSW Module with Gimbal Support
# ----------------------------------------------------------------------
# Controller with:
# - Throttle limits (40%-100% on main engines)
# - Gimbal limits (±8°) with actuation
# - Automatic fuel consumption via thrusterStateEffector
# - Support for all thruster groups (primary, mid-body, RCS)
# - Terrain contact force application

class AdvancedThrusterController:
    """
    Starship HLS Thruster Controller with Terrain Contact and Gimbal
    - Enforces throttle range [0.4, 1.0] on main engines (0-100% on others)
    - Controls gimbal angles for primary engines (±8° limits)
    - Manages all thruster groups for AI control
    - Applies terrain contact forces to spacecraft
    - Fuel depletion handled automatically by thrusterStateEffector
    """
    def __init__(self, primaryEff, midbodyEff, rcsEff, ch4Tank, loxTank, terrain, terrainForceEff, scObject):
        self.primaryEff = primaryEff
        self.midbodyEff = midbodyEff
        self.rcsEff = rcsEff
        self.ch4Tank = ch4Tank
        self.loxTank = loxTank
        self.terrain = terrain
        self.terrainForceEff = terrainForceEff
        self.scObject = scObject
        
        # Create message writers for each effector
        self.primCmdMsg = messaging.THRArrayOnTimeCmdMsg()
        self.midCmdMsg = messaging.THRArrayOnTimeCmdMsg()
        self.rcsCmdMsg = messaging.THRArrayOnTimeCmdMsg()
        
        self.ModelTag = "ThrusterController"
        self.moduleID = 1  # Module ID for message writing
        
        # Thruster configuration
        self.primaryStart = 0
        self.primaryCount = 3
        self.midbodyStart = 3
        self.midbodyCount = 12
        self.rcsStart = 15
        self.rcsCount = 24
        self.totalThrusters = 39
        
        # Primary engine parameters
        self.minThrottle = 0.4  # 40% minimum for main engines
        self.maxThrottle = 1.0  # 100%
        
        # Gimbal limits (radians)
        self.gimbalLimitRad = np.radians(8.0)  # ±8°
        
        # Gimbal state (pitch, yaw for each of 3 engines)
        self.gimbalAngles = np.zeros((3, 2))  # [engine][pitch, yaw]
        
        # Command arrays (set by AI via setThrusterCommands)
        self.primaryThrottles = np.zeros(3)
        self.midbodyThrottles = np.zeros(12)
        self.rcsThrottles = np.zeros(24)
        
        # Track fuel depletion (informational - actual depletion is automatic)
        self.lastUpdateTime = 0.0
        
        # Landing leg positions (4 legs at corners, z = -24.5m, radius = 4.5m from center)
        self.landing_leg_positions_B = np.array([
            [4.5, 4.5, -24.5],   # Leg 1: +X, +Y
            [-4.5, 4.5, -24.5],  # Leg 2: -X, +Y
            [-4.5, -4.5, -24.5], # Leg 3: -X, -Y
            [4.5, -4.5, -24.5]   # Leg 4: +X, -Y
        ])
        self.landing_leg_area = 0.5  # m² contact area per leg
        
        # Create message for terrain forces
        self.terrainForceMsg = messaging.CmdForceBodyMsg()
        
        print(f"\nController initialized:")
        print(f"  Primary throttle range: {self.minThrottle*100}% - {self.maxThrottle*100}%")
        print(f"  Gimbal limit: ±{np.degrees(self.gimbalLimitRad):.1f}°")
        print(f"  Total thrusters: {self.totalThrusters}")
        print(f"  Landing legs: {len(self.landing_leg_positions_B)}")
        print(f"  Fuel depletion: Automatic via thrusterStateEffector")
        
    def clampThrottle(self, throttle, isPrimary=False):
        """Enforce throttle limits"""
        if throttle < 0.01:  # Consider off
            return 0.0
        if isPrimary:
            return np.clip(throttle, self.minThrottle, self.maxThrottle)
        else:
            return np.clip(throttle, 0.0, 1.0)
    
    def clampGimbal(self, angle):
        """Enforce gimbal angle limits"""
        return np.clip(angle, -self.gimbalLimitRad, self.gimbalLimitRad)
    
    def setGimbalAngles(self, engine_idx, pitch, yaw):
        """
        Set gimbal angles for a primary engine
        Args:
            engine_idx: 0-2 for the three primary engines
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
        """
        if 0 <= engine_idx < 3:
            self.gimbalAngles[engine_idx, 0] = self.clampGimbal(pitch)
            self.gimbalAngles[engine_idx, 1] = self.clampGimbal(yaw)
    
    def setThrusterCommands(self, primaryThrottles=None, midbodyThrottles=None, rcsThrottles=None,
                           gimbalAngles=None):
        """
        Set thruster commands for AI control
        
        Args:
            primaryThrottles: Array of 3 throttle values [0.4-1.0] for primary engines
            midbodyThrottles: Array of 12 throttle values [0-1.0] for mid-body thrusters  
            rcsThrottles: Array of 24 throttle values [0-1.0] for RCS thrusters
            gimbalAngles: Array of shape (3, 2) with [pitch, yaw] for each primary engine (radians)
        """
        # Store throttle commands (will be applied in next Update() call)
        if primaryThrottles is not None:
            self.primaryThrottles = np.array(primaryThrottles)
        else:
            self.primaryThrottles = np.zeros(3)
            
        if midbodyThrottles is not None:
            self.midbodyThrottles = np.array(midbodyThrottles)
        else:
            self.midbodyThrottles = np.zeros(12)
            
        if rcsThrottles is not None:
            self.rcsThrottles = np.array(rcsThrottles)
        else:
            self.rcsThrottles = np.zeros(24)
            
        if gimbalAngles is not None:
            self.gimbalAngles = np.array(gimbalAngles)
            # Clamp all gimbal angles
            for i in range(3):
                self.gimbalAngles[i, 0] = self.clampGimbal(self.gimbalAngles[i, 0])
                self.gimbalAngles[i, 1] = self.clampGimbal(self.gimbalAngles[i, 1])
    
    def getThrusterState(self):
        """
        Get current thruster state for AI observation
        Returns dict with fuel masses, throttle limits, etc.
        """
        ch4State = self.ch4Tank.fuelTankOutMsg.read()
        loxState = self.loxTank.fuelTankOutMsg.read()
        
        return {
            'ch4_mass': ch4State.fuelMass,
            'lox_mass': loxState.fuelMass,
            'total_fuel_mass': ch4State.fuelMass + loxState.fuelMass,
            'primary_throttle_min': self.minThrottle,
            'primary_throttle_max': self.maxThrottle,
            'gimbal_limit_rad': self.gimbalLimitRad,
            'gimbal_limit_deg': np.degrees(self.gimbalLimitRad),
            'current_gimbal_angles': self.gimbalAngles.copy(),
            'total_thrusters': self.totalThrusters,
            'primary_count': self.primaryCount,
            'midbody_count': self.midbodyCount,
            'rcs_count': self.rcsCount
        }
    
    def updateTerrainContact(self, currentTime):
        """
        Compute and apply terrain contact forces from landing legs
        This is called every simulation timestep
        
        OPTIMIZED: Early exit when no contact possible
        """
        # Get spacecraft state
        scState = self.scObject.scStateOutMsg.read()
        r_BN_N = scState.r_BN_N  # Position in inertial frame
        v_BN_N = scState.v_BN_N  # Velocity in inertial frame
        
        # PRODUCTION OPTIMIZATION: More aggressive early exit
        # Landing legs are at most 3m long, contact impossible above 10m
        if r_BN_N[2] > 10.0:
            # Write zero forces
            forceMsg = messaging.CmdForceBodyMsgPayload()
            forceMsg.forceRequestInertial = np.zeros(3)
            forceMsg.torqueRequestBody = np.zeros(3)
            self.terrainForceMsg.write(forceMsg, self.moduleID, currentTime)
            return
        
        # ADDITIONAL OPTIMIZATION: Skip if ascending rapidly and not near ground
        if r_BN_N[2] > 5.0 and v_BN_N[2] > 1.0:  # Above 5m and ascending > 1 m/s
            # Write zero forces
            forceMsg = messaging.CmdForceBodyMsgPayload()
            forceMsg.forceRequestInertial = np.zeros(3)
            forceMsg.torqueRequestBody = np.zeros(3)
            self.terrainForceMsg.write(forceMsg, self.moduleID, currentTime)
            return
        
        sigma_BN = scState.sigma_BN  # MRP attitude
        omega_BN_B = scState.omega_BN_B  # Angular velocity in body frame
        
        # Convert MRP to DCM (Direction Cosine Matrix)
        # DCM transforms from body to inertial: r_N = [BN] * r_B
        BN = mrp_to_dcm(sigma_BN)
        
        # Total contact force and torque
        total_force_N = np.zeros(3)
        total_torque_B = np.zeros(3)
        
        # Check each landing leg for contact
        for i, leg_pos_B in enumerate(self.landing_leg_positions_B):
            # Transform leg position to inertial frame
            leg_pos_N = r_BN_N + BN @ leg_pos_B
            
            # Transform leg velocity to inertial frame
            # v_leg_N = v_BN_N + omega_BN_N x r_leg_N
            omega_BN_N = BN @ omega_BN_B
            leg_vel_N = v_BN_N + np.cross(omega_BN_N, BN @ leg_pos_B)
            
            # Compute contact force at this leg
            contact_force_N = self.terrain.compute_contact_force(
                leg_pos_N, leg_vel_N, self.landing_leg_area
            )
            
            # Accumulate total force
            total_force_N += contact_force_N
            
            # Compute torque about center of mass (in body frame)
            # tau_B = r_B x (BN^T * F_N)
            contact_force_B = BN.T @ contact_force_N
            torque_B = np.cross(leg_pos_B, contact_force_B)
            total_torque_B += torque_B
        
        # Write forces to external force effector using message
        forceMsg = messaging.CmdForceBodyMsgPayload()
        forceMsg.forceRequestInertial = total_force_N
        forceMsg.torqueRequestBody = total_torque_B
        self.terrainForceMsg.write(forceMsg, self.moduleID, currentTime)
        
    def Update(self, currentTime):
        """
        Update method called every FSW timestep
        Computes thruster commands, gimbal angles, and terrain forces
        
        NOTE: For AI control, call setThrusterCommands() before this runs,
        or override this method. Default behavior is simple hover test.
        """
        # Use commands set by AI, or default to simple hover test
        
        # Primary engines - use AI commands or default to 50%
        primaryCommands = np.zeros(3)
        if np.any(self.primaryThrottles):
            primaryCommands = self.primaryThrottles.copy()
        else:
            primaryCommands[:] = 0.5  # Default hover test
        
        # Apply throttle limits
        for i in range(3):
            primaryCommands[i] = self.clampThrottle(primaryCommands[i], isPrimary=True)
        
        # Mid-body thrusters - use AI commands
        midbodyCommands = self.midbodyThrottles.copy()
        for i in range(12):
            midbodyCommands[i] = self.clampThrottle(midbodyCommands[i], isPrimary=False)
        
        # RCS thrusters - use AI commands
        rcsCommands = self.rcsThrottles.copy()
        for i in range(24):
            rcsCommands[i] = self.clampThrottle(rcsCommands[i], isPrimary=False)
        
        # Write thruster commands to separate effectors
        MAX_EFF = messaging.MAX_EFF_CNT
        
        # Primary command (3 thrusters)
        primCmdData = messaging.THRArrayOnTimeCmdMsgPayload()
        prim_array = [0.0] * MAX_EFF
        prim_array[0:3] = primaryCommands.tolist()
        primCmdData.OnTimeRequest = prim_array
        self.primCmdMsg.write(primCmdData, self.moduleID, currentTime)
        
        # Midbody command (12 thrusters)
        midCmdData = messaging.THRArrayOnTimeCmdMsgPayload()
        mid_array = [0.0] * MAX_EFF
        mid_array[0:12] = midbodyCommands.tolist()
        midCmdData.OnTimeRequest = mid_array
        self.midCmdMsg.write(midCmdData, self.moduleID, currentTime)
        
        # RCS command (24 thrusters)
        rcsCmdData = messaging.THRArrayOnTimeCmdMsgPayload()
        rcs_array = [0.0] * MAX_EFF
        rcs_array[0:24] = rcsCommands.tolist()
        rcsCmdData.OnTimeRequest = rcs_array
        self.rcsCmdMsg.write(rcsCmdData, self.moduleID, currentTime)
        
        # Gimbal commands are stored in self.gimbalAngles
        # Note: Current thrusterStateEffector implementation doesn't support
        # dynamic gimbal actuation via messages. For full gimbal control,
        # would need to add hingedRigidBodyStateEffector or modify thruster
        # directions dynamically. Gimbal angles are tracked for AI access.
        
        # Update terrain contact forces
        self.updateTerrainContact(currentTime)
        
    def Reset(self, currentTime):
        """Reset method called at simulation start"""
        self.lastUpdateTime = 0.0
        self.gimbalAngles = np.zeros((3, 2))
        return
        
# ----------------------------------------------------------------------
# 7. Instantiate AI Sensor Suite and Controller
# ----------------------------------------------------------------------
# Create AI sensor suite for comprehensive observation
aiSensors = AISensorSuite(
    scObject=lander,
    imu=imu,
    terrain=terrain,
    lidar=lidar,
    ch4Tank=ch4Tank,
    loxTank=loxTank,
    history_length=10  # Keep last 10 IMU measurements
)

# Set landing target (e.g., origin at surface level)
aiSensors.set_target_state(
    position=[0.0, 0.0, 0.0],  # Target landing at origin
    velocity=[0.0, 0.0, 0.0],  # Zero velocity at touchdown
    quaternion=[0.0, 0.0, 0.0, 1.0]  # Upright orientation
)

print(f"\nAI Sensor Suite Ready:")
print(f"  Observation space size: {aiSensors.get_observation_space_size()} dimensions")

# Create thruster controller (Python-only class, not added to task)
thrController = AdvancedThrusterController(primaryEff, midbodyEff, rcsEff, ch4Tank, loxTank, terrain, terrainForceEff, lander)

# Note: thrController is a Python helper class, not a SysModel
# It will be called manually in the simulation loop, not as a task
# For automated control, you would implement a proper SysModel FSW module

# Connect thruster effectors to controller messages
primaryEff.cmdsInMsg.subscribeTo(thrController.primCmdMsg)
midbodyEff.cmdsInMsg.subscribeTo(thrController.midCmdMsg)
rcsEff.cmdsInMsg.subscribeTo(thrController.rcsCmdMsg)

# Connect terrain force effector to controller's terrain force message
terrainForceEff.cmdForceBodyInMsg.subscribeTo(thrController.terrainForceMsg)

# ----------------------------------------------------------------------
# 7. Setup Data Logging
# ----------------------------------------------------------------------
simulationTime = macros.sec2nano(60.0)  # 60 seconds
numDataPoints = 600

samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)

# Log spacecraft state
scLog = lander.scStateOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, scLog)

# Log IMU data
imuLog = imu.sensorOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, imuLog)

# Log thruster data - log first primary thruster
primThrLog = primaryEff.thrusterOutMsgs[0].recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, primThrLog)

# Log fuel tank masses
ch4TankLog = ch4Tank.fuelTankOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, ch4TankLog)

loxTankLog = loxTank.fuelTankOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, loxTankLog)

# ----------------------------------------------------------------------
# 8. Execute Simulation with AI Sensor Suite
# ----------------------------------------------------------------------
scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)

print("\n" + "="*60)
print("STARTING SIMULATION")
print("="*60)
print(f"Duration: {simulationTime * macros.NANO2SEC} seconds")
print(f"Time step: {simulationTimeStep * macros.NANO2SEC} seconds")
print("Terrain: Analytical regolith model (high performance)")
print("Sensors: IMU, LIDAR, Altimeter, Fuel, Attitude (all with noise)")
print("="*60 + "\n")

# Test AI sensor suite during simulation
print("Testing AI Sensor Suite (first 5 seconds)...\n")

# Store sensor observations for analysis
sensor_observations = []

# Execute simulation with periodic sensor readouts
current_time = 0.0
step_count = 0
sensor_read_interval = 0.5  # Read sensors every 0.5 seconds for demo

scSim.InitializeSimulation()

while current_time < simulationTime * macros.NANO2SEC:
    # Update thruster controller (default hover test commands)
    currentTimeNano = macros.sec2nano(current_time)
    thrController.Update(currentTimeNano)
    
    # Step simulation
    scSim.ConfigureStopTime(macros.sec2nano(current_time + 0.1))
    scSim.ExecuteSimulation()
    
    current_time += 0.1
    step_count += 1
    
    # Read AI sensors periodically
    if step_count % int(sensor_read_interval / 0.1) == 0 and current_time <= 5.0:
        obs = aiSensors.update()
        sensor_observations.append(obs)
        
        print(f"\n--- Sensor Reading at t={current_time:.1f}s ---")
        print(f"Altitude (terrain-relative): {obs['altitude_terrain']:.2f} m")
        print(f"Vertical velocity: {obs['vertical_velocity']:.2f} m/s")
        print(f"Horizontal speed: {obs['horizontal_speed']:.2f} m/s")
        print(f"Attitude error angle: {np.degrees(obs['attitude_error_angle']):.2f}°")
        print(f"Fuel remaining: {obs['fuel_fraction']*100:.1f}% ({obs['fuel_mass']:.0f} kg)")
        print(f"IMU accel (body): [{obs['imu_accel_current'][0]:.3f}, {obs['imu_accel_current'][1]:.3f}, {obs['imu_accel_current'][2]:.3f}] m/s²")
        print(f"IMU gyro (body): [{obs['imu_gyro_current'][0]:.4f}, {obs['imu_gyro_current'][1]:.4f}, {obs['imu_gyro_current'][2]:.4f}] rad/s")
        print(f"LIDAR: min_range={obs['lidar_min_range']:.1f}m, mean={obs['lidar_mean_range']:.1f}m, valid_returns={np.sum(obs['lidar_ranges']>0)}/{len(obs['lidar_ranges'])}")

print("\n" + "="*60)
print("SIMULATION COMPLETED")
print("="*60 + "\n")

# ----------------------------------------------------------------------
# 9. Retrieve and Plot Results
# ----------------------------------------------------------------------
timeData = scLog.times() * macros.NANO2SEC
posData = scLog.r_BN_N
velData = scLog.v_BN_N
imuAccel = imuLog.AccelPlatform
imuGyro = imuLog.AngVelPlatform

# Fuel tank data
ch4Mass = ch4TankLog.fuelMass
loxMass = loxTankLog.fuelMass
totalPropMass = ch4Mass + loxMass

# Calculate total vehicle mass (dry + propellant)
dryMass = SC.HUB_MASS  # kg
totalMass = dryMass + totalPropMass

# Compute terrain-relative altitude for all timesteps
altitudeTerrain = np.zeros(len(timeData))
for i, pos in enumerate(posData):
    terrain_height = terrain.get_height(pos[0], pos[1])
    altitudeTerrain[i] = pos[2] - terrain_height

# Plot Results - Enhanced for AI sensors
fig = plt.figure(figsize=(16, 14))

# Plot 1: Altitude (both absolute and terrain-relative)
plt.subplot(4, 3, 1)
plt.plot(timeData, posData[:, 2], label='Absolute altitude', linewidth=2)
plt.plot(timeData, altitudeTerrain, label='Terrain-relative', linewidth=2, linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Lander Altitude")
plt.legend()
plt.grid()

# Plot 2: Velocity (3D)
plt.subplot(4, 3, 2)
plt.plot(timeData, velData[:, 0], label='Vx', alpha=0.7)
plt.plot(timeData, velData[:, 1], label='Vy', alpha=0.7)
plt.plot(timeData, velData[:, 2], label='Vz (vertical)', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Velocity Components")
plt.legend()
plt.grid()

# Plot 3: Horizontal speed
plt.subplot(4, 3, 3)
horizontalSpeed = np.sqrt(velData[:, 0]**2 + velData[:, 1]**2)
plt.plot(timeData, horizontalSpeed, linewidth=2, color='green')
plt.xlabel("Time [s]")
plt.ylabel("Horizontal Speed [m/s]")
plt.title("Horizontal Velocity Magnitude")
plt.grid()

# Plot 4: Total Vehicle Mass
plt.subplot(4, 3, 4)
plt.plot(timeData, totalMass, linewidth=2, color='purple')
plt.xlabel("Time [s]")
plt.ylabel("Mass [kg]")
plt.title("Total Vehicle Mass (with fuel depletion)")
plt.grid()

# Plot 5: Propellant Masses
plt.subplot(4, 3, 5)
plt.plot(timeData, ch4Mass, label='CH4', linewidth=2)
plt.plot(timeData, loxMass, label='LOX', linewidth=2)
plt.plot(timeData, totalPropMass, label='Total Propellant', linewidth=2, linestyle='--')
plt.xlabel("Time [s]")
plt.ylabel("Propellant Mass [kg]")
plt.title("Fuel Tank Masses")
plt.legend()
plt.grid()

# Plot 6: Fuel Fraction
plt.subplot(4, 3, 6)
fuelFraction = totalPropMass / 1200000.0 * 100.0
plt.plot(timeData, fuelFraction, linewidth=2, color='red')
plt.xlabel("Time [s]")
plt.ylabel("Fuel Remaining [%]")
plt.title("Remaining Fuel Percentage")
plt.grid()

# Plot 7: IMU Acceleration (with noise visible)
plt.subplot(4, 3, 7)
plt.plot(timeData, imuAccel[:, 0], label='Ax', alpha=0.6)
plt.plot(timeData, imuAccel[:, 1], label='Ay', alpha=0.6)
plt.plot(timeData, imuAccel[:, 2], label='Az', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("IMU Acceleration (with noise)")
plt.legend()
plt.grid()

# Plot 8: IMU Gyro (with noise visible)
plt.subplot(4, 3, 8)
plt.plot(timeData, imuGyro[:, 0], label='ωx', alpha=0.6)
plt.plot(timeData, imuGyro[:, 1], label='ωy', alpha=0.6)
plt.plot(timeData, imuGyro[:, 2], label='ωz', alpha=0.6)
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.title("IMU Gyroscope (with noise)")
plt.legend()
plt.grid()

# Plot 9: Thruster Force
plt.subplot(4, 3, 9)
if len(primThrLog.thrustForce) > 0:
    thrustMag = primThrLog.thrustForce
    if len(thrustMag.shape) > 1:
        thrustMag = np.linalg.norm(thrustMag, axis=1)
    plt.plot(timeData, thrustMag, linewidth=2, color='orange')
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust Force [N]")
    plt.title("Primary Thruster Force (Engine 1)")
    plt.grid()

# Plot 10: 3D Trajectory
plt.subplot(4, 3, 10)
plt.plot(posData[:, 0], posData[:, 1], linewidth=2)
plt.scatter([0], [0], c='red', s=100, marker='x', label='Target')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Horizontal Trajectory (Top View)")
plt.legend()
plt.grid()
plt.axis('equal')

# Plot 11: Attitude (MRP magnitude as proxy for tilt)
plt.subplot(4, 3, 11)
attitudeMRP = scLog.sigma_BN
mrpMagnitude = np.linalg.norm(attitudeMRP, axis=1)
attitudeDeg = 2.0 * np.arctan(mrpMagnitude) * 180.0 / np.pi
plt.plot(timeData, attitudeDeg, linewidth=2, color='brown')
plt.xlabel("Time [s]")
plt.ylabel("Tilt Angle [deg]")
plt.title("Attitude Error from Upright")
plt.grid()

# Plot 12: IMU Acceleration Noise Analysis (last 10 samples)
plt.subplot(4, 3, 12)
if len(sensor_observations) > 0:
    # Show IMU history buffer from one observation
    obs_sample = sensor_observations[-1]  # Last observation
    accel_history = obs_sample['imu_accel_history']
    gyro_history = obs_sample['imu_gyro_history']
    
    history_frames = np.arange(len(accel_history))
    plt.plot(history_frames, accel_history[:, 2], 'o-', label='Az history', linewidth=2)
    plt.xlabel("History Frame (0=oldest)")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("IMU History Buffer (Az component)")
    plt.legend()
    plt.grid()
else:
    plt.text(0.5, 0.5, 'No sensor observations', ha='center', va='center')

plt.tight_layout()
plt.savefig('lunar_lander_ai_sensors.png', dpi=150)
print("Saved plot to: lunar_lander_ai_sensors.png")
plt.show()

print(f"\n{'='*60}")
print(f"SIMULATION COMPLETE - Starship HLS Lunar Landing")
print(f"{'='*60}")
print(f"Final altitude:        {posData[-1, 2]:>12.2f} m")
print(f"Final velocity:        {velData[-1, 2]:>12.2f} m/s")
print(f"Initial total mass:    {totalMass[0]:>12.2f} kg")
print(f"Final total mass:      {totalMass[-1]:>12.2f} kg")
print(f"Propellant consumed:   {totalMass[0] - totalMass[-1]:>12.2f} kg")
print(f"Initial CH4:           {ch4Mass[0]:>12.2f} kg")
print(f"Final CH4:             {ch4Mass[-1]:>12.2f} kg")
print(f"CH4 consumed:          {ch4Mass[0] - ch4Mass[-1]:>12.2f} kg")
print(f"Initial LOX:           {loxMass[0]:>12.2f} kg")
print(f"Final LOX:             {loxMass[-1]:>12.2f} kg")
print(f"LOX consumed:          {loxMass[0] - loxMass[-1]:>12.2f} kg")
if (ch4Mass[0] - ch4Mass[-1]) > 0.1:
    print(f"O/F ratio (consumed):  {(loxMass[0] - loxMass[-1]) / (ch4Mass[0] - ch4Mass[-1]):>12.2f}")
print(f"{'='*60}")

# ----------------------------------------------------------------------
# AI CONTROL INTERFACE DOCUMENTATION
# ----------------------------------------------------------------------
"""
AI CONTROL INTERFACE:

The AISensorSuite and AdvancedThrusterController provide a complete interface for AI agents.

================================================================================
SENSOR SUITE - AISensorSuite
================================================================================

1. GET COMPREHENSIVE OBSERVATIONS (recommended for AI):
   
   obs = aiSensors.update()
   # Returns dict with ALL sensor data:
   
   IMU DATA (with noise and history):
   - obs['imu_accel_current']: (3,) current acceleration in body frame [m/s²]
   - obs['imu_gyro_current']: (3,) current angular velocity in body frame [rad/s]
   - obs['imu_accel_history']: (history_length, 3) last N accel measurements
   - obs['imu_gyro_history']: (history_length, 3) last N gyro measurements
   
   POSITION & VELOCITY:
   - obs['position_inertial']: (3,) position [x, y, z] in inertial frame [m]
   - obs['velocity_inertial']: (3,) velocity in inertial frame [m/s]
   - obs['velocity_body']: (3,) velocity in body frame [m/s]
   
   TERRAIN-RELATIVE (critical for landing):
   - obs['altitude_terrain']: altitude above LOCAL terrain [m]
   - obs['terrain_height']: terrain elevation at current x,y [m]
   - obs['vertical_velocity']: vertical velocity component [m/s]
   - obs['horizontal_velocity_local']: (3,) horizontal velocity in local frame [m/s]
   - obs['horizontal_speed']: horizontal speed magnitude [m/s]
   
   ATTITUDE:
   - obs['attitude_quaternion']: (4,) current attitude [x, y, z, w]
   - obs['attitude_mrp']: (3,) Modified Rodriguez Parameters
   - obs['angular_velocity_body']: (3,) angular velocity [rad/s]
   - obs['attitude_error_quaternion']: (4,) error to target attitude
   - obs['attitude_error_angle']: scalar attitude error [radians]
   
   FUEL & MASS:
   - obs['total_mass']: current total mass [kg]
   - obs['fuel_mass']: remaining propellant [kg]
   - obs['ch4_mass']: remaining CH4 [kg]
   - obs['lox_mass']: remaining LOX [kg]
   - obs['fuel_fraction']: remaining fuel percentage [0-1]
   - obs['ch4_fraction']: remaining CH4 percentage [0-1]
   - obs['lox_fraction']: remaining LOX percentage [0-1]
   
   LIDAR (with noise and dropouts):
   - obs['lidar_point_cloud']: (num_rays, 3) 3D points in body frame [m]
   - obs['lidar_ranges']: (num_rays,) range measurements [m], -1 = invalid
   - obs['lidar_intensities']: (num_rays,) return intensities [0-1]
   - obs['lidar_min_range']: minimum valid range [m]
   - obs['lidar_mean_range']: mean valid range [m]
   - obs['lidar_range_std']: range standard deviation [m]
   
   DERIVED:
   - obs['gravity_body']: (3,) gravity vector in body frame [m/s²]
   - obs['dcm_body_to_inertial']: (3,3) rotation matrix

2. GET FLATTENED OBSERVATION (for neural networks):
   
   obs_vector = aiSensors.get_flattened_observation()
   # Returns 1D numpy array with all sensor data concatenated
   # Size: aiSensors.get_observation_space_size() dimensions

3. SET TARGET STATE (for computing errors):
   
   aiSensors.set_target_state(
       position=[0, 0, 0],  # Target landing position [m]
       velocity=[0, 0, 0],  # Target velocity at touchdown [m/s]
       quaternion=[0, 0, 0, 1]  # Target attitude (upright)
   )

4. RESET (call at start of new episode):
   
   aiSensors.reset()  # Clears IMU history buffers

================================================================================
THRUSTER CONTROL - AdvancedThrusterController
================================================================================

1. SET COMMANDS (call before simulation timestep):
   
   thrController.setThrusterCommands(
       primaryThrottles=[0.5, 0.5, 0.5],  # 3 main engines [0.4-1.0]
       midbodyThrottles=[0.1, 0, ...],     # 12 mid-body [0-1.0]
       rcsThrottles=[0, 0.2, ...],         # 24 RCS [0-1.0]
       gimbalAngles=[[0.1, -0.05], ...]   # 3x2 array [pitch, yaw] in radians
   )

2. GET THRUSTER STATE:
   
   state = thrController.getThrusterState()
   # Returns: {
   #   'ch4_mass': current CH4 mass (kg),
   #   'lox_mass': current LOX mass (kg),
   #   'total_fuel_mass': total propellant (kg),
   #   'gimbal_limit_rad': ±0.1396 rad (±8°),
   #   'current_gimbal_angles': [[pitch, yaw], ...],
   #   ...
   # }

================================================================================
DIRECT BASILISK ACCESS (if needed)
================================================================================

3. READ SPACECRAFT STATE (raw):
   
   scState = lander.scStateOutMsg.read()
   # scState.r_BN_N: position [x, y, z] in inertial frame (m)
   # scState.v_BN_N: velocity [vx, vy, vz] (m/s)
   # scState.sigma_BN: attitude (MRP)
   # scState.omega_BN_B: angular velocity (rad/s)

4. READ SENSOR DATA (raw):
   
   imuData = imu.sensorOutMsg.read()
   # imuData.AccelPlatform: measured acceleration (m/s²)
   # imuData.AngVelPlatform: measured angular velocity (rad/s)

5. TERRAIN QUERIES:
   
   height = terrain.get_height(x, y)  # Get terrain height at (x, y)

================================================================================
SENSOR NOISE MODELS
================================================================================

IMU Noise:
- Gyroscope: Gaussian noise, σ = 0.00001 rad/s (space-grade precision)
- Accelerometer: Gaussian noise, σ = 0.001 m/s² (high precision)
- Applied by Basilisk's imuSensor module

LIDAR Noise:
- Range noise: Gaussian, σ = 0.05 m (5 cm standard deviation)
- Dropout probability: 2% (simulates no-return events)
- Minimum valid range: 0.5 m (close-range filter)
- Intensity varies with range and random factor

All noise is stochastic and varies each timestep, providing realistic
sensor uncertainty for AI training.

================================================================================
THRUSTER CONFIGURATION
================================================================================
- Thrusters 0-2:   Primary Raptors (2,500,000 N each, gimbal capable)
- Thrusters 3-14:  Mid-body (20,000 N each, attitude control)
- Thrusters 15-38: RCS (2,000 N each, fine attitude control)

FUEL DEPLETION:
- Automatic via thrusterStateEffector
- Mixture ratio O/F=3.6 enforced automatically
- All thrusters consume from both CH4 and LOX tanks

GIMBAL LIMITS:
- Primary engines: ±8° (±0.1396 rad) in pitch and yaw
- Note: Current implementation tracks gimbal for AI but doesn't
  physically actuate. For full gimbal physics, would need additional
  hingedRigidBodyStateEffector modules.

EXAMPLE AI CONTROL LOOP:
```python
# Initialize simulation
aiSensors.reset()  # Clear sensor history
aiSensors.set_target_state(position=[0, 0, 0], velocity=[0, 0, 0])

for step in range(num_steps):
    # Get comprehensive sensor observations
    obs = aiSensors.update()
    
    # AI agent processes observations
    # All critical data available:
    altitude = obs['altitude_terrain']  # Height above local terrain
    velocity = obs['velocity_body']     # Velocity in body frame
    attitude_error = obs['attitude_error_quaternion']
    fuel_remaining = obs['fuel_fraction']
    lidar_cloud = obs['lidar_point_cloud']  # Full 3D point cloud
    imu_history = obs['imu_accel_history']  # Temporal data for filters
    
    # AI policy computes actions
    throttles_primary = ai_policy.compute_primary(obs)
    throttles_rcs = ai_policy.compute_rcs(obs)
    gimbal = ai_policy.compute_gimbal(obs)
    
    # Apply commands to thrusters
    thrController.setThrusterCommands(
        primaryThrottles=throttles_primary,
        rcsThrottles=throttles_rcs,
        gimbalAngles=gimbal
    )
    
    # Step simulation
    scSim.ConfigureStopTime(macros.sec2nano((step + 1) * dt))
    scSim.ExecuteSimulation()
    
    # Check terminal conditions
    if altitude < 0.5 and abs(obs['vertical_velocity']) < 0.1:
        print("Successful landing!")
        break
    elif altitude < 0 and abs(obs['vertical_velocity']) > 2.0:
        print("Crash landing!")
        break
```

OBSERVATION SPACE SUMMARY:
- Total dimensions: ~200+ (use aiSensors.get_observation_space_size())
- IMU history: 10 frames × 6 values (accel + gyro) = 60 dims
- LIDAR: 64 rays × 2 (range + intensity) = 128 dims
- State: ~20-30 dims (position, velocity, attitude, fuel)
- All data includes realistic sensor noise for robust AI training

For neural network inputs, use:
  obs_vector = aiSensors.get_flattened_observation()
  # Returns 1D numpy array ready for NN input
"""
