# Basilisk API Reference for Lunar Landing Missions

## Overview
This document provides a comprehensive reference for using the Basilisk astrodynamics framework for lunar landing simulations, specifically for the Starship HLS (Human Landing System) project.

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Simulation Setup](#simulation-setup)
3. [Spacecraft Dynamics](#spacecraft-dynamics)
4. [Gravity Models](#gravity-models)
5. [Propulsion Systems](#propulsion-systems)
6. [Fuel Management](#fuel-management)
7. [Sensors](#sensors)
8. [Atmospheric Models](#atmospheric-models)
9. [Flight Software (FSW) Modules](#flight-software-fsw-modules)
10. [Data Logging](#data-logging)
11. [Messaging System](#messaging-system)
12. [Best Practices](#best-practices)

---

## Core Architecture

### Basilisk Framework Principles
Basilisk is a modular spacecraft simulation framework where:
- **Modules** are atomic spacecraft behaviors/components
- **Messages** connect modules via input/output interfaces
- **Tasks** group modules with the same update rate
- **Processes** organize related tasks (e.g., dynamics vs. FSW)

### Key Import Statements
```python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add Basilisk path
basiliskPath = os.path.join(os.path.dirname(__file__), 'basilisk', 'dist3')
sys.path.insert(0, basiliskPath)

# Core imports
from Basilisk.simulation import spacecraft, thrusterDynamicEffector, imuSensor
from Basilisk.simulation import fuelTank, dragDynamicEffector, exponentialAtmosphere
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, unitTestSupport
from Basilisk.architecture import messaging
```

---

## Simulation Setup

### Creating Simulation Base
```python
# Create simulation base class
scSim = SimulationBaseClass.SimBaseClass()

# Create process and tasks
simProcessName = "simProcess"
dynTaskName = "dynTask"
fswTaskName = "fswTask"

dynProcess = scSim.CreateNewProcess(simProcessName)
simulationTimeStep = macros.sec2nano(0.1)  # 0.1 second timestep
fswTimeStep = macros.sec2nano(0.5)         # FSW runs at 2 Hz

# Add tasks to process
dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simulationTimeStep))
dynProcess.addTask(scSim.CreateNewTask(fswTaskName, fswTimeStep))
```

### Time Conversion Utilities
```python
# Macros for time conversion
macros.sec2nano(seconds)    # Convert seconds to nanoseconds
macros.NANO2SEC             # Constant for nano to seconds
macros.min2nano(minutes)    # Convert minutes to nanoseconds
macros.hour2nano(hours)     # Convert hours to nanoseconds
```

### Simulation Execution
```python
# Initialize and run
scSim.InitializeSimulation()
scSim.ConfigureStopTime(simulationTime)
scSim.ExecuteSimulation()
```

---

## Spacecraft Dynamics

### Spacecraft Module: `spacecraft.Spacecraft()`

The core dynamics module for rigid spacecraft with state effectors and dynamic effectors.

#### Creating a Spacecraft
```python
lander = spacecraft.Spacecraft()
lander.ModelTag = "Starship_HLS"  # Name identifier
```

#### Hub Properties (Rigid Body)
```python
# Mass properties
lander.hub.mHub = 105000.0  # kg - Dry mass (structure + payload)

# Center of mass in body frame [m]
lander.hub.r_BcB_B = np.zeros(3)  # At vehicle origin

# Inertia tensor [kg·m²] about center of mass
lander.hub.IHubPntBc_B = np.array([
    [231513125.0, 0.0, 0.0],
    [0.0, 231513125.0, 0.0],
    [0.0, 0.0, 14276250.0]
])
```

#### Initial State Conditions
```python
# Position [m] in inertial frame N
lander.hub.r_CN_NInit = np.array([0., 0., 1500.0])  # 1500m altitude

# Velocity [m/s] in inertial frame
lander.hub.v_CN_NInit = np.array([0., 0., -10.0])   # Descending at 10 m/s

# Attitude (MRP - Modified Rodrigues Parameters)
lander.hub.sigma_BNInit = np.array([0., 0., 0.])    # No rotation

# Angular velocity [rad/s] in body frame
lander.hub.omega_BN_BInit = np.zeros(3)             # No rotation
```

#### Adding to Simulation
```python
scSim.AddModelToTask(dynTaskName, lander)
```

#### Important Methods
- `addStateEffector(effector)` - Add state effector (fuel tanks, RWs, flexible bodies)
- `addDynamicEffector(effector)` - Add dynamic effector (thrusters, drag, solar pressure)
- `scStateOutMsg` - Output message containing spacecraft state

#### Output Message: `SCStatesMsgPayload`
```python
# Access via recorder
scLog = lander.scStateOutMsg.recorder(samplingTime)
# Contains:
# - r_BN_N: position [m]
# - v_BN_N: velocity [m/s]
# - sigma_BN: MRP attitude
# - omega_BN_B: angular velocity [rad/s]
```

---

## Gravity Models

### Gravity Factory: `simIncludeGravBody.gravBodyFactory()`

Creates celestial bodies with gravitational models.

#### Creating Moon Gravity
```python
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True  # Set as central reference body
gravFactory.addBodiesTo(lander)  # Attach to spacecraft
```

#### Available Body Creation Methods
- `createEarth()` - Earth with J2-J6 gravity harmonics
- `createMoon()` - Moon with spherical gravity model
- `createSun()` - Sun
- `createMars()` - Mars
- `createJupiter()` - Jupiter
- `createVenus()` - Venus
- Custom bodies via `createCustomGravObject()`

#### Gravity Properties
```python
moon.radEquator  # Equatorial radius [m]
moon.mu          # Gravitational parameter [m³/s²]
```

#### Notes
- Moon gravity: μ = 4.9028 × 10¹² m³/s²
- Moon radius: 1,737,400 m
- Use SPICE for high-fidelity ephemeris data

---

## Propulsion Systems

### Thruster Dynamic Effector: `thrusterDynamicEffector.ThrusterDynamicEffector()`

Models thrusters that apply forces to spacecraft. Does NOT model fuel consumption directly.

#### Creating Thruster Effector
```python
primaryEff = thrusterDynamicEffector.ThrusterDynamicEffector()
primaryEff.ModelTag = "PrimaryThrusters"
scSim.AddModelToTask(dynTaskName, primaryEff)
lander.addDynamicEffector(primaryEff)
```

#### Configuring Individual Thrusters
```python
thr = thrusterDynamicEffector.THRSimConfig()
thr.thrLoc_B = np.array([3.5, 0.0, -24.5])   # Position [m] in body frame
thr.thrDir_B = np.array([0., 0., 1.])         # Direction (unit vector)
thr.MaxThrust = 2500000.0                      # Maximum thrust [N]
primaryEff.addThruster(thr)
```

#### Commanding Thrusters
```python
# Create command message (must be length MAX_EFF_CNT, typically 36)
MAX_EFF = messaging.MAX_EFF_CNT

primCmdData = messaging.THRArrayOnTimeCmdMsgPayload()
prim_array = [0.0] * MAX_EFF
prim_array[0:3] = [0.5, 0.5, 0.5]  # 50% throttle for first 3 thrusters
primCmdData.OnTimeRequest = prim_array
primCmdWriter = messaging.THRArrayOnTimeCmdMsg().write(primCmdData)

# Subscribe effector to command
primaryEff.cmdsInMsg.subscribeTo(primCmdWriter)
```

#### Output Messages
```python
# Each thruster outputs a message
primThrLog = primaryEff.thrusterOutMsgs[0].recorder(samplingTime)
# Contains:
# - thrustForce: 3D force vector [N]
# - thrustLocation: position in body frame [m]
```

#### Important Notes
- Use multiple effectors for different thruster groups (main, RCS, DV)
- Throttle values: 0.0 (off) to 1.0 (full thrust)
- Fuel consumption must be tracked separately or use `thrusterStateEffector`

### Thruster State Effector (Alternative)
For high-fidelity fuel coupling, use `thrusterStateEffector.ThrusterStateEffector()`:
- Automatically depletes fuel tanks
- Models Isp and mass flow
- More complex setup but integrated fuel dynamics

---

## Fuel Management

### Fuel Tank: `fuelTank.FuelTank()`

Models propellant tanks as state effectors that affect spacecraft mass/inertia.

#### Creating Fuel Tanks
```python
# CH4 Tank (Methane)
ch4Tank = fuelTank.FuelTank()
ch4Tank.ModelTag = "CH4_Tank"
ch4TankModel = fuelTank.FuelTankModelConstantVolume()
ch4TankModel.propMassInit = 260869.565  # kg - Initial fuel mass
ch4TankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]  # Tank CM in tank frame

# Compute tank radius from volume: V = (4/3)πr³
ch4TankRadius = (3.0 * 617.005 / (4.0 * np.pi)) ** (1.0/3.0)
ch4TankModel.radiusTankInit = ch4TankRadius  # m

ch4Tank.setTankModel(ch4TankModel)
ch4Tank.r_TB_B = [[0.0], [0.0], [-10.0]]  # Position in body frame [m]
ch4Tank.nameOfMassState = "ch4TankMass"
lander.addStateEffector(ch4Tank)
scSim.AddModelToTask(dynTaskName, ch4Tank)
```

#### Available Tank Models
1. **FuelTankModelConstantVolume** - Fixed volume, mass changes
2. **FuelTankModelConstantDensity** - Fixed density, volume changes
3. **FuelTankModelEmptying** - Simple emptying model
4. **FuelTankModelUniformBurn** - Uniform fuel consumption
5. **FuelTankModelCentrifugalBurn** - Centrifugal burn pattern

#### Manual Fuel Depletion
```python
# For thrusterDynamicEffector, track fuel manually
def updateFuelConsumption(dt, throttleValues):
    # Vacuum Raptor parameters
    Isp = 375.0  # seconds
    g0 = 9.80665  # m/s²
    maxThrust = 2500000.0  # N
    massFlowPerEngine = maxThrust / (Isp * g0)  # kg/s
    
    # Calculate total mass flow
    totalMassFlow = sum(throttle * massFlowPerEngine for throttle in throttleValues[:3])
    
    # Split by mixture ratio (O/F = 3.6)
    mixtureRatio = 3.6
    ch4MassFlow = totalMassFlow / (1.0 + mixtureRatio)
    loxMassFlow = totalMassFlow * mixtureRatio / (1.0 + mixtureRatio)
    
    # Deplete tanks (requires custom tank access)
    ch4Consumed = ch4MassFlow * dt
    loxConsumed = loxMassFlow * dt
```

#### Output Messages
```python
ch4TankLog = ch4Tank.fuelTankOutMsg.recorder(samplingTime)
# Contains:
# - fuelMass: current fuel mass [kg]
# - fuelMassDot: fuel consumption rate [kg/s]
```

#### Important Formulas
- **Mass Flow Rate**: ṁ = F / (Isp × g₀)
- **Specific Impulse**: Isp [seconds]
- **Standard Gravity**: g₀ = 9.80665 m/s²
- **Mixture Ratio**: O/F = mₒₓ / m_fuel

---

## Sensors

### IMU Sensor: `imuSensor.ImuSensor()`

Inertial Measurement Unit providing acceleration and angular rate measurements.

#### Creating IMU
```python
imu = imuSensor.ImuSensor()
imu.ModelTag = "StarshipIMU"

# Set orientation (yaw, pitch, roll in radians)
imu.setBodyToPlatformDCM(0.0, 0.0, 0.0)  # Aligned with body frame

# Subscribe to spacecraft state
imu.scStateInMsg.subscribeTo(lander.scStateOutMsg)

# Set noise bounds
imu.setErrorBoundsGyro([0.00001] * 3)   # Gyro noise [rad/s]
imu.setErrorBoundsAccel([0.001] * 3)    # Accel noise [m/s²]

scSim.AddModelToTask(dynTaskName, imu)
```

#### Output Messages
```python
imuLog = imu.sensorOutMsg.recorder(samplingTime)
# Contains:
# - AccelPlatform: 3D acceleration [m/s²]
# - AngVelPlatform: 3D angular velocity [rad/s]
# - DRFramePlatform: DCM (optional)
```

#### Noise Modeling
- Random walk for gyroscope drift
- Gaussian noise on accelerometer
- Bias and scale factor errors (optional)

---

## Atmospheric Models

### Exponential Atmosphere: `exponentialAtmosphere.ExponentialAtmosphere()`

Simple exponential decay atmosphere model.

#### Creating Atmosphere
```python
moonAtmo = exponentialAtmosphere.ExponentialAtmosphere()
moonAtmo.ModelTag = "MoonAtmosphere"
moonAtmo.planetRadius = moon.radEquator  # m
moonAtmo.scaleHeight = 100000.0          # m - Scale height
moonAtmo.baseDensity = 1e-15             # kg/m³ - Surface density
moonAtmo.envMinReach = -10000.0          # m - Min altitude
moonAtmo.envMaxReach = 10000.0           # m - Max altitude

# Add spacecraft to atmosphere model
moonAtmo.addSpacecraftToModel(lander.scStateOutMsg)
scSim.AddModelToTask(dynTaskName, moonAtmo)
```

#### Density Model
ρ(h) = ρ₀ × exp(-h / H)
- ρ₀: base density [kg/m³]
- H: scale height [m]
- h: altitude above surface [m]

#### For Earth-like Atmospheres
Use `simSetPlanetEnvironment.exponentialAtmosphere(atmo, "earth")` for preset values.

#### MSIS Atmosphere (High-Fidelity)
For more accurate Earth atmosphere:
```python
from Basilisk.simulation import msisAtmosphere
atmo = msisAtmosphere.MsisAtmosphere()
# Requires solar weather data messages
```

### Drag Dynamic Effector: `dragDynamicEffector.DragDynamicEffector()`

Models atmospheric drag forces.

#### Creating Drag Effector
```python
dragEffector = dragDynamicEffector.DragDynamicEffector()
dragEffector.ModelTag = "DragEffector"
dragEffector.coreParams.projectedArea = 63.617  # m² - Frontal area
dragEffector.coreParams.dragCoeff = 0.6         # Drag coefficient
dragEffector.coreParams.comOffset = [0.0, 0.0, 0.0]  # COM offset [m]

# Add to spacecraft
lander.addDynamicEffector(dragEffector)
scSim.AddModelToTask(dynTaskName, dragEffector)

# Connect atmosphere density
dragEffector.atmoDensInMsg.subscribeTo(moonAtmo.envOutMsgs[0])
```

#### Drag Force
F_drag = 0.5 × ρ × v² × C_D × A
- ρ: atmospheric density [kg/m³]
- v: velocity magnitude [m/s]
- C_D: drag coefficient
- A: projected area [m²]

---

## Flight Software (FSW) Modules

### Attitude Guidance

#### Inertial 3D: `inertial3D.inertial3D()`
Points spacecraft to inertial attitude.

```python
from Basilisk.fswAlgorithms import inertial3D

guidModule = inertial3D.inertial3D()
guidModule.ModelTag = "inertial3D"
guidModule.sigma_R0N = [0., 0., 0.]  # Desired MRP attitude
scSim.AddModelToTask(fswTaskName, guidModule)
```

### Attitude Control

#### MRP Feedback: `mrpFeedback.mrpFeedback()`
PD controller for attitude control.

```python
from Basilisk.fswAlgorithms import mrpFeedback

ctrlModule = mrpFeedback.mrpFeedback()
ctrlModule.ModelTag = "mrpFeedback"
ctrlModule.K = 3.5      # Proportional gain
ctrlModule.P = 30.0     # Derivative gain
ctrlModule.Ki = -1.0    # Integral gain (negative to disable)
ctrlModule.integralLimit = 2. / ctrlModule.Ki * 0.1

# Connect to guidance and navigation
ctrlModule.guidInMsg.subscribeTo(guidModule.attGuidOutMsg)
ctrlModule.vehConfigInMsg.subscribeTo(lander.vehicleConfigOutMsg)

scSim.AddModelToTask(fswTaskName, ctrlModule)
```

### Thruster Mapping

#### Thruster Force Mapping: `thrForceMapping.thrForceMapping()`
Maps desired torque to thruster forces.

```python
from Basilisk.fswAlgorithms import thrForceMapping

thrMapModule = thrForceMapping.thrForceMapping()
thrMapModule.ModelTag = "thrForceMapping"
thrMapModule.controlAxes_B = [1, 0, 0,
                               0, 1, 0,
                               0, 0, 1]  # Full 3-axis control
thrMapModule.thrForceSign = 1  # +1 for on-pulse, -1 for off-pulse

# Connect inputs
thrMapModule.cmdTorqueInMsg.subscribeTo(ctrlModule.cmdTorqueOutMsg)
thrMapModule.thrConfigInMsg.subscribeTo(fswThrConfigMsg)
thrMapModule.vehConfigInMsg.subscribeTo(lander.vehicleConfigOutMsg)

scSim.AddModelToTask(fswTaskName, thrMapModule)
```

#### Thruster Firing Schmitt: `thrFiringSchmitt.thrFiringSchmitt()`
Converts thrust levels to on-time commands using Schmitt trigger.

```python
from Basilisk.fswAlgorithms import thrFiringSchmitt

thrFireModule = thrFiringSchmitt.thrFiringSchmitt()
thrFireModule.ModelTag = "thrFiringSchmitt"
thrFireModule.thrMinFireTime = 0.02  # Minimum fire time [s]
thrFireModule.level_on = 0.5         # On threshold
thrFireModule.level_off = 0.1        # Off threshold
thrFireModule.baseThrustState = 0    # 0 for ACS, 1 for DV

# Connect inputs
thrFireModule.thrForceInMsg.subscribeTo(thrMapModule.thrForceCmdOutMsg)
thrFireModule.thrConfInMsg.subscribeTo(fswThrConfigMsg)

scSim.AddModelToTask(fswTaskName, thrFireModule)

# Connect to thruster effector
primaryEff.cmdsInMsg.subscribeTo(thrFireModule.onTimeOutMsg)
```

### Navigation

#### Simple Navigation: `simpleNav.simpleNav()`
Perfect navigation (truth model).

```python
from Basilisk.fswAlgorithms import simpleNav

navModule = simpleNav.simpleNav()
navModule.ModelTag = "simpleNav"
navModule.scStateInMsg.subscribeTo(lander.scStateOutMsg)
scSim.AddModelToTask(fswTaskName, navModule)
```

---

## Data Logging

### Message Recorders

Record module outputs for analysis.

```python
# Spacecraft state
scLog = lander.scStateOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, scLog)

# IMU data
imuLog = imu.sensorOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, imuLog)

# Thruster forces
primThrLog = primaryEff.thrusterOutMsgs[0].recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, primThrLog)

# Fuel tank masses
ch4TankLog = ch4Tank.fuelTankOutMsg.recorder(samplingTime)
scSim.AddModelToTask(dynTaskName, ch4TankLog)
```

### Accessing Logged Data
```python
# After simulation
timeData = scLog.times() * macros.NANO2SEC  # Convert to seconds
posData = scLog.r_BN_N                       # Position [m]
velData = scLog.v_BN_N                       # Velocity [m/s]
attData = scLog.sigma_BN                     # Attitude MRP
rateData = scLog.omega_BN_B                  # Angular rate [rad/s]

# IMU data
imuAccel = imuLog.AccelPlatform
imuGyro = imuLog.AngVelPlatform

# Fuel data
ch4Mass = ch4TankLog.fuelMass
```

### Sampling Time Utility
```python
simulationTime = macros.sec2nano(60.0)  # 60 seconds
numDataPoints = 600

samplingTime = unitTestSupport.samplingTime(
    simulationTime, 
    simulationTimeStep, 
    numDataPoints
)
```

---

## Messaging System

### Message Architecture

Basilisk uses a publish-subscribe messaging system:
- **Output Messages** (`*OutMsg`) - Published by modules
- **Input Messages** (`*InMsg`) - Subscribed by modules
- **Message Payloads** (`*MsgPayload`) - Data structures

### Creating Standalone Messages

```python
# Create message payload
msgData = messaging.THRArrayOnTimeCmdMsgPayload()
msgData.OnTimeRequest = [0.5] * MAX_EFF

# Write message
msgWriter = messaging.THRArrayOnTimeCmdMsg().write(msgData)

# Subscribe to message
module.cmdsInMsg.subscribeTo(msgWriter)
```

### Common Message Types

#### Spacecraft State: `SCStatesMsgPayload`
```python
# Contains:
# - r_BN_N: position [m]
# - v_BN_N: velocity [m/s]
# - sigma_BN: MRP attitude
# - omega_BN_B: angular rate [rad/s]
```

#### Thruster Commands: `THRArrayOnTimeCmdMsgPayload`
```python
# Contains:
# - OnTimeRequest: array of throttle values [0-1]
```

#### Attitude Guidance: `AttGuidMsgPayload`
```python
# Contains:
# - sigma_BR: reference attitude MRP
# - omega_BR_B: reference angular rate [rad/s]
# - omega_RN_B: reference frame rate [rad/s]
# - domega_RN_B: reference acceleration [rad/s²]
```

#### Vehicle Configuration: `VehicleConfigMsgPayload`
```python
# Contains:
# - massSC: spacecraft mass [kg]
# - ISCPntB_B: inertia tensor [kg·m²]
# - CoM_B: center of mass [m]
```

---

## Best Practices

### 1. Coordinate Frames
- **N frame**: Inertial frame (typically planet-centered J2000)
- **B frame**: Body-fixed frame (spacecraft)
- **R frame**: Reference frame (for guidance)
- **T frame**: Tank frame (for fuel tanks)

### 2. Unit Conventions
- Position: meters [m]
- Velocity: meters/second [m/s]
- Time: nanoseconds (simulation), seconds (user)
- Mass: kilograms [kg]
- Force: Newtons [N]
- Torque: Newton-meters [N·m]
- Angular velocity: radians/second [rad/s]
- Attitude: MRP (Modified Rodrigues Parameters)

### 3. MRP Attitude Representation
```python
# MRP properties:
# - 3-parameter minimal representation
# - No singularities (with shadow set)
# - σ = tan(Φ/4) * ê
# where Φ is rotation angle, ê is rotation axis

# Shadow set: switch when |σ| > 1
sigma_shadow = -sigma / np.dot(sigma, sigma)
```

### 4. Simulation Time Steps
- **Dynamics**: 0.01 - 0.1 seconds (faster for stiff systems)
- **FSW**: 0.5 - 2.0 seconds (typical sensor/control rates)
- **Logging**: Balance between data fidelity and file size

### 5. Module Organization
```python
# Typical task structure:
# Dynamics Process:
#   - Dynamics Task (10-100 Hz)
#   - Sensor Task (10-100 Hz)
#   - Environment Task (1-10 Hz)
# FSW Process:
#   - Guidance Task (0.5-2 Hz)
#   - Control Task (1-10 Hz)
#   - Navigation Task (1-10 Hz)
```

### 6. Debugging Tips
- Use `ModelTag` for all modules (helpful in error messages)
- Check message connections with print statements
- Validate initial conditions before long runs
- Use smaller time steps if seeing instabilities
- Plot intermediate results to catch issues early

### 7. Performance Optimization
- Minimize logging frequency for long simulations
- Use appropriate integrators (RK4 for most cases)
- Disable unnecessary modules
- Profile code to identify bottlenecks

### 8. Thruster Configuration
```python
# For lunar landing, typical setup:
# 1. Main engines (3-6): High thrust, low rate
# 2. Mid-body thrusters (8-16): Medium thrust, attitude control
# 3. RCS thrusters (12-24): Low thrust, fine control
```

### 9. Fuel Mass Tracking
```python
# Total vehicle mass = dry mass + sum(fuel masses)
# Inertia changes with fuel depletion
# For Starship HLS:
#   - Full propellant: ~1,200,000 kg
#   - Dry mass: ~105,000 kg (structure + payload)
#   - Total launch mass: ~1,305,000 kg
```

### 10. Lunar Landing Phases
1. **De-orbit Burn**: Reduce orbital velocity
2. **Braking Phase**: High-thrust descent (1-10 km altitude)
3. **Approach Phase**: Medium thrust, gravity turn (100m - 1km)
4. **Terminal Descent**: Low thrust, vertical descent (0-100m)
5. **Touchdown**: Final deceleration and landing

---

## Example: Complete Lunar Landing Setup

```python
# 1. Create simulation
scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess("simProcess")
simulationTimeStep = macros.sec2nano(0.1)
dynProcess.addTask(scSim.CreateNewTask("dynTask", simulationTimeStep))

# 2. Create spacecraft
lander = spacecraft.Spacecraft()
lander.ModelTag = "Lander"
lander.hub.mHub = 105000.0  # Dry mass
lander.hub.r_CN_NInit = np.array([0., 0., 1500.0])  # 1.5 km altitude
lander.hub.v_CN_NInit = np.array([0., 0., -10.0])   # Descending
scSim.AddModelToTask("dynTask", lander)

# 3. Add gravity
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True
gravFactory.addBodiesTo(lander)

# 4. Add fuel tanks
ch4Tank = fuelTank.FuelTank()
ch4Tank.ModelTag = "CH4_Tank"
ch4TankModel = fuelTank.FuelTankModelConstantVolume()
ch4TankModel.propMassInit = 260869.565
ch4Tank.setTankModel(ch4TankModel)
lander.addStateEffector(ch4Tank)
scSim.AddModelToTask("dynTask", ch4Tank)

# 5. Add thrusters
primaryEff = thrusterDynamicEffector.ThrusterDynamicEffector()
primaryEff.ModelTag = "MainEngines"
for i in range(3):
    thr = thrusterDynamicEffector.THRSimConfig()
    thr.thrLoc_B = np.array([3.5, 0., -24.5])  # Aft location
    thr.thrDir_B = np.array([0., 0., 1.])       # Pointing up
    thr.MaxThrust = 2500000.0                   # 2.5 MN
    primaryEff.addThruster(thr)
lander.addDynamicEffector(primaryEff)
scSim.AddModelToTask("dynTask", primaryEff)

# 6. Add sensors
imu = imuSensor.ImuSensor()
imu.ModelTag = "IMU"
imu.scStateInMsg.subscribeTo(lander.scStateOutMsg)
scSim.AddModelToTask("dynTask", imu)

# 7. Setup logging
scLog = lander.scStateOutMsg.recorder(simulationTimeStep)
scSim.AddModelToTask("dynTask", scLog)

# 8. Run simulation
scSim.InitializeSimulation()
scSim.ConfigureStopTime(macros.sec2nano(60.0))
scSim.ExecuteSimulation()

# 9. Plot results
timeData = scLog.times() * macros.NANO2SEC
altitude = scLog.r_BN_N[:, 2]
plt.plot(timeData, altitude)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.show()
```

---

## Additional Resources

### Documentation
- Basilisk Docs: `basilisk/docs/source/`
- Examples: `basilisk/examples/`
- Unit Tests: `basilisk/src/*/*/test/`

### Key Example Scenarios
- `scenarioBasicOrbit.py` - Basic orbital dynamics
- `scenarioAttitudeFeedback2T_TH.py` - Thruster control
- `scenarioDragDeorbit.py` - Atmospheric effects
- `scenarioFuelSlosh.py` - Fuel tank dynamics

### Module Documentation Pattern
Each module folder contains:
- `*.rst` - Documentation file
- `_UnitTest/` - Test scripts
- `*.h` / `*.cpp` - C++ implementation
- Python wrapper for interface

---

## Lunar-Specific Considerations

### Moon Properties
- Radius: 1,737,400 m
- μ (GM): 4.9028 × 10¹² m³/s²
- Surface gravity: 1.625 m/s²
- No atmosphere (exosphere density ~10⁻¹⁵ kg/m³)
- No magnetic field

### Landing Site Coordinates
Use planetographic coordinates:
- Latitude: degrees
- Longitude: degrees
- Altitude: meters above reference sphere

### SPICE Integration
For high-fidelity ephemeris:
```python
from Basilisk.utilities import spiceInterface
spice = spiceInterface.SpiceInterface()
spice.loadSpiceKernel("de430.bsp", "path/to/kernel")
```

### Terrain Avoidance
- Implement custom height map reader
- Use hill shading analysis
- Integrate with navigation filter

---

## Version Compatibility
This reference is based on Basilisk version 2.x architecture.
- Messaging: Modern `messaging` module (not deprecated `cMsgCInterfacePy`)
- Python 3.x required
- NumPy for array operations
- Matplotlib for visualization

---

## Quick Reference Tables

### Common Module Import Paths

| Module Type | Import Path | Purpose |
|------------|-------------|---------|
| Spacecraft | `Basilisk.simulation.spacecraft` | Rigid body dynamics |
| Thrusters | `Basilisk.simulation.thrusterDynamicEffector` | Thruster forces |
| Fuel Tanks | `Basilisk.simulation.fuelTank` | Propellant management |
| IMU | `Basilisk.simulation.imuSensor` | Inertial measurements |
| Drag | `Basilisk.simulation.dragDynamicEffector` | Atmospheric drag |
| Atmosphere | `Basilisk.simulation.exponentialAtmosphere` | Atmosphere model |
| Gravity | `Basilisk.utilities.simIncludeGravBody` | Celestial bodies |
| MRP Control | `Basilisk.fswAlgorithms.mrpFeedback` | Attitude control |
| Thr Mapping | `Basilisk.fswAlgorithms.thrForceMapping` | Torque to thrust |

### Message Payload Types

| Payload Type | Contains | Used For |
|--------------|----------|----------|
| `SCStatesMsgPayload` | Position, velocity, attitude, rates | Spacecraft state |
| `THRArrayOnTimeCmdMsgPayload` | Thruster on-time requests | Thruster commands |
| `AttGuidMsgPayload` | Reference attitude and rates | Guidance output |
| `VehicleConfigMsgPayload` | Mass, inertia, COM | Vehicle properties |
| `FuelTankMsgPayload` | Fuel mass, depletion rate | Tank status |
| `IMUSensorMsgPayload` | Acceleration, angular rates | IMU output |

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-26  
**For**: Starship HLS Lunar Landing Simulation
