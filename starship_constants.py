"""
Starship HLS (Human Landing System) Configuration Constants

This module contains all physical and engineering constants for the Starship HLS
lunar lander used in the Basilisk simulation environment.

STARSHIP HLS CONFIGURATION:
- Height: 50 m
- Diameter: 9 m
- Initial mass: 1,305,000 kg (dry: 85,000 kg, payload: 20,000 kg, propellant: 1,200,000 kg)
- Propulsion: 3 Raptor engines (2,500 kN each @ vacuum)
- Propellant: CH4/LOX (O/F ratio 3.6)
"""

import numpy as np

# ==============================================================================
# CELESTIAL BODY CONSTANTS
# ==============================================================================

# Moon radius (from Basilisk's gravBodyFactory.createMoon())
# CRITICAL: Basilisk uses Moon-centered inertial coordinates
# All positions in simulation are: z = MOON_RADIUS + terrain_height + altitude
MOON_RADIUS = 1737400.0  # meters (mean radius of the Moon)

# ==============================================================================
# MASS PROPERTIES
# ==============================================================================

# Dry mass (kg) - includes structure and payload
DRY_MASS = 85000.0  # kg - Starship HLS dry mass
PAYLOAD_MASS = 20000.0  # kg - Payload mass
HUB_MASS = DRY_MASS + PAYLOAD_MASS  # 105,000 kg total

# Initial propellant masses (kg)
TOTAL_PROPELLANT_MASS = 1200000.0  # kg
CH4_INITIAL_MASS = 260869.565  # kg - Methane mass
LOX_INITIAL_MASS = 939130.435  # kg - Liquid oxygen mass
TOTAL_INITIAL_MASS = HUB_MASS + TOTAL_PROPELLANT_MASS  # 1,305,000 kg

# Mixture ratio (O/F)
MIXTURE_RATIO = 3.6  # LOX to CH4 mass ratio

# Inertia tensor (kg·m²) - full propellant configuration
# NOTE: These values represent the spacecraft with FULL propellant
# Inertia will change as fuel depletes during simulation
INERTIA_TENSOR_FULL = np.array([
    [231513125.0, 0.0, 0.0],
    [0.0, 231513125.0, 0.0],
    [0.0, 0.0, 14276250.0]
])

# Center of mass offset (body frame)
CENTER_OF_MASS_OFFSET = np.zeros(3)  # At vehicle geometric center

# ==============================================================================
# FUEL TANK CONFIGURATION
# ==============================================================================

# CH4 Tank (Methane)
CH4_TANK_VOLUME = 617.005  # m³
CH4_TANK_DENSITY = 422.8  # kg/m³
CH4_TANK_RADIUS = (3.0 * CH4_TANK_VOLUME / (4.0 * np.pi)) ** (1.0/3.0)  # m
CH4_TANK_POSITION = np.array([[0.0], [0.0], [-10.0]])  # Body frame position (aft section)

# LOX Tank (Liquid Oxygen)
LOX_TANK_VOLUME = 823.077  # m³
LOX_TANK_DENSITY = 1141.0  # kg/m³
LOX_TANK_RADIUS = (3.0 * LOX_TANK_VOLUME / (4.0 * np.pi)) ** (1.0/3.0)  # m
LOX_TANK_POSITION = np.array([[0.0], [0.0], [-5.0]])  # Body frame position (aft section)

# Propellant cylinder height (90% of vehicle)
PROPELLANT_CYLINDER_HEIGHT = 45.0  # m (90% of 50m vehicle height)

# ==============================================================================
# PROPULSION SYSTEM
# ==============================================================================

# Engine performance
VACUUM_ISP = 375.0  # seconds - Raptor vacuum specific impulse
STANDARD_GRAVITY = 9.80665  # m/s² - Standard gravitational acceleration
MAX_THRUST_PER_ENGINE = 2500000.0  # N - Maximum thrust per Raptor engine

# Mass flow calculations (per engine at 100% throttle)
PER_ENGINE_MASS_FLOW = MAX_THRUST_PER_ENGINE / (VACUUM_ISP * STANDARD_GRAVITY)  # kg/s
CH4_FLOW_PER_ENGINE = PER_ENGINE_MASS_FLOW / (1.0 + MIXTURE_RATIO)  # kg/s
LOX_FLOW_PER_ENGINE = PER_ENGINE_MASS_FLOW * MIXTURE_RATIO / (1.0 + MIXTURE_RATIO)  # kg/s

# Primary engine positions (body frame) - 3 Raptor engines
# Triangle configuration at aft section (z = -24.5 m)
PRIMARY_ENGINE_POSITIONS = [
    np.array([3.500, 0.000, -24.500]),      # Engine A1
    np.array([-1.750, 3.031, -24.500]),     # Engine A2
    np.array([-1.750, -3.031, -24.500])     # Engine A3
]

# Primary engine thrust direction (body frame) - all pointing +Z (up)
PRIMARY_ENGINE_DIRECTION = np.array([0., 0., 1.])

# Engine indexing
PRIMARY_ENGINE_COUNT = 3
PRIMARY_ENGINE_START_INDEX = 0

# ==============================================================================
# ATTITUDE CONTROL THRUSTERS
# ==============================================================================

# Mid-body thrusters (12 thrusters for attitude control)
# Positioned at radius 4.0 m, z = 0.0 m (mid-height)
MIDBODY_THRUST = 20000.0  # N - Thrust per mid-body thruster
MIDBODY_RADIUS = 4.0  # m - Radial distance from centerline
MIDBODY_Z_POSITION = 0.0  # m - Mid-height of vehicle

MIDBODY_THRUSTER_POSITIONS = [
    np.array([4.000, 0.000, 0.000]),    # M1
    np.array([3.464, 2.000, 0.000]),    # M2
    np.array([2.000, 3.464, 0.000]),    # M3
    np.array([0.000, 4.000, 0.000]),    # M4
    np.array([-2.000, 3.464, 0.000]),   # M5
    np.array([-3.464, 2.000, 0.000]),   # M6
    np.array([-4.000, 0.000, 0.000]),   # M7
    np.array([-3.464, -2.000, 0.000]),  # M8
    np.array([-2.000, -3.464, 0.000]),  # M9
    np.array([0.000, -4.000, 0.000]),   # M10
    np.array([2.000, -3.464, 0.000]),   # M11
    np.array([3.464, -2.000, 0.000])    # M12
]

MIDBODY_THRUSTER_COUNT = 12
MIDBODY_THRUSTER_START_INDEX = PRIMARY_ENGINE_COUNT

# RCS thrusters (24 thrusters: 12 at top ring, 12 at bottom ring)
# 2,000 N each, positioned at radius 4.2 m
RCS_THRUST = 2000.0  # N - Thrust per RCS thruster
RCS_RADIUS = 4.2  # m - Radial distance from centerline
RCS_TOP_Z = 22.5  # m - Top ring height
RCS_BOTTOM_Z = -22.5  # m - Bottom ring height

RCS_THRUSTER_POSITIONS = [
    # Ring 1 (top, z = 22.5 m)
    np.array([4.200, 0.000, 22.500]),    # R1
    np.array([3.637, 2.100, 22.500]),    # R2
    np.array([2.100, 3.637, 22.500]),    # R3
    np.array([0.000, 4.200, 22.500]),    # R4
    np.array([-2.100, 3.637, 22.500]),   # R5
    np.array([-3.637, 2.100, 22.500]),   # R6
    np.array([-4.200, 0.000, 22.500]),   # R7
    np.array([-3.637, -2.100, 22.500]),  # R8
    np.array([-2.100, -3.637, 22.500]),  # R9
    np.array([0.000, -4.200, 22.500]),   # R10
    np.array([2.100, -3.637, 22.500]),   # R11
    np.array([3.637, -2.100, 22.500]),   # R12
    # Ring 2 (bottom, z = -22.5 m)
    np.array([4.200, 0.000, -22.500]),   # R13
    np.array([3.637, 2.100, -22.500]),   # R14
    np.array([2.100, 3.637, -22.500]),   # R15
    np.array([0.000, 4.200, -22.500]),   # R16
    np.array([-2.100, 3.637, -22.500]),  # R17
    np.array([-3.637, 2.100, -22.500]),  # R18
    np.array([-4.200, 0.000, -22.500]),  # R19
    np.array([-3.637, -2.100, -22.500]), # R20
    np.array([-2.100, -3.637, -22.500]), # R21
    np.array([0.000, -4.200, -22.500]),  # R22
    np.array([2.100, -3.637, -22.500]),  # R23
    np.array([3.637, -2.100, -22.500])   # R24
]

RCS_THRUSTER_COUNT = 24
RCS_THRUSTER_START_INDEX = MIDBODY_THRUSTER_START_INDEX + MIDBODY_THRUSTER_COUNT

# Total thruster count
TOTAL_THRUSTER_COUNT = PRIMARY_ENGINE_COUNT + MIDBODY_THRUSTER_COUNT + RCS_THRUSTER_COUNT  # 39

# ==============================================================================
# VEHICLE GEOMETRY
# ==============================================================================

VEHICLE_HEIGHT = 50.0  # m - Total vehicle height
VEHICLE_DIAMETER = 9.0  # m - Vehicle diameter
VEHICLE_RADIUS = VEHICLE_DIAMETER / 2.0  # 4.5 m

# ==============================================================================
# LANDING GEAR CONFIGURATION
# ==============================================================================

# Landing leg positions (body frame)
# 4 legs positioned symmetrically around vehicle base
LEG_RADIUS = 4.5  # m - Radial distance from centerline
LEG_Z_ATTACH = -24.0  # m - Attachment point height
LEG_LENGTH = 3.0  # m - Extended leg length
LEG_GROUND_Z = LEG_Z_ATTACH - LEG_LENGTH  # -27.0 m - Ground contact point when extended

LANDING_LEG_POSITIONS = [
    np.array([LEG_RADIUS, 0.0, LEG_GROUND_Z]),           # Leg 1 (+X)
    np.array([0.0, LEG_RADIUS, LEG_GROUND_Z]),           # Leg 2 (+Y)
    np.array([-LEG_RADIUS, 0.0, LEG_GROUND_Z]),          # Leg 3 (-X)
    np.array([0.0, -LEG_RADIUS, LEG_GROUND_Z])           # Leg 4 (-Y)
]

LANDING_LEG_COUNT = 4

# ==============================================================================
# AERODYNAMIC PROPERTIES (for atmospheric flight phases)
# ==============================================================================

# Reference area for drag calculations
REFERENCE_AREA = np.pi * VEHICLE_RADIUS**2  # m² - Cross-sectional area

# Drag coefficient (approximate for cylindrical body)
DRAG_COEFFICIENT = 0.82  # Dimensionless - typical for cylinder

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_midbody_thruster_direction(position):
    """
    Calculate the thrust direction for a mid-body thruster.
    Direction is radially outward for attitude control.
    
    Args:
        position: np.array([x, y, z]) position in body frame
        
    Returns:
        np.array([x, y, z]) normalized direction vector
    """
    direction = np.array([position[0], position[1], 0.0])
    return direction / np.linalg.norm(direction)

def get_rcs_thruster_direction(position):
    """
    Calculate the thrust direction for an RCS thruster.
    Direction is radially outward for attitude control.
    
    Args:
        position: np.array([x, y, z]) position in body frame
        
    Returns:
        np.array([x, y, z]) normalized direction vector
    """
    direction = np.array([position[0], position[1], 0.0])
    return direction / np.linalg.norm(direction)

def print_configuration_summary():
    """Print a summary of the Starship HLS configuration."""
    print("="*70)
    print("STARSHIP HLS CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nMASS PROPERTIES:")
    print(f"  Dry mass:              {DRY_MASS:,.0f} kg")
    print(f"  Payload mass:          {PAYLOAD_MASS:,.0f} kg")
    print(f"  Total dry + payload:   {HUB_MASS:,.0f} kg")
    print(f"  CH4 propellant:        {CH4_INITIAL_MASS:,.2f} kg")
    print(f"  LOX propellant:        {LOX_INITIAL_MASS:,.2f} kg")
    print(f"  Total propellant:      {TOTAL_PROPELLANT_MASS:,.0f} kg")
    print(f"  Total initial mass:    {TOTAL_INITIAL_MASS:,.0f} kg")
    
    print(f"\nPROPULSION:")
    print(f"  Primary engines:       {PRIMARY_ENGINE_COUNT} x Raptor")
    print(f"  Max thrust per engine: {MAX_THRUST_PER_ENGINE:,.0f} N")
    print(f"  Total max thrust:      {MAX_THRUST_PER_ENGINE * PRIMARY_ENGINE_COUNT:,.0f} N")
    print(f"  Vacuum Isp:            {VACUUM_ISP:.0f} s")
    print(f"  Mass flow (per engine): {PER_ENGINE_MASS_FLOW:.2f} kg/s")
    print(f"    - CH4 flow:          {CH4_FLOW_PER_ENGINE:.2f} kg/s")
    print(f"    - LOX flow:          {LOX_FLOW_PER_ENGINE:.2f} kg/s")
    print(f"  Mixture ratio (O/F):   {MIXTURE_RATIO}")
    
    print(f"\nATTITUDE CONTROL:")
    print(f"  Mid-body thrusters:    {MIDBODY_THRUSTER_COUNT} x {MIDBODY_THRUST:,.0f} N")
    print(f"  RCS thrusters:         {RCS_THRUSTER_COUNT} x {RCS_THRUST:,.0f} N")
    print(f"  Total thrusters:       {TOTAL_THRUSTER_COUNT}")
    
    print(f"\nGEOMETRY:")
    print(f"  Height:                {VEHICLE_HEIGHT:.1f} m")
    print(f"  Diameter:              {VEHICLE_DIAMETER:.1f} m")
    print(f"  Landing legs:          {LANDING_LEG_COUNT} x {LEG_LENGTH:.1f} m")
    print("="*70)


if __name__ == "__main__":
    # Print configuration when run directly
    print_configuration_summary()
