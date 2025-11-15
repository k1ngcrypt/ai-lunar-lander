"""
Isolate whether thruster effectors cause divergence
Tests identical spacecraft with and without thruster effectors
"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.simulation import spacecraft, thrusterStateEffector, fuelTank
import starship_constants as SC
import numpy as np

def test_without_thrusters():
    """Test spacecraft with NO thruster effectors (just gravity)"""
    print("\n" + "="*70)
    print("TEST 1: Spacecraft WITHOUT thruster effectors")
    print("="*70)
    
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynamics")
    dt = 0.01
    dynProcess.addTask(scSim.CreateNewTask("dynTask", macros.sec2nano(dt)))
    
    # Create spacecraft
    lander = spacecraft.Spacecraft()
    lander.ModelTag = "TestLander"
    lander.hub.mHub = SC.HUB_MASS
    lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
    lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
    lander.hub.r_CN_NInit = np.array([0., 0., 80.0])
    lander.hub.v_CN_NInit = np.array([0., 0., -4.0])
    lander.hub.sigma_BNInit = np.zeros(3)
    lander.hub.omega_BN_BInit = np.zeros(3)
    
    scSim.AddModelToTask("dynTask", lander)
    
    # Add lunar gravity (using gravFactory pattern from ScenarioLunarLanderStarter)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    moon = gravFactory.createMoon()
    moon.isCentralBody = True
    gravFactory.addBodiesTo(lander)
    
    # Initialize
    scSim.InitializeSimulation()
    
    print(f"Initial: alt=80.0m, vel_z=-4.0 m/s")
    print(f"Mass: {lander.hub.mHub:.0f} kg")
    inertia_val = SC.INERTIA_TENSOR_FULL[0,0]
    print(f"Inertia: {inertia_val:.1e} kg·m²")
    
    # Step simulation
    scSim.ConfigureStopTime(macros.sec2nano(dt))
    scSim.ExecuteSimulation()
    
    # Read state
    pos = lander.scStateOutMsg.read().r_BN_N
    vel = lander.scStateOutMsg.read().v_BN_N
    
    print(f"After {dt}s step:")
    print(f"  alt={pos[2]:.2f}m, vel_z={vel[2]:.2f} m/s")
    
    # Check for divergence
    success = abs(pos[2] - 80) < 10
    if success:
        print("✓ NO DIVERGENCE - Simulation stable")
    else:
        print(f"✗ DIVERGENCE DETECTED! (expected ~80m, got {pos[2]:.2f}m)")
    
    return success

def test_with_thrusters_no_firing():
    """Test spacecraft WITH thruster effectors but zero throttle"""
    print("\n" + "="*70)
    print("TEST 2: Spacecraft WITH thruster effectors (0% throttle)")
    print("="*70)
    
    scSim = SimulationBaseClass.SimBaseClass()
    dynProcess = scSim.CreateNewProcess("dynamics")
    dt = 0.01
    dynProcess.addTask(scSim.CreateNewTask("dynTask", macros.sec2nano(dt)))
    
    # Create spacecraft (identical to test 1)
    lander = spacecraft.Spacecraft()
    lander.ModelTag = "TestLander"
    lander.hub.mHub = SC.HUB_MASS
    lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
    lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
    lander.hub.r_CN_NInit = np.array([0., 0., 80.0])
    lander.hub.v_CN_NInit = np.array([0., 0., -4.0])
    lander.hub.sigma_BNInit = np.zeros(3)
    lander.hub.omega_BN_BInit = np.zeros(3)
    
    scSim.AddModelToTask("dynTask", lander)
    
    # Add fuel tanks (NOT connected to thrusters)
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
    scSim.AddModelToTask("dynTask", ch4Tank)
    
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
    scSim.AddModelToTask("dynTask", loxTank)
    
    # Add thruster effectors
    primaryEff = thrusterStateEffector.ThrusterStateEffector()
    primaryEff.ModelTag = "PrimaryThrusters"
    scSim.AddModelToTask("dynTask", primaryEff)
    lander.addStateEffector(primaryEff)
    
    # Add 3 primary thrusters
    for pos in SC.PRIMARY_ENGINE_POSITIONS:
        thrConfig = thrusterStateEffector.THRSimConfig()
        thrConfig.thrLoc_B = np.array(pos, dtype=float)
        thrConfig.thrDir_B = SC.PRIMARY_ENGINE_DIRECTION
        thrConfig.MaxThrust = SC.MAX_THRUST_PER_ENGINE
        thrConfig.steadyIsp = SC.VACUUM_ISP
        primaryEff.addThruster(thrConfig, lander.scStateOutMsg)
    
    # DO NOT connect fuel tanks (this was already disabled in ScenarioLunarLanderStarter.py)
    # ch4Tank.addThrusterSet(primaryEff) - OMITTED
    # loxTank.addThrusterSet(primaryEff) - OMITTED
    
    # Add lunar gravity (using gravFactory pattern from ScenarioLunarLanderStarter)
    gravFactory = simIncludeGravBody.gravBodyFactory()
    moon = gravFactory.createMoon()
    moon.isCentralBody = True
    gravFactory.addBodiesTo(lander)
    
    # Initialize
    scSim.InitializeSimulation()
    
    print(f"Initial: alt=80.0m, vel_z=-4.0 m/s")
    print(f"Mass: {lander.hub.mHub:.0f} kg")
    inertia_val = SC.INERTIA_TENSOR_FULL[0,0]
    print(f"Inertia: {inertia_val:.1e} kg·m²")
    print(f"Thrusters: {SC.PRIMARY_ENGINE_COUNT} engines attached (default 0% throttle)")
    
    # Don't set thruster commands - they default to zero
    # Just let them exist but not fire
    
    # Step simulation
    scSim.ConfigureStopTime(macros.sec2nano(dt))
    scSim.ExecuteSimulation()
    
    # Read state
    pos = lander.scStateOutMsg.read().r_BN_N
    vel = lander.scStateOutMsg.read().v_BN_N
    
    print(f"After {dt}s step:")
    print(f"  alt={pos[2]:.2f}m, vel_z={vel[2]:.2f} m/s")
    
    # Check for divergence
    success = abs(pos[2] - 80) < 10
    if success:
        print("✓ NO DIVERGENCE - Simulation stable")
    else:
        print(f"✗ DIVERGENCE DETECTED! (expected ~80m, got {pos[2]:.2f}m)")
    
    return success

if __name__ == "__main__":
    # Run both tests
    test1_pass = test_without_thrusters()
    test2_pass = test_with_thrusters_no_firing()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (no thrusters):   {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (with thrusters): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    
    if test1_pass and not test2_pass:
        print("\n🔍 CONCLUSION: Thruster effectors ARE the problem!")
        print("   The simulation is stable without thrusters but diverges with them,")
        print("   even when thrusters are set to 0% throttle.")
    elif test1_pass and test2_pass:
        print("\n✓ Both tests passed - problem is elsewhere")
    else:
        print("\n⚠ Test 1 failed - problem is more fundamental than thrusters")
