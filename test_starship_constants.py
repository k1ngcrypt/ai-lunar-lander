"""
Unit tests for starship_constants.py
Tests Starship HLS configuration constants and derived calculations.
"""

import unittest
import numpy as np

# Import module under test
import starship_constants as SC


class TestMassProperties(unittest.TestCase):
    """Test mass property constants"""
    
    def test_total_initial_mass(self):
        """Test that total initial mass is sum of components"""
        expected = SC.HUB_MASS + SC.TOTAL_PROPELLANT_MASS
        self.assertEqual(SC.TOTAL_INITIAL_MASS, expected)
    
    def test_hub_mass(self):
        """Test that hub mass is sum of dry mass and payload"""
        expected = SC.DRY_MASS + SC.PAYLOAD_MASS
        self.assertEqual(SC.HUB_MASS, expected)
    
    def test_propellant_mass(self):
        """Test that propellant mass is sum of CH4 and LOX"""
        total_prop = SC.CH4_INITIAL_MASS + SC.LOX_INITIAL_MASS
        # Should be close to TOTAL_PROPELLANT_MASS (within rounding)
        self.assertAlmostEqual(total_prop, SC.TOTAL_PROPELLANT_MASS, places=2)
    
    def test_mixture_ratio(self):
        """Test that mixture ratio matches LOX/CH4 mass ratio"""
        calculated_ratio = SC.LOX_INITIAL_MASS / SC.CH4_INITIAL_MASS
        self.assertAlmostEqual(calculated_ratio, SC.MIXTURE_RATIO, places=2)
    
    def test_mass_values_positive(self):
        """Test that all masses are positive"""
        self.assertGreater(SC.DRY_MASS, 0)
        self.assertGreater(SC.PAYLOAD_MASS, 0)
        self.assertGreater(SC.HUB_MASS, 0)
        self.assertGreater(SC.CH4_INITIAL_MASS, 0)
        self.assertGreater(SC.LOX_INITIAL_MASS, 0)
        self.assertGreater(SC.TOTAL_PROPELLANT_MASS, 0)
        self.assertGreater(SC.TOTAL_INITIAL_MASS, 0)
    
    def test_realistic_mass_range(self):
        """Test that masses are in realistic range for Starship HLS"""
        # Total mass should be around 1.3M kg
        self.assertGreater(SC.TOTAL_INITIAL_MASS, 1_000_000)
        self.assertLess(SC.TOTAL_INITIAL_MASS, 2_000_000)
        
        # Dry mass should be reasonable (50-150k kg)
        self.assertGreater(SC.DRY_MASS, 50_000)
        self.assertLess(SC.DRY_MASS, 150_000)


class TestInertiaProperties(unittest.TestCase):
    """Test inertia tensor properties"""
    
    def test_inertia_tensor_shape(self):
        """Test inertia tensor is 3x3"""
        self.assertEqual(SC.INERTIA_TENSOR_FULL.shape, (3, 3))
    
    def test_inertia_tensor_symmetric(self):
        """Test inertia tensor is symmetric"""
        np.testing.assert_array_equal(
            SC.INERTIA_TENSOR_FULL,
            SC.INERTIA_TENSOR_FULL.T
        )
    
    def test_inertia_diagonal_positive(self):
        """Test that diagonal elements are positive"""
        for i in range(3):
            self.assertGreater(SC.INERTIA_TENSOR_FULL[i, i], 0)
    
    def test_inertia_diagonal_dominant(self):
        """Test diagonal dominance (typical for spacecraft)"""
        # For a cylinder-like shape, Ixx ~ Iyy >> Izz
        Ixx = SC.INERTIA_TENSOR_FULL[0, 0]
        Iyy = SC.INERTIA_TENSOR_FULL[1, 1]
        Izz = SC.INERTIA_TENSOR_FULL[2, 2]
        
        # Ixx and Iyy should be similar (symmetric about z-axis)
        ratio = Ixx / Iyy
        self.assertAlmostEqual(ratio, 1.0, places=2)
        
        # Ixx, Iyy should be much larger than Izz (long cylinder)
        self.assertGreater(Ixx, Izz * 10)
    
    def test_center_of_mass_offset(self):
        """Test center of mass offset is 3D vector"""
        self.assertEqual(len(SC.CENTER_OF_MASS_OFFSET), 3)


class TestFuelTankConfiguration(unittest.TestCase):
    """Test fuel tank configuration"""
    
    def test_tank_volumes_positive(self):
        """Test that tank volumes are positive"""
        self.assertGreater(SC.CH4_TANK_VOLUME, 0)
        self.assertGreater(SC.LOX_TANK_VOLUME, 0)
    
    def test_tank_densities_realistic(self):
        """Test that densities are realistic for CH4 and LOX"""
        # CH4 density should be around 400-450 kg/m³
        self.assertGreater(SC.CH4_TANK_DENSITY, 400)
        self.assertLess(SC.CH4_TANK_DENSITY, 500)
        
        # LOX density should be around 1100-1200 kg/m³
        self.assertGreater(SC.LOX_TANK_DENSITY, 1000)
        self.assertLess(SC.LOX_TANK_DENSITY, 1300)
    
    def test_tank_radius_calculation(self):
        """Test that tank radius is calculated correctly from volume"""
        # For spherical tanks: V = (4/3) * π * r³
        # The formula in constants uses: r = (3V / (4π))^(1/3)
        calculated_ch4_radius = (3.0 * SC.CH4_TANK_VOLUME / (4.0 * np.pi)) ** (1.0/3.0)
        self.assertAlmostEqual(SC.CH4_TANK_RADIUS, calculated_ch4_radius, places=5)
        
        calculated_lox_radius = (3.0 * SC.LOX_TANK_VOLUME / (4.0 * np.pi)) ** (1.0/3.0)
        self.assertAlmostEqual(SC.LOX_TANK_RADIUS, calculated_lox_radius, places=5)
    
    def test_tank_positions_shape(self):
        """Test that tank positions are 3D vectors in correct shape"""
        self.assertEqual(SC.CH4_TANK_POSITION.shape, (3, 1))
        self.assertEqual(SC.LOX_TANK_POSITION.shape, (3, 1))
    
    def test_propellant_cylinder_height_realistic(self):
        """Test that propellant cylinder height is reasonable"""
        # Should be significant portion of vehicle (but not all of it)
        self.assertGreater(SC.PROPELLANT_CYLINDER_HEIGHT, 30)
        self.assertLess(SC.PROPELLANT_CYLINDER_HEIGHT, 60)


class TestPropulsionSystem(unittest.TestCase):
    """Test propulsion system constants and calculations"""
    
    def test_isp_positive(self):
        """Test that specific impulse is positive"""
        self.assertGreater(SC.VACUUM_ISP, 0)
    
    def test_isp_realistic_for_raptor(self):
        """Test that ISP is realistic for Raptor vacuum engine"""
        # Raptor vacuum ISP should be around 350-380 seconds
        self.assertGreater(SC.VACUUM_ISP, 350)
        self.assertLess(SC.VACUUM_ISP, 400)
    
    def test_standard_gravity(self):
        """Test standard gravity constant"""
        # Should be ~9.80665 m/s²
        self.assertAlmostEqual(SC.STANDARD_GRAVITY, 9.80665, places=5)
    
    def test_max_thrust_positive(self):
        """Test that max thrust per engine is positive"""
        self.assertGreater(SC.MAX_THRUST_PER_ENGINE, 0)
    
    def test_max_thrust_realistic_for_raptor(self):
        """Test that thrust is realistic for Raptor engine"""
        # Raptor vacuum should be 2-3 MN
        self.assertGreater(SC.MAX_THRUST_PER_ENGINE, 2_000_000)
        self.assertLess(SC.MAX_THRUST_PER_ENGINE, 3_000_000)
    
    def test_mass_flow_calculation(self):
        """Test that mass flow is calculated correctly from thrust equation"""
        # Thrust = mdot * Isp * g
        calculated_flow = SC.MAX_THRUST_PER_ENGINE / (SC.VACUUM_ISP * SC.STANDARD_GRAVITY)
        self.assertAlmostEqual(SC.PER_ENGINE_MASS_FLOW, calculated_flow, places=3)
    
    def test_propellant_flow_split(self):
        """Test that CH4 and LOX flow rates match mixture ratio"""
        # Total flow should equal sum of CH4 and LOX
        total_flow = SC.CH4_FLOW_PER_ENGINE + SC.LOX_FLOW_PER_ENGINE
        self.assertAlmostEqual(total_flow, SC.PER_ENGINE_MASS_FLOW, places=3)
        
        # Ratio should match mixture ratio
        ratio = SC.LOX_FLOW_PER_ENGINE / SC.CH4_FLOW_PER_ENGINE
        self.assertAlmostEqual(ratio, SC.MIXTURE_RATIO, places=2)
    
    def test_flow_rates_positive(self):
        """Test that all flow rates are positive"""
        self.assertGreater(SC.PER_ENGINE_MASS_FLOW, 0)
        self.assertGreater(SC.CH4_FLOW_PER_ENGINE, 0)
        self.assertGreater(SC.LOX_FLOW_PER_ENGINE, 0)


class TestEngineConfiguration(unittest.TestCase):
    """Test engine positioning and configuration"""
    
    def test_engine_positions_exist(self):
        """Test that engine positions are defined"""
        # Check if PRIMARY_ENGINE_POSITIONS is defined
        self.assertTrue(hasattr(SC, 'PRIMARY_ENGINE_POSITIONS'))
    
    def test_num_engines(self):
        """Test that PRIMARY_ENGINE_COUNT is defined and equals 3"""
        self.assertTrue(hasattr(SC, 'PRIMARY_ENGINE_COUNT'))
        self.assertEqual(SC.PRIMARY_ENGINE_COUNT, 3)


if __name__ == '__main__':
    unittest.main()
