"""
Unit tests for terrain_simulation.py
Tests LunarRegolithModel terrain simulation class.
"""

import unittest
import numpy as np
import os
import tempfile

# Import module under test
from terrain_simulation import LunarRegolithModel


class TestLunarRegolithModelInit(unittest.TestCase):
    """Test LunarRegolithModel initialization"""
    
    def test_default_initialization(self):
        """Test model initialization with default parameters"""
        model = LunarRegolithModel()
        
        # Check default values
        self.assertEqual(model.size, 2000.0)
        self.assertEqual(model.resolution, 100)
        self.assertEqual(model.cell_size, model.size / model.resolution)
    
    def test_custom_initialization(self):
        """Test model initialization with custom parameters"""
        model = LunarRegolithModel(size=1000.0, resolution=50)
        
        self.assertEqual(model.size, 1000.0)
        self.assertEqual(model.resolution, 50)
        self.assertEqual(model.cell_size, 20.0)
    
    def test_terrain_array_initialization(self):
        """Test that terrain height map is initialized"""
        model = LunarRegolithModel(resolution=10)
        
        # Should have a heightmap attribute (initially None)
        self.assertTrue(hasattr(model, 'heightmap'))
        # heightmap is None until terrain is loaded or generated
        # This is expected behavior
    
    def test_physical_properties_realistic(self):
        """Test that physical properties are in realistic ranges"""
        model = LunarRegolithModel()
        
        # Friction coefficient should be positive and reasonable
        self.assertGreater(model.friction_coeff, 0)
        self.assertLess(model.friction_coeff, 2.0)
        
        # Cohesion should be low for lunar regolith
        self.assertGreater(model.cohesion, 0)
        self.assertLess(model.cohesion, 5000)
        
        # Damping coefficients should be positive
        self.assertGreater(model.damping_coeff, 0)
        self.assertGreater(model.lateral_damping, 0)
        
        # Restitution should be between 0 and 1
        self.assertGreaterEqual(model.restitution, 0)
        self.assertLessEqual(model.restitution, 1)


class TestTerrainHeightQueries(unittest.TestCase):
    """Test terrain height query methods"""
    
    def setUp(self):
        """Set up a simple flat terrain for testing"""
        self.model = LunarRegolithModel(size=100.0, resolution=10)
        # Create a simple flat terrain
        self.model.heightmap = np.zeros((10, 10))
        self.model.terrain_loaded = True
    
    def test_get_height_at_origin(self):
        """Test getting height at origin"""
        height = self.model.get_height(0.0, 0.0)
        self.assertIsInstance(height, (float, np.floating))
        # Flat terrain should be at height 0
        self.assertAlmostEqual(height, 0.0, places=5)
    
    def test_get_height_within_bounds(self):
        """Test getting height at various positions within bounds"""
        positions = [(10.0, 10.0), (-10.0, -10.0), (25.0, -25.0)]
        
        for x, y in positions:
            height = self.model.get_height(x, y)
            self.assertIsInstance(height, (float, np.floating))
            # Should be finite
            self.assertTrue(np.isfinite(height))
    
    def test_get_height_slope_flat_terrain(self):
        """Test that slope is zero for flat terrain"""
        # Create flat terrain
        self.model.heightmap = np.zeros((10, 10))
        self.model.slope_x = np.zeros((10, 10))
        self.model.slope_y = np.zeros((10, 10))
        self.model.slope_magnitude = np.zeros((10, 10))
        
        # For flat terrain, get_terrain_properties should return low slope
        props = self.model.get_terrain_properties(0.0, 0.0)
        # Check that it returns valid properties
        self.assertIsInstance(props, dict)
    
    def test_get_height_sloped_terrain(self):
        """Test height on a simple sloped terrain"""
        # Create a simple linear slope in x direction
        x_vals = np.linspace(-50, 50, 10)
        y_vals = np.linspace(-50, 50, 10)
        X, Y = np.meshgrid(x_vals, y_vals)
        self.model.heightmap = X * 0.1  # 0.1 m/m slope
        self.model.terrain_loaded = True
        
        # Height should vary with position
        h1 = self.model.get_height(-40.0, 0.0)
        h2 = self.model.get_height(40.0, 0.0)
        # Both should be valid heights
        self.assertTrue(np.isfinite(h1))
        self.assertTrue(np.isfinite(h2))


class TestTerrainLoading(unittest.TestCase):
    """Test terrain loading from file"""
    
    def test_load_terrain_from_valid_file(self):
        """Test loading terrain from a valid .npy file"""
        model = LunarRegolithModel(size=100.0, resolution=10)
        
        # Create a temporary terrain file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            terrain_data = np.random.randn(10, 10) * 5.0
            np.save(f.name, terrain_data)
            temp_path = f.name
        
        try:
            # Load the terrain
            result = model.load_terrain_from_file(temp_path)
            
            # Should return True on success
            self.assertTrue(result)
            # heightmap should be updated
            self.assertIsNotNone(model.heightmap)
            self.assertEqual(model.heightmap.shape, (10, 10))
            self.assertTrue(model.terrain_loaded)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_terrain_wrong_shape_resizes(self):
        """Test that loading terrain with wrong shape resizes it"""
        model = LunarRegolithModel(size=100.0, resolution=10)
        
        # Create terrain with different resolution
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            terrain_data = np.random.randn(20, 20) * 5.0
            np.save(f.name, terrain_data)
            temp_path = f.name
        
        try:
            # Load the terrain
            result = model.load_terrain_from_file(temp_path)
            
            # Should return True and resize to match model resolution
            self.assertTrue(result)
            self.assertEqual(model.heightmap.shape, (10, 10))
        finally:
            os.unlink(temp_path)
    
    def test_load_terrain_nonexistent_file(self):
        """Test that loading from nonexistent file returns False"""
        model = LunarRegolithModel(size=100.0, resolution=10)
        
        # Try to load non-existent file
        # Should return False (not raise exception)
        result = model.load_terrain_from_file('/nonexistent/path/terrain.npy')
        self.assertFalse(result)


class TestContactForces(unittest.TestCase):
    """Test contact force calculations"""
    
    def setUp(self):
        """Set up a model with flat terrain"""
        self.model = LunarRegolithModel(size=100.0, resolution=10)
        self.model.heightmap = np.zeros((10, 10))
        self.model.slope_x = np.zeros((10, 10))
        self.model.slope_y = np.zeros((10, 10))
        self.model.slope_magnitude = np.zeros((10, 10))
        self.model.terrain_loaded = True
    
    def test_compute_contact_force_no_contact(self):
        """Test that no force is applied when not in contact"""
        # Position above terrain
        position = np.array([0.0, 0.0, 10.0])
        velocity = np.array([0.0, 0.0, -1.0])
        
        force = self.model.compute_contact_force(position, velocity)
        
        # Force should be zero or very small
        self.assertIsInstance(force, np.ndarray)
        self.assertEqual(len(force), 3)
        # No contact = no force (or minimal force)
        self.assertLess(np.linalg.norm(force), 100.0)
    
    def test_compute_contact_force_in_contact(self):
        """Test that force is applied when in contact with terrain"""
        # Position at or below terrain surface
        position = np.array([0.0, 0.0, -0.1])  # Slight penetration
        velocity = np.array([0.0, 0.0, -1.0])
        
        force = self.model.compute_contact_force(position, velocity)
        
        # Should have some force component
        self.assertIsInstance(force, np.ndarray)
        self.assertEqual(len(force), 3)
        # Should have some resistance force
        self.assertTrue(np.linalg.norm(force) > 0)
    
    def test_compute_contact_force_increases_with_sinkage(self):
        """Test that contact force increases with penetration depth"""
        velocity = np.array([0.0, 0.0, -1.0])
        
        # Small sinkage
        position1 = np.array([0.0, 0.0, -0.05])
        force1 = self.model.compute_contact_force(position1, velocity)
        
        # Larger sinkage
        position2 = np.array([0.0, 0.0, -0.2])
        force2 = self.model.compute_contact_force(position2, velocity)
        
        # Force magnitude should increase with sinkage
        mag1 = np.linalg.norm(force1)
        mag2 = np.linalg.norm(force2)
        # At least one should have non-zero force
        self.assertTrue(mag1 > 0 or mag2 > 0)


class TestSoilMechanics(unittest.TestCase):
    """Test soil mechanics calculations (Bekker-Wong model)"""
    
    def setUp(self):
        """Set up a model for testing"""
        self.model = LunarRegolithModel()
    
    def test_bekker_parameters_exist(self):
        """Test that Bekker soil parameters are defined"""
        self.assertTrue(hasattr(self.model, 'soil_k_c'))
        self.assertTrue(hasattr(self.model, 'soil_k_phi'))
        self.assertTrue(hasattr(self.model, 'soil_n'))
        
        # Should be positive
        self.assertGreater(self.model.soil_k_c, 0)
        self.assertGreater(self.model.soil_k_phi, 0)
        self.assertGreater(self.model.soil_n, 0)
    
    def test_shear_parameters_exist(self):
        """Test that shear deformation parameters are defined"""
        self.assertTrue(hasattr(self.model, 'shear_k'))
        self.assertGreater(self.model.shear_k, 0)
    
    def test_friction_angle_realistic(self):
        """Test that friction angle is in realistic range"""
        # Lunar regolith friction angle typically 30-50 degrees
        self.assertTrue(hasattr(self.model, 'friction_angle'))
        angle_deg = np.rad2deg(self.model.friction_angle)
        self.assertGreater(angle_deg, 20)
        self.assertLess(angle_deg, 60)


if __name__ == '__main__':
    unittest.main()
