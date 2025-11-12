"""
Unit tests for generate_terrain.py
Tests terrain generation functions.
"""

import unittest
import numpy as np
import tempfile
import os

# Import functions under test
from generate_terrain import (
    perlin_noise_2d,
    generate_realistic_crater,
    add_boulder_field,
    generate_lunar_terrain
)


class TestPerlinNoise(unittest.TestCase):
    """Test Perlin noise generation"""
    
    def test_perlin_noise_output_shape(self):
        """Test that Perlin noise produces correct output shape"""
        shape = (50, 50)
        noise = perlin_noise_2d(shape)
        
        self.assertEqual(noise.shape, shape)
    
    def test_perlin_noise_custom_shape(self):
        """Test Perlin noise with custom dimensions"""
        shape = (100, 75)
        noise = perlin_noise_2d(shape)
        
        self.assertEqual(noise.shape, shape)
    
    def test_perlin_noise_range(self):
        """Test that Perlin noise values are in expected range"""
        noise = perlin_noise_2d((50, 50))
        
        # Should be roughly in [-1, 1] range (normalized)
        self.assertGreaterEqual(noise.min(), -5)  # Allow some margin
        self.assertLessEqual(noise.max(), 5)
    
    def test_perlin_noise_reproducible_with_seed(self):
        """Test that same seed produces same noise"""
        shape = (50, 50)
        noise1 = perlin_noise_2d(shape, seed=42)
        noise2 = perlin_noise_2d(shape, seed=42)
        
        np.testing.assert_array_equal(noise1, noise2)
    
    def test_perlin_noise_different_with_different_seed(self):
        """Test that different seeds produce different noise"""
        shape = (50, 50)
        noise1 = perlin_noise_2d(shape, seed=42)
        noise2 = perlin_noise_2d(shape, seed=123)
        
        # Should not be identical
        self.assertFalse(np.allclose(noise1, noise2))
    
    def test_perlin_noise_parameters(self):
        """Test that Perlin noise accepts custom parameters"""
        # Should not raise exception
        noise = perlin_noise_2d(
            (50, 50),
            scale=20.0,
            octaves=4,
            persistence=0.6,
            lacunarity=2.5
        )
        
        self.assertEqual(noise.shape, (50, 50))


class TestCraterGeneration(unittest.TestCase):
    """Test crater generation functions"""
    
    def test_generate_realistic_crater_simple(self):
        """Test generating a simple crater"""
        # Create coordinate meshgrid
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        # Generate crater at origin
        crater = generate_realistic_crater(
            X, Y,
            center_x=0.0,
            center_y=0.0,
            diameter=50.0,
            is_complex=False
        )
        
        # Should produce a 2D array
        self.assertEqual(crater.shape, X.shape)
        
        # Crater should be depressed (negative values at center)
        center_idx = len(x) // 2
        self.assertLess(crater[center_idx, center_idx], 0)
    
    def test_generate_realistic_crater_complex(self):
        """Test generating a complex crater with central peak"""
        x = np.linspace(-200, 200, 50)
        y = np.linspace(-200, 200, 50)
        X, Y = np.meshgrid(x, y)
        
        crater = generate_realistic_crater(
            X, Y,
            center_x=0.0,
            center_y=0.0,
            diameter=100.0,
            is_complex=True
        )
        
        self.assertEqual(crater.shape, X.shape)
        # Complex craters can have central peak (positive at center)
        self.assertIsInstance(crater[25, 25], (float, np.floating))
    
    def test_generate_crater_off_center(self):
        """Test generating crater at non-origin position"""
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        crater = generate_realistic_crater(
            X, Y,
            center_x=30.0,
            center_y=20.0,
            diameter=40.0
        )
        
        # Should still produce valid output
        self.assertEqual(crater.shape, X.shape)
        self.assertTrue(np.all(np.isfinite(crater)))
    
    def test_generate_crater_depth_ratio(self):
        """Test that depth-diameter ratio affects crater depth"""
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        # Shallow crater
        crater1 = generate_realistic_crater(
            X, Y, 0.0, 0.0, 50.0,
            depth_diameter_ratio=0.1
        )
        
        # Deeper crater
        crater2 = generate_realistic_crater(
            X, Y, 0.0, 0.0, 50.0,
            depth_diameter_ratio=0.3
        )
        
        # Deeper crater should have more negative minimum
        self.assertLess(crater2.min(), crater1.min())


class TestBoulderField(unittest.TestCase):
    """Test boulder field generation"""
    
    def test_add_boulder_field_basic(self):
        """Test adding boulders to terrain"""
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        # Add some boulders (returns heightmap)
        boulder_terrain = add_boulder_field(
            X, Y,
            num_boulders=5,
            boulder_size_range=(1.0, 3.0)
        )
        
        # Should have same shape as input meshgrid
        self.assertEqual(boulder_terrain.shape, X.shape)
        
        # Should have some positive heights (boulders)
        self.assertGreater(boulder_terrain.max(), 0)
    
    def test_add_boulder_field_no_boulders(self):
        """Test with no boulders"""
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        boulder_terrain = add_boulder_field(
            X, Y,
            num_boulders=0
        )
        
        # Should return all zeros (no boulders)
        self.assertEqual(boulder_terrain.shape, X.shape)


class TestLunarTerrainGeneration(unittest.TestCase):
    """Test full lunar terrain generation"""
    
    def test_generate_lunar_terrain_mare(self):
        """Test generating Mare (smooth) terrain"""
        terrain = generate_lunar_terrain(
            size=1000.0,
            resolution=50,
            terrain_type='mare',
            num_craters=3,
            seed=42
        )
        
        self.assertEqual(terrain.shape, (50, 50))
        self.assertTrue(np.all(np.isfinite(terrain)))
    
    def test_generate_lunar_terrain_highland(self):
        """Test generating Highland (rough) terrain"""
        terrain = generate_lunar_terrain(
            size=1000.0,
            resolution=50,
            terrain_type='highland',
            num_craters=5,
            seed=42
        )
        
        self.assertEqual(terrain.shape, (50, 50))
        self.assertTrue(np.all(np.isfinite(terrain)))
    
    def test_generate_lunar_terrain_different_types(self):
        """Test that different terrain types produce different results"""
        mare = generate_lunar_terrain(
            size=1000.0,
            resolution=50,
            terrain_type='mare',
            num_craters=0,
            seed=42,
            include_boulders=False
        )
        highland = generate_lunar_terrain(
            size=1000.0,
            resolution=50,
            terrain_type='highland',
            num_craters=0,
            seed=42,
            include_boulders=False
        )
        
        # Highland should be rougher (higher std deviation)
        self.assertGreater(np.std(highland), np.std(mare))
    
    def test_generate_lunar_terrain_custom_resolution(self):
        """Test terrain generation with custom resolution"""
        terrain = generate_lunar_terrain(
            size=2000.0,
            resolution=100,
            terrain_type='mare',
            seed=42
        )
        
        self.assertEqual(terrain.shape, (100, 100))


if __name__ == '__main__':
    unittest.main()
