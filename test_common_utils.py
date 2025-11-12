"""
Unit tests for common_utils.py
Tests utility functions for Basilisk path setup and attitude conversions.
"""

import unittest
import numpy as np
import sys
import os

# Import module under test
import common_utils


class TestBasiliskPathSetup(unittest.TestCase):
    """Test Basilisk path setup functionality"""
    
    def test_setup_basilisk_path_returns_string(self):
        """Test that setup_basilisk_path returns a valid path string"""
        path = common_utils.setup_basilisk_path()
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isabs(path))
        self.assertIn('basilisk', path)
        self.assertIn('dist3', path)
    
    def test_setup_basilisk_path_adds_to_sys_path(self):
        """Test that Basilisk path is added to sys.path"""
        path = common_utils.setup_basilisk_path()
        # Path should be in sys.path (either already there or just added)
        self.assertIn(path, sys.path)


class TestQuaternionToEuler(unittest.TestCase):
    """Test quaternion to Euler angle conversion"""
    
    def test_identity_quaternion(self):
        """Test conversion of identity quaternion (no rotation)"""
        q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        euler = common_utils.quaternion_to_euler(q)
        
        # Should be all zeros (no rotation)
        np.testing.assert_array_almost_equal(euler, np.zeros(3), decimal=6)
    
    def test_90_degree_roll(self):
        """Test 90-degree roll rotation"""
        # 90-degree roll: rotation around x-axis
        q = np.array([np.sin(np.pi/4), 0.0, 0.0, np.cos(np.pi/4)])
        euler = common_utils.quaternion_to_euler(q)
        
        # Expected: [pi/2, 0, 0] (90-degree roll)
        expected = np.array([np.pi/2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(euler, expected, decimal=6)
    
    def test_90_degree_pitch(self):
        """Test 90-degree pitch rotation"""
        # 90-degree pitch: rotation around y-axis
        q = np.array([0.0, np.sin(np.pi/4), 0.0, np.cos(np.pi/4)])
        euler = common_utils.quaternion_to_euler(q)
        
        # Expected: [0, pi/2, 0] (90-degree pitch)
        expected = np.array([0.0, np.pi/2, 0.0])
        np.testing.assert_array_almost_equal(euler, expected, decimal=6)
    
    def test_90_degree_yaw(self):
        """Test 90-degree yaw rotation"""
        # 90-degree yaw: rotation around z-axis
        q = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
        euler = common_utils.quaternion_to_euler(q)
        
        # Expected: [0, 0, pi/2] (90-degree yaw)
        expected = np.array([0.0, 0.0, np.pi/2])
        np.testing.assert_array_almost_equal(euler, expected, decimal=6)
    
    def test_output_dtype(self):
        """Test that output is float32"""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        euler = common_utils.quaternion_to_euler(q)
        self.assertEqual(euler.dtype, np.float32)
    
    def test_gimbal_lock_handling(self):
        """Test handling of gimbal lock condition"""
        # Pitch = +90 degrees (singularity)
        q = np.array([0.0, np.sin(np.pi/4), 0.0, np.cos(np.pi/4)])
        euler = common_utils.quaternion_to_euler(q)
        
        # Should not raise exception and should be finite
        self.assertTrue(np.all(np.isfinite(euler)))


class TestQuaternionOperations(unittest.TestCase):
    """Test quaternion mathematical operations"""
    
    def test_quaternion_multiply_identity(self):
        """Test multiplying by identity quaternion"""
        q1 = np.array([0.1, 0.2, 0.3, 0.9])
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])
        
        result = common_utils.quaternion_multiply(q1, q_identity)
        np.testing.assert_array_almost_equal(result, q1, decimal=6)
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugate"""
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q_conj = common_utils.quaternion_conjugate(q)
        
        expected = np.array([-0.1, -0.2, -0.3, 0.9])
        np.testing.assert_array_almost_equal(q_conj, expected, decimal=6)
    
    def test_quaternion_inverse_property(self):
        """Test that q * q_conj = identity (for unit quaternions)"""
        # Create a unit quaternion
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)  # Normalize
        
        q_conj = common_utils.quaternion_conjugate(q)
        result = common_utils.quaternion_multiply(q, q_conj)
        
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, identity, decimal=6)
    
    def test_quaternion_error(self):
        """Test quaternion error computation"""
        q_current = np.array([0.0, 0.0, 0.0, 1.0])
        q_target = np.array([0.1, 0.0, 0.0, 0.995])  # Small rotation
        
        q_error = common_utils.quaternion_error(q_current, q_target)
        
        # Error should be non-zero
        self.assertFalse(np.allclose(q_error, [0, 0, 0, 1]))


class TestMRPConversions(unittest.TestCase):
    """Test MRP (Modified Rodriguez Parameters) conversions"""
    
    def test_mrp_to_dcm_identity(self):
        """Test MRP to DCM for zero rotation"""
        sigma = np.array([0.0, 0.0, 0.0])
        dcm = common_utils.mrp_to_dcm(sigma)
        
        # Should be identity matrix
        np.testing.assert_array_almost_equal(dcm, np.eye(3), decimal=6)
    
    def test_mrp_to_dcm_orthogonal(self):
        """Test that DCM is orthogonal (DCM @ DCM^T = I)"""
        sigma = np.array([0.1, 0.2, 0.05])
        dcm = common_utils.mrp_to_dcm(sigma)
        
        # Check orthogonality
        product = dcm @ dcm.T
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=6)
    
    def test_mrp_to_dcm_determinant(self):
        """Test that DCM has determinant = 1 (proper rotation)"""
        sigma = np.array([0.1, 0.2, 0.05])
        dcm = common_utils.mrp_to_dcm(sigma)
        
        det = np.linalg.det(dcm)
        self.assertAlmostEqual(det, 1.0, places=6)
    
    def test_mrp_to_quaternion_identity(self):
        """Test MRP to quaternion for zero rotation"""
        sigma = np.array([0.0, 0.0, 0.0])
        q = common_utils.mrp_to_quaternion(sigma)
        
        # Should be identity quaternion
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(q, expected, decimal=6)
    
    def test_mrp_to_quaternion_normalized(self):
        """Test that output quaternion is normalized"""
        sigma = np.array([0.1, 0.2, 0.05])
        q = common_utils.mrp_to_quaternion(sigma)
        
        # Quaternion should be normalized
        norm = np.linalg.norm(q)
        self.assertAlmostEqual(norm, 1.0, places=6)


class TestSuppressBasiliskWarnings(unittest.TestCase):
    """Test the context manager for suppressing warnings"""
    
    def test_suppress_warnings_context_manager(self):
        """Test that suppress_basilisk_warnings works as context manager"""
        # This should not raise an exception
        with common_utils.suppress_basilisk_warnings():
            # Write something to stderr (normally would be visible)
            sys.stderr.write("Test warning message\n")
        
        # After exiting context, stderr should be restored
        self.assertIsNotNone(sys.stderr)
        self.assertTrue(hasattr(sys.stderr, 'write'))


if __name__ == '__main__':
    unittest.main()
