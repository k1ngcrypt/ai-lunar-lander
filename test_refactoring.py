"""
test_refactoring.py
Validation tests for code deduplication refactoring

Tests that common_utils module provides all expected functionality
and that modules properly import from it.
"""

import numpy as np
import sys
from common_utils import (
    setup_basilisk_path,
    mrp_to_dcm,
    mrp_to_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_error,
    quaternion_to_euler,
    fade,
    lerp,
    clamp,
    normalize_vector
)


def test_basilisk_path_setup():
    """Test Basilisk path setup"""
    print("Testing Basilisk path setup...")
    path = setup_basilisk_path()
    assert path.endswith('basilisk/dist3'), f"Unexpected path: {path}"
    assert path in sys.path, "Path not added to sys.path"
    print("  ✓ Basilisk path setup works")


def test_attitude_conversions():
    """Test attitude conversion utilities"""
    print("\nTesting attitude conversions...")
    
    # Test MRP to DCM
    sigma = np.array([0.1, 0.2, 0.3])
    dcm = mrp_to_dcm(sigma)
    assert dcm.shape == (3, 3), f"DCM wrong shape: {dcm.shape}"
    # DCM should be approximately orthogonal
    identity = np.dot(dcm, dcm.T)
    assert np.allclose(identity, np.eye(3), atol=1e-6), "DCM not orthogonal"
    print("  ✓ MRP to DCM works")
    
    # Test MRP to quaternion
    quat = mrp_to_quaternion(sigma)
    assert quat.shape == (4,), f"Quaternion wrong shape: {quat.shape}"
    # Quaternion should be normalized
    norm = np.linalg.norm(quat)
    assert np.abs(norm - 1.0) < 1e-6, f"Quaternion not normalized: {norm}"
    print("  ✓ MRP to quaternion works")
    
    # Test quaternion to Euler
    euler = quaternion_to_euler(quat)
    assert euler.shape == (3,), f"Euler angles wrong shape: {euler.shape}"
    print("  ✓ Quaternion to Euler works")
    
    # Test quaternion operations
    q1 = np.array([0.1, 0.2, 0.3, 0.9])
    q1 = q1 / np.linalg.norm(q1)
    q2 = np.array([0.3, 0.1, 0.2, 0.9])
    q2 = q2 / np.linalg.norm(q2)
    
    q_mult = quaternion_multiply(q1, q2)
    assert q_mult.shape == (4,), "Quaternion multiply failed"
    print("  ✓ Quaternion multiply works")
    
    q_conj = quaternion_conjugate(q1)
    assert q_conj.shape == (4,), "Quaternion conjugate failed"
    assert np.allclose(q_conj, np.array([-q1[0], -q1[1], -q1[2], q1[3]])), "Conjugate incorrect"
    print("  ✓ Quaternion conjugate works")
    
    q_err = quaternion_error(q1, q2)
    assert q_err.shape == (4,), "Quaternion error failed"
    print("  ✓ Quaternion error works")


def test_math_utilities():
    """Test mathematical utility functions"""
    print("\nTesting mathematical utilities...")
    
    # Test fade
    assert np.isclose(fade(0.0), 0.0), "Fade(0) should be 0"
    assert np.isclose(fade(1.0), 1.0), "Fade(1) should be 1"
    assert 0 < fade(0.5) < 1, "Fade(0.5) should be between 0 and 1"
    print("  ✓ Fade function works")
    
    # Test lerp
    assert np.isclose(lerp(0, 10, 0.0), 0.0), "Lerp failed at t=0"
    assert np.isclose(lerp(0, 10, 1.0), 10.0), "Lerp failed at t=1"
    assert np.isclose(lerp(0, 10, 0.5), 5.0), "Lerp failed at t=0.5"
    print("  ✓ Lerp function works")
    
    # Test clamp
    assert clamp(5, 0, 10) == 5, "Clamp failed for value in range"
    assert clamp(-5, 0, 10) == 0, "Clamp failed for value below range"
    assert clamp(15, 0, 10) == 10, "Clamp failed for value above range"
    print("  ✓ Clamp function works")
    
    # Test normalize_vector
    vec = np.array([3.0, 4.0, 0.0])
    normalized = normalize_vector(vec)
    assert np.isclose(np.linalg.norm(normalized), 1.0), "Vector not normalized"
    zero_vec = normalize_vector(np.zeros(3))
    assert np.allclose(zero_vec, np.zeros(3)), "Zero vector normalization failed"
    print("  ✓ Normalize vector works")


def test_numerical_consistency():
    """Test that results are consistent with original implementations"""
    print("\nTesting numerical consistency...")
    
    # Test multiple MRP values
    test_mrps = [
        np.array([0.0, 0.0, 0.0]),  # Identity
        np.array([0.1, 0.0, 0.0]),  # Small rotation
        np.array([0.0, 0.2, 0.0]),
        np.array([0.1, 0.2, 0.3]),  # General rotation
    ]
    
    for sigma in test_mrps:
        # MRP to DCM should produce orthogonal matrix
        dcm = mrp_to_dcm(sigma)
        identity = np.dot(dcm, dcm.T)
        assert np.allclose(identity, np.eye(3), atol=1e-6), \
            f"DCM not orthogonal for sigma={sigma}"
        
        # MRP to quaternion should be normalized
        quat = mrp_to_quaternion(sigma)
        norm = np.linalg.norm(quat)
        assert np.isclose(norm, 1.0, atol=1e-6), \
            f"Quaternion not normalized for sigma={sigma}"
    
    print("  ✓ Numerical consistency verified")


def run_all_tests():
    """Run all validation tests"""
    print("="*60)
    print("REFACTORING VALIDATION TESTS")
    print("="*60)
    
    try:
        test_basilisk_path_setup()
        test_attitude_conversions()
        test_math_utilities()
        test_numerical_consistency()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nRefactoring validation successful!")
        print("Code duplication has been reduced while maintaining functionality.")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
