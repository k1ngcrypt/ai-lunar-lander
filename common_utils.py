"""
common_utils.py
Common utility functions shared across the AI Lunar Lander project

This module consolidates frequently-used utility functions to reduce code duplication:
- Basilisk path setup
- Attitude conversion utilities (MRP, quaternion, DCM)
- Common mathematical operations
"""

import numpy as np
import sys
import os


# ============================================================================
# BASILISK PATH SETUP
# ============================================================================

def setup_basilisk_path():
    """
    Add Basilisk to Python path for imports.
    This function should be called at the start of any script that uses Basilisk.
    
    Returns:
        str: Path to Basilisk dist3 directory
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    basilisk_path = os.path.join(script_dir, 'basilisk', 'dist3')
    
    # Add to path if not already there
    if basilisk_path not in sys.path:
        sys.path.insert(0, basilisk_path)
    
    return basilisk_path


# ============================================================================
# ATTITUDE CONVERSION UTILITIES
# ============================================================================

def mrp_to_dcm(sigma):
    """
    Convert Modified Rodriguez Parameters (MRP) to Direction Cosine Matrix (DCM).
    
    The DCM transforms vectors from body frame to inertial frame: r_N = DCM @ r_B
    
    Args:
        sigma: (3,) array of MRP values [s1, s2, s3]
        
    Returns:
        DCM: (3, 3) Direction Cosine Matrix
    """
    s1, s2, s3 = sigma
    s_squared = s1*s1 + s2*s2 + s3*s3
    
    # Skew-symmetric matrix from MRP
    s_tilde = np.array([[0, -s3, s2],
                        [s3, 0, -s1],
                        [-s2, s1, 0]])
    
    # DCM formula: C = I + (8 * s_tilde^2) / (1 + s^2)^2 + (4 * (1 - s^2) * s_tilde) / (1 + s^2)^2
    C = np.eye(3)
    denom_sq = (1.0 + s_squared) ** 2
    C += (8.0 / denom_sq) * (s_tilde @ s_tilde)
    C += (4.0 * (1.0 - s_squared) / denom_sq) * s_tilde
    
    return C


def mrp_to_quaternion(sigma):
    """
    Convert Modified Rodriguez Parameters (MRP) to quaternion.
    
    Args:
        sigma: (3,) array of MRP values [s1, s2, s3]
        
    Returns:
        q: (4,) array representing quaternion [x, y, z, w]
    """
    s1, s2, s3 = sigma
    s_squared = s1*s1 + s2*s2 + s3*s3
    
    # MRP to quaternion formula
    denom = 1.0 + s_squared
    q = np.array([
        2.0 * s1 / denom,
        2.0 * s2 / denom,
        2.0 * s3 / denom,
        (1.0 - s_squared) / denom
    ])
    return q


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: (4,) array [x, y, z, w]
        q2: (4,) array [x, y, z, w]
        
    Returns:
        q: (4,) array representing q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_conjugate(q):
    """
    Compute quaternion conjugate (inverse for unit quaternions).
    
    Args:
        q: (4,) array [x, y, z, w]
        
    Returns:
        q_conj: (4,) array representing conjugate
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_error(q_current, q_target):
    """
    Compute attitude error quaternion between current and target attitudes.
    
    Args:
        q_current: (4,) array [x, y, z, w] current attitude
        q_target: (4,) array [x, y, z, w] target attitude
        
    Returns:
        q_error: (4,) array representing rotation from current to target
    """
    q_target_inv = quaternion_conjugate(q_target)
    q_error = quaternion_multiply(q_target_inv, q_current)
    return q_error


def quaternion_to_euler(quat):
    """
    Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw] in radians.
    Uses ZYX convention (yaw-pitch-roll).
    
    Args:
        quat: (4,) array [x, y, z, w]
        
    Returns:
        euler: (3,) array [roll, pitch, yaw] in radians
    """
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


# ============================================================================
# TERRAIN UTILITIES
# ============================================================================

def fade(t):
    """
    Smoothstep interpolation function for Perlin noise.
    
    Args:
        t: Input value (typically 0-1)
        
    Returns:
        Smoothed value
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation parameter (0-1)
        
    Returns:
        Interpolated value
    """
    return a + t * (b - a)


# ============================================================================
# COMMON MATHEMATICAL OPERATIONS
# ============================================================================

def clamp(value, min_val, max_val):
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def normalize_vector(vec):
    """
    Normalize a vector to unit length.
    
    Args:
        vec: Input vector (numpy array)
        
    Returns:
        Normalized vector, or zero vector if input is zero
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return np.zeros_like(vec)
    return vec / norm


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

import contextlib
import io


@contextlib.contextmanager
def suppress_basilisk_warnings():
    """
    Temporarily redirect stderr to suppress Basilisk warnings.
    
    Basilisk prints warnings directly to stderr (not through Python's warning system),
    so we need to temporarily redirect stderr to suppress them during initialization.
    
    These warnings are harmless - they occur when Basilisk's state engine registers
    properties during initialization, which is expected behavior.
    
    Usage:
        with suppress_basilisk_warnings():
            # Code that produces Basilisk warnings
            scSim.InitializeSimulation()
    """
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr
