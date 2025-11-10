"""
terrain_simulation.py
Lunar terrain simulation module for the lunar lander

This module provides realistic lunar regolith simulation with:
- Bilinear interpolated height maps from terrain data
- Realistic lunar soil mechanics (Bekker-Wong model)
- Slope-dependent friction and contact forces
- Boulder/rock contact detection

Separated from ScenarioLunarLanderStarter.py for better code maintainability.
"""

import os
import numpy as np


class LunarRegolithModel:
    """
    Enhanced analytical terrain model for realistic lunar landing simulation.
    
    Features:
    - Bilinear interpolated height map from high-resolution terrain data
    - Realistic lunar regolith mechanical properties (based on Apollo data)
    - Bekker-Wong soil mechanics model for bearing capacity
    - Janosi-Hanamoto shear model for lateral slip
    - Depth-dependent sinkage with realistic exponents
    - Slope-dependent friction and slip
    - Boulder/rock contact detection
    - High performance (vectorized operations)
    
    Physical Parameters Based On:
    - Apollo Soil Mechanics Surface Sampler data
    - Lunar regolith bearing strength: 3-50 kN/m² (varies with depth)
    - Internal friction angle: 30-50° (loose to dense regolith)
    - Cohesion: 0.1-1.0 kN/m² (very low)
    """
    
    def __init__(self, size=2000.0, resolution=100):
        """
        Args:
            size: Terrain size in meters (square terrain)
            resolution: Grid resolution (cells per side)
        """
        self.size = size
        self.resolution = resolution
        self.cell_size = size / resolution
        
        # ══════════════════════════════════════════════════════════════
        # REALISTIC LUNAR REGOLITH PROPERTIES (Apollo-derived)
        # ══════════════════════════════════════════════════════════════
        
        # Surface friction (varies with regolith composition)
        self.friction_angle = np.deg2rad(35)  # Internal friction angle (degrees)
        self.friction_coeff = np.tan(self.friction_angle)  # μ ≈ 0.7
        
        # Cohesion (very low for lunar regolith)
        self.cohesion = 500.0  # N/m² (0.5 kPa)
        
        # Bekker soil parameters for bearing capacity
        self.soil_k_c = 1000.0  # N/m^(n+1) - cohesive modulus
        self.soil_k_phi = 8000.0  # N/m^(n+2) - frictional modulus  
        self.soil_n = 1.2  # Sinkage exponent (1.0-1.5 for lunar soil)
        
        # Janosi shear deformation parameter
        self.shear_k = 0.025  # m (shear deformation modulus)
        
        # Damping (energy dissipation)
        self.damping_coeff = 3000.0  # N·s/m (vertical damping)
        self.lateral_damping = 500.0  # N·s/m (lateral damping)
        
        # Restitution coefficient (minimal bouncing)
        self.restitution = 0.05  # Very low (regolith absorbs energy)
        
        # Density of regolith (affects sinkage)
        self.regolith_density = 1500.0  # kg/m³ (varies 1200-1800)
        
        # ══════════════════════════════════════════════════════════════
        # TERRAIN DATA
        # ══════════════════════════════════════════════════════════════
        
        # Height map (z = f(x, y))
        self.heightmap = None
        self.slope_x = None  # Cached slope in x direction
        self.slope_y = None  # Cached slope in y direction
        self.slope_magnitude = None  # Cached total slope
        self.terrain_loaded = False
        
        # Terrain feature classification (for advanced contact)
        self.is_boulder = None  # Boolean mask for boulder locations
        self.is_bedrock = None  # Boolean mask for exposed bedrock
        
        # Random terrain variation parameters
        self.rng = np.random.RandomState(42)  # Reproducible randomness
        self.friction_variation = 0.15  # ±15% stochastic friction variation
        
        print(f"Enhanced Terrain Model Parameters:")
        print(f"  Size: {self.size}m x {self.size}m")
        print(f"  Resolution: {self.resolution} x {self.resolution}")
        print(f"  Cell size: {self.cell_size:.2f}m")
        print(f"\nLunar Regolith Properties:")
        print(f"  Friction angle: {np.rad2deg(self.friction_angle):.1f}deg (mu={self.friction_coeff:.2f})")
        print(f"  Cohesion: {self.cohesion:.1f} N/m²")
        print(f"  Sinkage exponent: {self.soil_n}")
        print(f"  Restitution: {self.restitution}")
        print(f"  Regolith density: {self.regolith_density} kg/m³")
        
    def load_terrain_from_file(self, filepath):
        """
        Load terrain heightmap from a file (e.g., generated terrain)
        Expected format: NumPy array (.npy) or CSV with height values
        
        Also computes terrain derivatives for slope-dependent mechanics.
        """
        if os.path.exists(filepath):
            if filepath.endswith('.npy'):
                self.heightmap = np.load(filepath)
            elif filepath.endswith('.csv'):
                self.heightmap = np.loadtxt(filepath, delimiter=',')
            else:
                print(f"⚠ Unsupported terrain file format: {filepath}")
                return False
            
            # Verify shape
            if self.heightmap.shape != (self.resolution, self.resolution):
                print(f"⚠ Heightmap shape mismatch: expected ({self.resolution}, {self.resolution}), got {self.heightmap.shape}")
                # Resize if needed
                print(f"  Using simple nearest-neighbor resize...")
                from numpy import linspace
                old_x = linspace(0, self.heightmap.shape[0]-1, self.resolution).astype(int)
                old_y = linspace(0, self.heightmap.shape[1]-1, self.resolution).astype(int)
                self.heightmap = self.heightmap[old_x][:, old_y]
                print(f"  Resized heightmap to ({self.resolution}, {self.resolution})")
            
            # Compute terrain gradients for slope-dependent contact
            print(f"  Computing terrain slopes...")
            self.slope_y, self.slope_x = np.gradient(self.heightmap, self.cell_size)
            self.slope_magnitude = np.sqrt(self.slope_x**2 + self.slope_y**2)
            
            # Identify terrain features
            self._classify_terrain_features()
            
            self.terrain_loaded = True
            print(f"[OK] Loaded terrain from: {filepath}")
            print(f"  Height range: [{np.min(self.heightmap):.2f}, {np.max(self.heightmap):.2f}] m")
            print(f"  Max slope: {np.rad2deg(np.arctan(np.max(self.slope_magnitude))):.1f}°")
            print(f"  Mean slope: {np.rad2deg(np.arctan(np.mean(self.slope_magnitude))):.1f}°")
            
            return True
        else:
            print(f"⚠ Terrain file not found: {filepath}")
            return False
    
    def _classify_terrain_features(self):
        """
        Classify terrain features based on local topology.
        Identifies boulders (sharp height changes) and bedrock (steep slopes).
        """
        if self.heightmap is None:
            return
        
        # Detect boulders: regions with high curvature (sharp peaks)
        laplacian = np.zeros_like(self.heightmap)
        laplacian[1:-1, 1:-1] = (
            self.heightmap[:-2, 1:-1] + self.heightmap[2:, 1:-1] +
            self.heightmap[1:-1, :-2] + self.heightmap[1:-1, 2:] -
            4 * self.heightmap[1:-1, 1:-1]
        ) / (self.cell_size**2)
        
        # Boulders have high positive curvature
        boulder_threshold = 0.5  # Adjust based on terrain scale
        self.is_boulder = laplacian > boulder_threshold
        
        # Bedrock: very steep slopes (>30°)
        bedrock_slope_threshold = np.tan(np.deg2rad(30))
        self.is_bedrock = self.slope_magnitude > bedrock_slope_threshold
        
        num_boulders = np.sum(self.is_boulder)
        num_bedrock = np.sum(self.is_bedrock)
        total_cells = self.resolution * self.resolution
        
        print(f"  Terrain features:")
        print(f"    Boulder regions: {num_boulders} cells ({100*num_boulders/total_cells:.1f}%)")
        print(f"    Bedrock regions: {num_bedrock} cells ({100*num_bedrock/total_cells:.1f}%)")
    
    def generate_procedural_terrain(self, num_craters=10, crater_depth_range=(2, 10), 
                                    crater_radius_range=(10, 50)):
        """
        Generate simple procedural terrain with craters and random noise
        """
        # Start with flat terrain
        x = np.linspace(-self.size/2, self.size/2, self.resolution)
        y = np.linspace(-self.size/2, self.size/2, self.resolution)
        X, Y = np.meshgrid(x, y)
        self.heightmap = np.zeros((self.resolution, self.resolution))
        
        # Add craters (simple Gaussian depressions)
        for _ in range(num_craters):
            cx = self.rng.uniform(-self.size/2 + 100, self.size/2 - 100)
            cy = self.rng.uniform(-self.size/2 + 100, self.size/2 - 100)
            depth = self.rng.uniform(*crater_depth_range)
            radius = self.rng.uniform(*crater_radius_range)
            
            # Gaussian crater shape
            dist_sq = (X - cx)**2 + (Y - cy)**2
            crater = -depth * np.exp(-dist_sq / (2 * radius**2))
            self.heightmap += crater
        
        # Add small-scale roughness (high-frequency noise)
        roughness = self.rng.normal(0, 0.1, (self.resolution, self.resolution))
        self.heightmap += roughness
        
        self.terrain_loaded = True
        print(f"✓ Generated procedural terrain with {num_craters} craters")
        print(f"  Height range: [{np.min(self.heightmap):.2f}, {np.max(self.heightmap):.2f}] m")
    
    def get_height(self, x, y):
        """
        Get terrain height at position (x, y) using bilinear interpolation
        """
        if not self.terrain_loaded:
            return 0.0  # Flat terrain fallback
        
        # Convert world coordinates to grid indices
        # Grid: -size/2 to +size/2 maps to 0 to resolution-1
        grid_x = (x + self.size/2) / self.cell_size
        grid_y = (y + self.size/2) / self.cell_size
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.resolution - 1 or grid_y < 0 or grid_y >= self.resolution - 1:
            return 0.0  # Outside terrain bounds
        
        # Bilinear interpolation
        ix = int(np.floor(grid_x))
        iy = int(np.floor(grid_y))
        fx = grid_x - ix
        fy = grid_y - iy
        
        # Get 4 corner heights
        h00 = self.heightmap[iy, ix]
        h10 = self.heightmap[iy, ix + 1]
        h01 = self.heightmap[iy + 1, ix]
        h11 = self.heightmap[iy + 1, ix + 1]
        
        # Interpolate
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        height = h0 * (1 - fy) + h1 * fy
        
        return height
    
    def get_terrain_normal(self, x, y):
        """
        Get terrain surface normal at position (x, y).
        Returns normalized vector pointing up from surface.
        """
        if not self.terrain_loaded or self.slope_x is None:
            return np.array([0., 0., 1.])  # Flat terrain
        
        # Convert to grid coordinates
        grid_x = (x + self.size/2) / self.cell_size
        grid_y = (y + self.size/2) / self.cell_size
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.resolution - 1 or grid_y < 0 or grid_y >= self.resolution - 1:
            return np.array([0., 0., 1.])
        
        # Bilinear interpolation of slopes
        ix = int(np.floor(grid_x))
        iy = int(np.floor(grid_y))
        fx = grid_x - ix
        fy = grid_y - iy
        
        # Interpolate slope_x
        sx00 = self.slope_x[iy, ix]
        sx10 = self.slope_x[iy, ix + 1]
        sx01 = self.slope_x[iy + 1, ix]
        sx11 = self.slope_x[iy + 1, ix + 1]
        sx = (sx00 * (1-fx) + sx10 * fx) * (1-fy) + (sx01 * (1-fx) + sx11 * fx) * fy
        
        # Interpolate slope_y
        sy00 = self.slope_y[iy, ix]
        sy10 = self.slope_y[iy, ix + 1]
        sy01 = self.slope_y[iy + 1, ix]
        sy11 = self.slope_y[iy + 1, ix + 1]
        sy = (sy00 * (1-fx) + sy10 * fx) * (1-fy) + (sy01 * (1-fx) + sy11 * fx) * fy
        
        # Normal vector from cross product of tangent vectors
        # tangent_x = [1, 0, slope_x], tangent_y = [0, 1, slope_y]
        normal = np.array([-sx, -sy, 1.0])
        normal = normal / np.linalg.norm(normal)
        
        return normal
    
    def get_terrain_properties(self, x, y):
        """
        Get local terrain properties at position (x, y).
        
        Returns:
            dict: friction_coeff, is_boulder, is_bedrock, cohesion
        """
        if not self.terrain_loaded:
            return {
                'friction_coeff': self.friction_coeff,
                'is_boulder': False,
                'is_bedrock': False,
                'cohesion': self.cohesion
            }
        
        # Convert to grid coordinates
        grid_x = (x + self.size/2) / self.cell_size
        grid_y = (y + self.size/2) / self.cell_size
        
        # Check bounds
        if grid_x < 0 or grid_x >= self.resolution - 1 or grid_y < 0 or grid_y >= self.resolution - 1:
            return {
                'friction_coeff': self.friction_coeff,
                'is_boulder': False,
                'is_bedrock': False,
                'cohesion': self.cohesion
            }
        
        ix = int(np.floor(grid_x))
        iy = int(np.floor(grid_y))
        
        # Get terrain classification
        is_boulder = self.is_boulder[iy, ix] if self.is_boulder is not None else False
        is_bedrock = self.is_bedrock[iy, ix] if self.is_bedrock is not None else False
        
        # Adjust properties based on terrain type
        friction = self.friction_coeff
        cohesion = self.cohesion
        
        if is_boulder:
            # Boulders have higher friction, no cohesion
            friction *= 1.5
            cohesion = 0.0
        elif is_bedrock:
            # Bedrock is very hard, high friction
            friction *= 1.3
            cohesion = 0.0
        
        return {
            'friction_coeff': friction,
            'is_boulder': is_boulder,
            'is_bedrock': is_bedrock,
            'cohesion': cohesion
        }
    
    def compute_contact_force(self, position, velocity, contact_area=1.0, contact_width=0.5):
        """
        Compute realistic contact force using Bekker-Wong terramechanics model.
        
        Implements:
        - Bekker bearing capacity equation for normal force
        - Janosi-Hanamoto shear model for lateral slip
        - Slope-dependent force direction
        - Terrain-type dependent properties
        
        Args:
            position: [x, y, z] in inertial frame (m)
            velocity: [vx, vy, vz] in inertial frame (m/s)
            contact_area: Contact patch area (m²)
            contact_width: Contact patch width (m) - for pressure calculation
        
        Returns:
            force: [fx, fy, fz] contact force in inertial frame (N)
        """
        x, y, z = position
        vx, vy, vz = velocity
        
        # Get terrain height and properties
        terrain_height = self.get_height(x, y)
        terrain_normal = self.get_terrain_normal(x, y)
        terrain_props = self.get_terrain_properties(x, y)
        
        # Penetration depth (positive when below surface)
        penetration = terrain_height - z
        
        if penetration <= 0:
            return np.zeros(3)
        
        # ══════════════════════════════════════════════════════════════
        # NORMAL FORCE (Bekker bearing capacity model)
        # ══════════════════════════════════════════════════════════════
        # Pressure: p = (k_c/b + k_phi) * z^n
        # Force: F_n = p * A
        
        b = max(contact_width, 0.1)
        
        if terrain_props['is_boulder'] or terrain_props['is_bedrock']:
            # Hard surface: high stiffness, nearly linear
            bearing_pressure = 500000.0 * (penetration ** 1.0)
        else:
            # Soft regolith: Bekker model
            cohesive_term = self.soil_k_c / b + terrain_props['cohesion'] / b
            frictional_term = self.soil_k_phi
            bearing_pressure = (cohesive_term + frictional_term) * (penetration ** self.soil_n)
        
        # Normal force magnitude (before damping)
        normal_force_mag = bearing_pressure * contact_area
        
        # Add velocity-dependent damping (energy dissipation)
        # Project velocity onto surface normal
        velocity_vec = np.array([vx, vy, vz])
        v_normal = np.dot(velocity_vec, terrain_normal)
        damping_force = -self.damping_coeff * v_normal
        
        # Total normal force magnitude
        normal_force_mag = max(0.0, normal_force_mag + damping_force)
        
        # Normal force vector (along surface normal)
        normal_force_vec = normal_force_mag * terrain_normal
        
        # ══════════════════════════════════════════════════════════════
        # LATERAL SHEAR FORCE (Janosi-Hanamoto model)
        # ══════════════════════════════════════════════════════════════
        # Shear stress: τ = (c + σ tan φ) * (1 - e^(-j/K))
        # j = slip distance, K = shear deformation parameter
        
        v_lateral_vec = velocity_vec - v_normal * terrain_normal
        v_lateral_speed = np.linalg.norm(v_lateral_vec)
        
        if v_lateral_speed > 1e-6:
            # Approximate slip distance over characteristic time
            slip_distance = v_lateral_speed * 0.1
            
            # Maximum shear stress
            cohesive_shear = terrain_props['cohesion']
            frictional_shear = bearing_pressure * np.tan(self.friction_angle)
            max_shear_stress = cohesive_shear + frictional_shear
            
            # Janosi-Hanamoto mobilization factor
            mobilization = 1.0 - np.exp(-slip_distance / self.shear_k)
            
            # Add stochastic variation (terrain heterogeneity)
            variation_factor = 1.0 + self.rng.uniform(-self.friction_variation, 
                                                       self.friction_variation)
            
            # Shear stress
            shear_stress = max_shear_stress * mobilization * variation_factor
            
            # Shear force (oppose lateral motion)
            shear_force_mag = min(shear_stress * contact_area, 
                                  normal_force_mag * terrain_props['friction_coeff'] * 1.5)  # Cap at friction limit
            
            # Shear force vector (opposite to lateral velocity)
            shear_force_vec = -shear_force_mag * v_lateral_vec / v_lateral_speed
            
            # Add lateral damping
            lateral_damping_vec = -self.lateral_damping * v_lateral_vec
            shear_force_vec += lateral_damping_vec
            
        else:
            # No lateral motion - static friction
            shear_force_vec = np.zeros(3)
        
        # ══════════════════════════════════════════════════════════════
        # TOTAL CONTACT FORCE
        # ══════════════════════════════════════════════════════════════
        total_force = normal_force_vec + shear_force_vec
        
        return total_force
