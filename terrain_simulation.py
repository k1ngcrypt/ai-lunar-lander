"""
terrain_simulation.py
Lunar terrain simulation module for the lunar lander

Provides realistic lunar regolith simulation with:
- Bilinear interpolated height maps from terrain data
- Realistic lunar soil mechanics (Bekker-Wong model)
- Slope-dependent friction and contact forces
- Boulder/rock contact detection
"""

import os
import numpy as np

import starship_constants as SC


class LunarRegolithModel:
    """
    Analytical terrain model for realistic lunar landing simulation.
    
    Features:
    - Bilinear interpolated height map from high-resolution terrain data
    - Realistic lunar regolith mechanical properties (Apollo data)
    - Bekker-Wong soil mechanics for bearing capacity
    - Janosi-Hanamoto shear model for lateral slip
    - Depth-dependent sinkage
    - Slope-dependent friction
    - Boulder/rock contact detection
    
    Physical Parameters Based On:
    - Apollo Soil Mechanics Surface Sampler data
    - Lunar regolith bearing strength: 3-50 kN/m²
    - Internal friction angle: 30-50°
    - Cohesion: 0.1-1.0 kN/m²
    """
    
    def __init__(self, size=2000.0, resolution=100):
        """
        Args:
            size: Terrain size in meters (square)
            resolution: Grid resolution (cells per side)
        """
        self.size = size
        self.resolution = resolution
        self.cell_size = size / resolution
        
        # Realistic lunar regolith properties (Apollo-derived)
        
        self.friction_angle = np.deg2rad(35)
        self.friction_coeff = np.tan(self.friction_angle)  # μ ≈ 0.7
        
        self.cohesion = 500.0  # N/m² (0.5 kPa)
        
        # Bekker soil parameters for bearing capacity
        self.soil_k_c = 1000.0  # N/m^(n+1)
        self.soil_k_phi = 8000.0  # N/m^(n+2)
        self.soil_n = 1.2  # Sinkage exponent (1.0-1.5 for lunar soil)
        
        self.shear_k = 0.025  # m (Janosi shear deformation modulus)
        
        self.damping_coeff = 3000.0  # N·s/m (vertical)
        self.lateral_damping = 500.0  # N·s/m (lateral)
        
        self.restitution = 0.05  # Minimal bouncing
        
        self.regolith_density = 1500.0  # kg/m³ (varies 1200-1800)
        
        # Terrain data
        self.heightmap = None
        self.slope_x = None
        self.slope_y = None
        self.slope_magnitude = None
        self.terrain_loaded = False
        
        self.is_boulder = None
        self.is_bedrock = None
        
        self.rng = np.random.RandomState(42)
        self.friction_variation = 0.15  # ±15% stochastic variation
        
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
        Load terrain heightmap from file (NumPy .npy or CSV).
        
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
            
            if self.heightmap.shape != (self.resolution, self.resolution):
                print(f"⚠ Heightmap shape mismatch: expected ({self.resolution}, {self.resolution}), got {self.heightmap.shape}")
                print(f"  Using simple nearest-neighbor resize...")
                from numpy import linspace
                old_x = linspace(0, self.heightmap.shape[0]-1, self.resolution).astype(int)
                old_y = linspace(0, self.heightmap.shape[1]-1, self.resolution).astype(int)
                self.heightmap = self.heightmap[old_x][:, old_y]
                print(f"  Resized heightmap to ({self.resolution}, {self.resolution})")
            
            print(f"  Computing terrain slopes...")
            self.slope_y, self.slope_x = np.gradient(self.heightmap, self.cell_size)
            self.slope_magnitude = np.sqrt(self.slope_x**2 + self.slope_y**2)
            
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
        Classify terrain features based on topology.
        Identifies boulders (sharp height changes) and bedrock (steep slopes).
        """
        if self.heightmap is None:
            return
        
        laplacian = np.zeros_like(self.heightmap)
        laplacian[1:-1, 1:-1] = (
            self.heightmap[:-2, 1:-1] + self.heightmap[2:, 1:-1] +
            self.heightmap[1:-1, :-2] + self.heightmap[1:-1, 2:] -
            4 * self.heightmap[1:-1, 1:-1]
        ) / (self.cell_size**2)
        
        boulder_threshold = 0.5
        self.is_boulder = laplacian > boulder_threshold
        
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
        """Generate simple procedural terrain with craters and random noise"""
        x = np.linspace(-self.size/2, self.size/2, self.resolution)
        y = np.linspace(-self.size/2, self.size/2, self.resolution)
        X, Y = np.meshgrid(x, y)
        self.heightmap = np.zeros((self.resolution, self.resolution))
        
        for _ in range(num_craters):
            cx = self.rng.uniform(-self.size/2 + 100, self.size/2 - 100)
            cy = self.rng.uniform(-self.size/2 + 100, self.size/2 - 100)
            depth = self.rng.uniform(*crater_depth_range)
            radius = self.rng.uniform(*crater_radius_range)
            
            dist_sq = (X - cx)**2 + (Y - cy)**2
            crater = -depth * np.exp(-dist_sq / (2 * radius**2))
            self.heightmap += crater
        
        roughness = self.rng.normal(0, 0.1, (self.resolution, self.resolution))
        self.heightmap += roughness
        
        self.terrain_loaded = True
    
    def get_height(self, x, y):
        """Get terrain height at (x, y) using bilinear interpolation"""
        if not self.terrain_loaded:
            return 0.0
        
        # Convert world coords to grid indices (-size/2 to +size/2 → 0 to resolution-1)
        grid_x = (x + self.size/2) / self.cell_size
        grid_y = (y + self.size/2) / self.cell_size
        
        if grid_x < 0 or grid_x >= self.resolution - 1 or grid_y < 0 or grid_y >= self.resolution - 1:
            return 0.0
        
        ix = int(np.floor(grid_x))
        iy = int(np.floor(grid_y))
        fx = grid_x - ix
        fy = grid_y - iy
        
        h00 = self.heightmap[iy, ix]
        h10 = self.heightmap[iy, ix + 1]
        h01 = self.heightmap[iy + 1, ix]
        h11 = self.heightmap[iy + 1, ix + 1]
        
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
        
        normal = np.array([-sx, -sy, 1.0])
        normal = normal / np.linalg.norm(normal)
        
        return normal
    
    def get_terrain_properties(self, x, y):
        """
        Get local terrain properties at (x, y).
        
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
        
        grid_x = (x + self.size/2) / self.cell_size
        grid_y = (y + self.size/2) / self.cell_size
        
        if grid_x < 0 or grid_x >= self.resolution - 1 or grid_y < 0 or grid_y >= self.resolution - 1:
            return {
                'friction_coeff': self.friction_coeff,
                'is_boulder': False,
                'is_bedrock': False,
                'cohesion': self.cohesion
            }
        
        ix = int(np.floor(grid_x))
        iy = int(np.floor(grid_y))
        
        is_boulder = self.is_boulder[iy, ix] if self.is_boulder is not None else False
        is_bedrock = self.is_bedrock[iy, ix] if self.is_bedrock is not None else False
        
        friction = self.friction_coeff
        cohesion = self.cohesion
        
        if is_boulder:
            friction *= 1.5
            cohesion = 0.0
        elif is_bedrock:
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
        Compute realistic contact force using Bekker-Wong terramechanics.
        
        Implements:
        - Bekker bearing capacity for normal force
        - Janosi-Hanamoto shear for lateral slip
        - Slope-dependent force direction
        - Terrain-type dependent properties
        
        Args:
            position: [x, y, z] in inertial frame (m)
            velocity: [vx, vy, vz] in inertial frame (m/s)
            contact_area: Contact patch area (m²)
            contact_width: Contact patch width (m)
        
        Returns:
            force: [fx, fy, fz] in inertial frame (N)
        """
        x, y, z = position
        vx, vy, vz = velocity
        
        terrain_height = self.get_height(x, y)
        terrain_normal = self.get_terrain_normal(x, y)
        terrain_props = self.get_terrain_properties(x, y)
        
        # CRITICAL: z is Moon-centered, subtract MOON_RADIUS for altitude
        altitude_above_surface = z - SC.MOON_RADIUS
        
        penetration = terrain_height - altitude_above_surface
        
        if penetration <= 0:
            return np.zeros(3)
        
        # Normal force (Bekker bearing capacity)
        # Pressure: p = (k_c/b + k_phi) * z^n
        
        b = max(contact_width, 0.1)
        
        if terrain_props['is_boulder'] or terrain_props['is_bedrock']:
            bearing_pressure = 500000.0 * (penetration ** 1.0)
        else:
            cohesive_term = self.soil_k_c / b + terrain_props['cohesion'] / b
            frictional_term = self.soil_k_phi
            bearing_pressure = (cohesive_term + frictional_term) * (penetration ** self.soil_n)
        
        normal_force_mag = bearing_pressure * contact_area
        
        velocity_vec = np.array([vx, vy, vz])
        v_normal = np.dot(velocity_vec, terrain_normal)
        damping_force = -self.damping_coeff * v_normal
        
        normal_force_mag = max(0.0, normal_force_mag + damping_force)
        
        normal_force_vec = normal_force_mag * terrain_normal
        
        # Lateral shear force (Janosi-Hanamoto)
        # τ = (c + σ tan φ) * (1 - e^(-j/K))
        
        v_lateral_vec = velocity_vec - v_normal * terrain_normal
        v_lateral_speed = np.linalg.norm(v_lateral_vec)
        
        if v_lateral_speed > 1e-6:
            slip_distance = v_lateral_speed * 0.1
            
            cohesive_shear = terrain_props['cohesion']
            frictional_shear = bearing_pressure * np.tan(self.friction_angle)
            max_shear_stress = cohesive_shear + frictional_shear
            
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
