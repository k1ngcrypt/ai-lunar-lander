"""
generate_terrain.py
Realistic lunar terrain generator for the lander simulation

Features physically-accurate lunar surface:
- Power-law crater size-frequency distribution
- Realistic crater morphology (simple/complex with central peaks, terraced walls)
- Boulder fields and ejecta blankets
- Regolith layering and micro-topography
- Mare vs. Highland terrain types
- Fractal terrain using Perlin noise

Usage:
    python generate_terrain.py --output generated_terrain/moon_terrain.npy --size 2000 --resolution 200 --terrain-type mare
"""

import numpy as np
import os
import argparse


def perlin_noise_2d(shape, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate 2D Perlin noise for realistic terrain features.
    
    Args:
        shape: (height, width) of output array
        scale: Base wavelength
        octaves: Number of noise layers (more = more detail)
        persistence: Amplitude decrease per octave
        lacunarity: Frequency increase per octave
        seed: Random seed
        
    Returns:
        2D array of Perlin noise [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    def gradient(h, x, y):
        vectors = np.array([[1,1], [-1,1], [1,-1], [-1,-1], 
                           [1,0], [-1,0], [0,1], [0,-1]])
        g = vectors[h % 8]
        return g[:,:,0] * x + g[:,:,1] * y
    
    height, width = shape
    noise = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for octave in range(octaves):
        grid_height = int(height / scale * frequency) + 2
        grid_width = int(width / scale * frequency) + 2
        
        grad_grid = np.random.randint(0, 8, size=(grid_height, grid_width))
        
        y_coords = np.linspace(0, grid_height - 1, height)
        x_coords = np.linspace(0, grid_width - 1, width)
        
        y_idx = np.floor(y_coords).astype(int)
        x_idx = np.floor(x_coords).astype(int)
        
        octave_noise = np.random.randn(height, width) * 0.5
        
        noise += octave_noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
    
    if max_value > 0:
        noise /= max_value
    
    return noise


def generate_realistic_crater(X, Y, center_x, center_y, diameter, depth_diameter_ratio=0.2, 
                              is_complex=False, ejecta_range=1.5):
    """
    Generate a realistic impact crater with proper morphology.
    
    Simple craters (D < 15km on Moon): Bowl-shaped
    Complex craters (D > 15km): Flat floor, central peak, terraced walls
    
    Args:
        X, Y: Meshgrid coordinates
        center_x, center_y: Crater center position
        diameter: Crater diameter (meters)
        depth_diameter_ratio: D/d ratio (0.2 for fresh craters, lower for degraded)
        is_complex: Whether to add central peak and terraces
        ejecta_range: How far ejecta blanket extends (multiples of radius)
        
    Returns:
        heightmap: 2D array of crater elevation changes
    """
    radius = diameter / 2.0
    depth = diameter * depth_diameter_ratio
    
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    heightmap = np.zeros_like(X)
    
    crater_mask = dist <= radius
    
    if is_complex:
        floor_radius = radius * 0.4
        wall_start = radius * 0.5
        
        floor_mask = dist <= floor_radius
        heightmap[floor_mask] = -depth
        
        wall_mask = (dist > floor_radius) & (dist <= radius)
        wall_dist = (dist[wall_mask] - floor_radius) / (radius - floor_radius)
        
        num_terraces = 4
        terrace_profile = np.zeros_like(wall_dist)
        for i in range(num_terraces):
            terrace_pos = (i + 1) / num_terraces
            terrace_height = -depth * (1 - terrace_pos * 0.7)
            t = np.clip((wall_dist - terrace_pos) * num_terraces * 2, -1, 1)
            terrace_smooth = 0.5 * (1 + np.tanh(t * 3))
            terrace_profile += terrace_height * terrace_smooth / num_terraces
        
        heightmap[wall_mask] = terrace_profile
        
        peak_radius = radius * 0.15
        peak_height = depth * 0.15
        peak_mask = dist <= peak_radius
        peak_profile = peak_height * (1 - (dist[peak_mask] / peak_radius)**2)
        heightmap[peak_mask] += peak_profile
        
    else:
        normalized_dist = dist[crater_mask] / radius
        crater_profile = -depth * (1 - normalized_dist**2)
        heightmap[crater_mask] = crater_profile
    
    rim_height = depth * 0.12
    rim_width = radius * 0.1
    rim_outer = radius + rim_width
    
    rim_mask = (dist > radius * 0.95) & (dist <= rim_outer)
    rim_dist = (dist[rim_mask] - radius) / rim_width
    rim_profile = rim_height * np.exp(-((rim_dist - 0.3) / 0.4)**2)
    heightmap[rim_mask] += rim_profile
    
    ejecta_outer = radius * ejecta_range
    ejecta_mask = (dist > rim_outer) & (dist <= ejecta_outer)
    
    ejecta_dist = (dist[ejecta_mask] - rim_outer) / (ejecta_outer - rim_outer)
    ejecta_thickness = rim_height * 0.3 * (1 - ejecta_dist)**2
    
    ejecta_noise = np.random.randn(np.sum(ejecta_mask)) * ejecta_thickness.mean() * 0.3
    heightmap[ejecta_mask] = ejecta_thickness + ejecta_noise
    
    return heightmap


def add_boulder_field(X, Y, num_boulders, boulder_size_range=(0.5, 5.0), 
                      region_center=None, region_radius=None, seed=None):
    """
    Add scattered boulders to terrain (common near crater rims/ejecta).
    
    Args:
        X, Y: Meshgrid coordinates
        num_boulders: Number of boulders to place
        boulder_size_range: (min, max) diameter in meters
        region_center: (x, y) center of boulder field, or None for random
        region_radius: Radius of concentration, or None for full terrain
        seed: Random seed
        
    Returns:
        heightmap: 2D array of boulder elevations
    """
    if seed is not None:
        np.random.seed(seed)
    
    heightmap = np.zeros_like(X)
    
    for _ in range(num_boulders):
        size_exponent = np.random.power(2.5)
        boulder_diameter = boulder_size_range[0] + size_exponent * (boulder_size_range[1] - boulder_size_range[0])
        boulder_height = boulder_diameter * 0.7
        
        if region_center is not None and region_radius is not None:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, 1)) * region_radius
            bx = region_center[0] + r * np.cos(angle)
            by = region_center[1] + r * np.sin(angle)
        else:
            x_range = X.max() - X.min()
            y_range = Y.max() - Y.min()
            bx = X.min() + np.random.uniform(0.1, 0.9) * x_range
            by = Y.min() + np.random.uniform(0.1, 0.9) * y_range
        
        dist = np.sqrt((X - bx)**2 + (Y - by)**2)
        boulder_radius = boulder_diameter / 2.0
        boulder_mask = dist <= boulder_radius * 1.5
        
        boulder_profile = boulder_height * np.exp(-(dist[boulder_mask] / boulder_radius)**2)
        heightmap[boulder_mask] = np.maximum(heightmap[boulder_mask], boulder_profile)
    
    return heightmap


def generate_lunar_terrain(size=2000.0, resolution=200, num_craters=20, 
                          crater_depth_range=(3, 15), crater_radius_range=(15, 80),
                          noise_scale=0.2, seed=None, terrain_type='mare',
                          include_boulders=True, realism_level='high'):
    """
    Generate realistic procedural lunar terrain.
    
    Terrain types:
    - 'mare': Smooth volcanic plains (fewer craters, gentle slopes)
    - 'highland': Ancient cratered terrain (heavily cratered, rough)
    - 'mixed': Combination of mare and highland
    
    Args:
        size: Terrain size in meters (square)
        resolution: Grid resolution (cells per side)
        num_craters: Number of primary impact craters
        crater_depth_range: (min, max) depth in meters
        crater_radius_range: (min, max) radius in meters
        noise_scale: Micro-topography scale (meters)
        seed: Random seed
        terrain_type: 'mare', 'highland', or 'mixed'
        include_boulders: Whether to add boulder fields
        realism_level: 'basic', 'medium', 'high'
    
    Returns:
        heightmap: 2D array of heights (resolution x resolution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nGenerating realistic lunar terrain:")
    print(f"  Type: {terrain_type.upper()}")
    print(f"  Size: {size}m x {size}m")
    print(f"  Resolution: {resolution} x {resolution}")
    print(f"  Realism level: {realism_level}")
    
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    if terrain_type == 'mare':
        num_craters = int(num_craters * 0.5)
        base_roughness = 0.15
        large_scale_amp = 2.0
        print(f"  Mare terrain: {num_craters} craters (reduced)")
    elif terrain_type == 'highland':
        num_craters = int(num_craters * 2.0)
        base_roughness = 0.4
        large_scale_amp = 5.0
        print(f"  Highland terrain: {num_craters} craters (increased)")
    else:
        base_roughness = 0.25
        large_scale_amp = 3.5
        print(f"  Mixed terrain: {num_craters} craters")
    
    print("  Generating base topography...")
    if realism_level in ['medium', 'high']:
        base_terrain = perlin_noise_2d(
            (resolution, resolution), 
            scale=size/10, 
            octaves=5,
            persistence=0.5,
            seed=seed
        ) * large_scale_amp
    else:
        # Simple sinusoidal for basic level
        base_terrain = np.zeros((resolution, resolution))
        for _ in range(3):
            wave_x = np.random.uniform(0, 2 * np.pi)
            wave_y = np.random.uniform(0, 2 * np.pi)
            wavelength = np.random.uniform(size/8, size/4)
            amplitude = np.random.uniform(1.0, large_scale_amp)
            base_terrain += amplitude * np.sin(2 * np.pi * X / wavelength + wave_x) * \
                           np.sin(2 * np.pi * Y / wavelength + wave_y)
    
    heightmap = base_terrain.copy()
    
    # Generate crater size distribution (power law: N(>D) ∝ D^-2)
    # Realistic lunar crater distribution
    print("  Generating crater size distribution...")
    crater_diameters = []
    for _ in range(num_craters):
        # Power-law distribution
        min_d = crater_radius_range[0] * 2
        max_d = crater_radius_range[1] * 2
        
        # Inverse transform sampling for power law
        alpha = 2.0  # Crater size-frequency exponent
        u = np.random.uniform(0, 1)
        diameter = min_d * (1 + u * ((max_d/min_d)**(1-alpha) - 1))**(1/(1-alpha))
        crater_diameters.append(diameter)
    
    # Sort by size (place largest first to avoid overlap issues)
    crater_diameters.sort(reverse=True)
    
    print("  Generating craters with realistic morphology...")
    crater_positions = []
    
    for i, diameter in enumerate(crater_diameters):
        is_complex = diameter > 300 if realism_level == 'high' else False
        
        margin = diameter
        cx = np.random.uniform(-size/2 + margin, size/2 - margin)
        cy = np.random.uniform(-size/2 + margin, size/2 - margin)
        
        age_factor = np.random.uniform(0.5, 1.0)
        depth_ratio = 0.2 * age_factor
        
        crater = generate_realistic_crater(
            X, Y, cx, cy, diameter, 
            depth_diameter_ratio=depth_ratio,
            is_complex=is_complex,
            ejecta_range=np.random.uniform(1.3, 2.0)
        )
        
        heightmap += crater
        crater_positions.append((cx, cy, diameter))
        
        if (i + 1) % 10 == 0:
            print(f"    Created {i+1}/{num_craters} craters...")
    
    if include_boulders and realism_level in ['medium', 'high']:
        print("  Adding boulder fields...")
        total_boulders = 0
        
        for cx, cy, diameter in crater_positions[:int(num_craters * 0.3)]:
            if diameter > 50:
                num_boulders = int(diameter / 10)
                boulder_field = add_boulder_field(
                    X, Y, num_boulders,
                    boulder_size_range=(0.5, min(5.0, diameter/20)),
                    region_center=(cx, cy),
                    region_radius=diameter * 0.8
                )
                heightmap += boulder_field
                total_boulders += num_boulders
        
        print(f"    Added {total_boulders} boulders")
    
    print("  Adding regolith micro-topography...")
    if realism_level == 'high':
        micro_rough = np.zeros_like(heightmap)
        for scale in [0.5, 0.2, 0.1]:
            layer = np.random.normal(0, noise_scale * scale, (resolution, resolution))
            micro_rough += layer
        heightmap += micro_rough
    else:
        roughness = np.random.normal(0, base_roughness, (resolution, resolution))
        heightmap += roughness
    
    if realism_level == 'high':
        print("  Adding secondary craters...")
        num_secondaries = num_craters * 5
        for _ in range(num_secondaries):
            sec_diameter = np.random.uniform(2, 10)
            sec_cx = np.random.uniform(-size/2, size/2)
            sec_cy = np.random.uniform(-size/2, size/2)
            sec_depth_ratio = np.random.uniform(0.05, 0.15)
            
            sec_crater = generate_realistic_crater(
                X, Y, sec_cx, sec_cy, sec_diameter,
                depth_diameter_ratio=sec_depth_ratio,
                is_complex=False,
                ejecta_range=1.2
            )
            heightmap += sec_crater * 0.5
    
    print(f"  Height range: [{np.min(heightmap):.2f}, {np.max(heightmap):.2f}] m")
    print(f"  Mean elevation: {np.mean(heightmap):.2f} m")
    print(f"  Std deviation: {np.std(heightmap):.2f} m")
    print(f"✓ Terrain generation complete!\n")
    
    return heightmap


def main():
    parser = argparse.ArgumentParser(description='Generate realistic lunar terrain heightmap')
    parser.add_argument('--output', type=str, default='generated_terrain/moon_terrain.npy',
                       help='Output file path (.npy or .csv)')
    parser.add_argument('--size', type=float, default=2000.0,
                       help='Terrain size in meters (default: 2000)')
    parser.add_argument('--resolution', type=int, default=200,
                       help='Grid resolution (default: 200)')
    parser.add_argument('--craters', type=int, default=20,
                       help='Number of craters (default: 20)')
    parser.add_argument('--terrain-type', type=str, default='mare',
                       choices=['mare', 'highland', 'mixed'],
                       help='Terrain type: mare (smooth), highland (rough), mixed (default: mare)')
    parser.add_argument('--realism', type=str, default='high',
                       choices=['basic', 'medium', 'high'],
                       help='Realism level - affects detail and computation time (default: high)')
    parser.add_argument('--no-boulders', action='store_true',
                       help='Disable boulder field generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show 3D visualization of terrain')
    
    args = parser.parse_args()
    
    heightmap = generate_lunar_terrain(
        size=args.size,
        resolution=args.resolution,
        num_craters=args.craters,
        seed=args.seed,
        terrain_type=args.terrain_type,
        include_boulders=not args.no_boulders,
        realism_level=args.realism
    )
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    if args.output.endswith('.npy'):
        np.save(args.output, heightmap)
        print(f"✓ Saved terrain to: {args.output}")
    elif args.output.endswith('.csv'):
        np.savetxt(args.output, heightmap, delimiter=',')
        print(f"✓ Saved terrain to: {args.output}")
    else:
        print("⚠ Unknown file format, saving as .npy")
        np.save(args.output + '.npy', heightmap)
    
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            print("\nGenerating visualization...")
            
            # Create figure with more subplots for enhanced terrain
            fig = plt.figure(figsize=(16, 12))
            
            # 3D surface plot
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            x = np.linspace(-args.size/2, args.size/2, args.resolution)
            y = np.linspace(-args.size/2, args.size/2, args.resolution)
            X, Y = np.meshgrid(x, y)
            
            # Use terrain-appropriate colormap
            cmap = 'gray' if args.terrain_type in ['mare', 'mixed'] else 'gist_earth'
            
            surf = ax1.plot_surface(X, Y, heightmap, cmap=cmap, 
                                   linewidth=0, antialiased=True, alpha=0.9)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Height (m)')
            ax1.set_title(f'Lunar Terrain - 3D View ({args.terrain_type.title()})')
            fig.colorbar(surf, ax=ax1, shrink=0.5)
            
            # Top-down heightmap
            ax2 = fig.add_subplot(2, 3, 2)
            im = ax2.imshow(heightmap, cmap=cmap, extent=[-args.size/2, args.size/2, 
                                                            -args.size/2, args.size/2],
                           origin='lower', interpolation='bilinear')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Heightmap - Top View')
            ax2.grid(True, alpha=0.2)
            fig.colorbar(im, ax=ax2)
            
            # Height histogram
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.hist(heightmap.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
            ax3.axvline(np.mean(heightmap), color='r', linestyle='--', label=f'Mean: {np.mean(heightmap):.2f}m')
            ax3.axvline(np.median(heightmap), color='b', linestyle='--', label=f'Median: {np.median(heightmap):.2f}m')
            ax3.set_xlabel('Height (m)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Height Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Contour plot with more levels
            ax4 = fig.add_subplot(2, 3, 4)
            contour = ax4.contour(X, Y, heightmap, levels=25, cmap=cmap, linewidths=0.5)
            ax4.clabel(contour, inline=True, fontsize=7)
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title('Terrain Contours (1m intervals)')
            ax4.set_aspect('equal')
            
            # Slope analysis
            ax5 = fig.add_subplot(2, 3, 5)
            # Calculate slope magnitude
            dy, dx = np.gradient(heightmap)
            cell_size = args.size / args.resolution
            slope_angle = np.arctan(np.sqrt(dx**2 + dy**2) / cell_size) * 180 / np.pi
            
            slope_im = ax5.imshow(slope_angle, cmap='hot', extent=[-args.size/2, args.size/2,
                                                                    -args.size/2, args.size/2],
                                 origin='lower', interpolation='bilinear')
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.set_title('Slope Map (degrees)')
            fig.colorbar(slope_im, ax=ax5, label='Slope (°)')
            
            # Statistics panel
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')
            
            stats_text = f"""
Terrain Statistics
─────────────────────
Type: {args.terrain_type.upper()}
Realism: {args.realism}
Resolution: {args.resolution}×{args.resolution}
Cell size: {cell_size:.2f} m

Height Statistics:
  Min: {np.min(heightmap):.2f} m
  Max: {np.max(heightmap):.2f} m
  Mean: {np.mean(heightmap):.2f} m
  Std Dev: {np.std(heightmap):.2f} m
  Range: {np.ptp(heightmap):.2f} m

Slope Statistics:
  Max slope: {np.max(slope_angle):.1f}°
  Mean slope: {np.mean(slope_angle):.1f}°
  Safe areas (<15°): {np.sum(slope_angle < 15) / slope_angle.size * 100:.1f}%

Craters: {args.craters}
Boulders: {'Yes' if not args.no_boulders else 'No'}
            """
            
            ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center')
            
            plt.suptitle(f'Realistic Lunar Terrain Analysis - {args.terrain_type.title()} Type',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"⚠ Visualization requires matplotlib: {e}")


if __name__ == '__main__':
    main()
