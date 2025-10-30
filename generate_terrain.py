"""
generate_terrain.py
Utility to generate random lunar terrain heightmaps for the lander simulation

Usage:
    python generate_terrain.py --output generated_terrain/moon_terrain.npy --size 2000 --resolution 200 --craters 20
"""

import numpy as np
import os
import argparse


def generate_lunar_terrain(size=20000.0, resolution=200, num_craters=20, 
                          crater_depth_range=(3, 15), crater_radius_range=(15, 80),
                          noise_scale=0.2, seed=None):
    """
    Generate procedural lunar terrain with craters and surface roughness
    
    Args:
        size: Terrain size in meters (square)
        resolution: Grid resolution (cells per side)
        num_craters: Number of impact craters to generate
        crater_depth_range: (min, max) depth in meters
        crater_radius_range: (min, max) radius in meters
        noise_scale: Scale of small-scale surface roughness (meters)
        seed: Random seed for reproducibility
    
    Returns:
        heightmap: 2D numpy array of height values (resolution x resolution)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nGenerating lunar terrain:")
    print(f"  Size: {size}m x {size}m")
    print(f"  Resolution: {resolution} x {resolution}")
    print(f"  Craters: {num_craters}")
    
    # Create coordinate grid
    x = np.linspace(-size/2, size/2, resolution)
    y = np.linspace(-size/2, size/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Start with flat terrain
    heightmap = np.zeros((resolution, resolution))
    
    # Add craters (Gaussian-shaped depressions)
    print("  Generating craters...")
    for i in range(num_craters):
        cx = np.random.uniform(-size/2 + 100, size/2 - 100)
        cy = np.random.uniform(-size/2 + 100, size/2 - 100)
        depth = np.random.uniform(*crater_depth_range)
        radius = np.random.uniform(*crater_radius_range)
        
        # Gaussian crater shape
        dist_sq = (X - cx)**2 + (Y - cy)**2
        crater = -depth * np.exp(-dist_sq / (2 * radius**2))
        
        # Add raised rim (optional, realistic feature)
        rim_height = depth * 0.15  # Rim is 15% of crater depth
        rim = rim_height * np.exp(-dist_sq / (2 * (radius * 1.2)**2))
        rim -= rim_height * np.exp(-dist_sq / (2 * radius**2))
        
        heightmap += crater + rim
    
    # Add small-scale surface roughness
    print("  Adding surface roughness...")
    roughness = np.random.normal(0, noise_scale, (resolution, resolution))
    heightmap += roughness
    
    # Add some larger-scale undulations (low-frequency terrain)
    print("  Adding terrain undulations...")
    for _ in range(3):
        wave_x = np.random.uniform(0, 2 * np.pi)
        wave_y = np.random.uniform(0, 2 * np.pi)
        wavelength = np.random.uniform(size/8, size/4)
        amplitude = np.random.uniform(1.0, 3.0)
        
        undulation = amplitude * np.sin(2 * np.pi * X / wavelength + wave_x) * \
                     np.sin(2 * np.pi * Y / wavelength + wave_y)
        heightmap += undulation
    
    print(f"  Height range: [{np.min(heightmap):.2f}, {np.max(heightmap):.2f}] m")
    print(f"✓ Terrain generation complete!\n")
    
    return heightmap


def main():
    parser = argparse.ArgumentParser(description='Generate lunar terrain heightmap')
    parser.add_argument('--output', type=str, default='generated_terrain/moon_terrain.npy',
                       help='Output file path (.npy or .csv)')
    parser.add_argument('--size', type=float, default=2000.0,
                       help='Terrain size in meters (default: 2000)')
    parser.add_argument('--resolution', type=int, default=200,
                       help='Grid resolution (default: 200)')
    parser.add_argument('--craters', type=int, default=20,
                       help='Number of craters (default: 20)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show 3D visualization of terrain')
    
    args = parser.parse_args()
    
    # Generate terrain
    heightmap = generate_lunar_terrain(
        size=args.size,
        resolution=args.resolution,
        num_craters=args.craters,
        seed=args.seed
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save terrain
    if args.output.endswith('.npy'):
        np.save(args.output, heightmap)
        print(f"✓ Saved terrain to: {args.output}")
    elif args.output.endswith('.csv'):
        np.savetxt(args.output, heightmap, delimiter=',')
        print(f"✓ Saved terrain to: {args.output}")
    else:
        print("⚠ Unknown file format, saving as .npy")
        np.save(args.output + '.npy', heightmap)
    
    # Visualize if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            print("\nGenerating visualization...")
            
            # Create figure
            fig = plt.figure(figsize=(12, 10))
            
            # 3D surface plot
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            x = np.linspace(-args.size/2, args.size/2, args.resolution)
            y = np.linspace(-args.size/2, args.size/2, args.resolution)
            X, Y = np.meshgrid(x, y)
            surf = ax1.plot_surface(X, Y, heightmap, cmap='gray', 
                                   linewidth=0, antialiased=True)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Height (m)')
            ax1.set_title('Lunar Terrain - 3D View')
            fig.colorbar(surf, ax=ax1, shrink=0.5)
            
            # Top-down heightmap
            ax2 = fig.add_subplot(2, 2, 2)
            im = ax2.imshow(heightmap, cmap='gray', extent=[-args.size/2, args.size/2, 
                                                            -args.size/2, args.size/2],
                           origin='lower')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Heightmap - Top View')
            fig.colorbar(im, ax=ax2)
            
            # Height histogram
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.hist(heightmap.flatten(), bins=50, color='gray', edgecolor='black')
            ax3.set_xlabel('Height (m)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Height Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Contour plot
            ax4 = fig.add_subplot(2, 2, 4)
            contour = ax4.contour(X, Y, heightmap, levels=20, cmap='gray')
            ax4.clabel(contour, inline=True, fontsize=8)
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title('Terrain Contours')
            ax4.set_aspect('equal')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"⚠ Visualization requires matplotlib: {e}")


if __name__ == '__main__':
    main()
