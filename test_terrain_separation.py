"""
test_terrain_separation.py
Test that terrain simulation has been properly separated into its own module
"""

import sys
import numpy as np


def test_terrain_module_import():
    """Test that LunarRegolithModel can be imported from terrain_simulation"""
    print("Test 1: Importing LunarRegolithModel from terrain_simulation...")
    try:
        from terrain_simulation import LunarRegolithModel
        print("  ✓ LunarRegolithModel successfully imported from terrain_simulation")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False


def test_terrain_functionality():
    """Test basic terrain functionality"""
    print("\nTest 2: Testing terrain model functionality...")
    try:
        from terrain_simulation import LunarRegolithModel
        
        # Create terrain
        terrain = LunarRegolithModel(size=200, resolution=50)
        print("  ✓ Terrain model created")
        
        # Generate procedural terrain
        terrain.generate_procedural_terrain(num_craters=3)
        print("  ✓ Procedural terrain generated")
        
        # Test height query
        height = terrain.get_height(10, 20)
        assert isinstance(height, (int, float)), "Height should be a number"
        print(f"  ✓ Height query works: {height:.2f} m")
        
        # Test normal query
        normal = terrain.get_terrain_normal(10, 20)
        assert len(normal) == 3, "Normal should be 3D vector"
        print(f"  ✓ Normal query works: {normal}")
        
        # Test properties
        props = terrain.get_terrain_properties(10, 20)
        assert 'friction_coeff' in props, "Should return friction coefficient"
        print(f"  ✓ Properties query works: friction={props['friction_coeff']:.2f}")
        
        # Test contact force
        force = terrain.compute_contact_force(
            position=[10, 20, height - 1],  # 1m below surface
            velocity=[0, 0, -1],
            contact_area=0.5
        )
        assert len(force) == 3, "Force should be 3D vector"
        print(f"  ✓ Contact force computation works: {force}")
        
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_terrain_file_loading():
    """Test terrain loading from file"""
    print("\nTest 3: Testing terrain file loading...")
    try:
        from terrain_simulation import LunarRegolithModel
        import os
        
        # Create a test terrain file
        test_terrain = np.random.randn(50, 50) * 5.0
        test_file = '/tmp/test_terrain.npy'
        np.save(test_file, test_terrain)
        print(f"  ✓ Created test terrain file: {test_file}")
        
        # Load terrain
        terrain = LunarRegolithModel(size=200, resolution=50)
        success = terrain.load_terrain_from_file(test_file)
        
        if success and terrain.terrain_loaded:
            print("  ✓ Terrain loaded from file successfully")
            
            # Verify slopes were computed
            assert terrain.slope_x is not None, "Slope X should be computed"
            assert terrain.slope_y is not None, "Slope Y should be computed"
            print("  ✓ Terrain slopes computed")
            
            # Cleanup
            os.remove(test_file)
            return True
        else:
            print("  ✗ Failed to load terrain from file")
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_independence():
    """Test that terrain_simulation is independent of other modules"""
    print("\nTest 4: Testing module independence...")
    try:
        # Test that we can import terrain_simulation without importing Basilisk
        import sys
        
        # Remove any Basilisk-related imports if they exist
        basilisk_modules = [key for key in sys.modules.keys() if 'Basilisk' in key]
        print(f"  Found {len(basilisk_modules)} Basilisk modules already loaded")
        
        # Import terrain_simulation
        from terrain_simulation import LunarRegolithModel
        
        # Check that we didn't import Basilisk as a dependency
        new_basilisk_modules = [key for key in sys.modules.keys() if 'Basilisk' in key]
        
        if len(new_basilisk_modules) == len(basilisk_modules):
            print("  ✓ terrain_simulation does not depend on Basilisk")
            return True
        else:
            print(f"  ✗ terrain_simulation imported Basilisk modules: {set(new_basilisk_modules) - set(basilisk_modules)}")
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("TERRAIN SIMULATION SEPARATION TESTS")
    print("="*60)
    
    tests = [
        test_terrain_module_import,
        test_terrain_functionality,
        test_terrain_file_loading,
        test_module_independence,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED! Terrain simulation successfully separated.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
