#!/usr/bin/env python3
"""
run_tests.py
Master test runner for AI Lunar Lander project

Runs all unit tests for critical components:
- common_utils: Utility functions and attitude conversions
- starship_constants: Configuration constants and calculations
- terrain_simulation: Terrain model and physics
- generate_terrain: Terrain generation functions
- lunar_lander_env: Environment configuration and reward logic
- unified_training: Curriculum and training setup

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k test_name # Run specific test pattern
"""

import sys
import unittest
import argparse
import os


def main():
    """Run all unit tests"""
    
    parser = argparse.ArgumentParser(
        description='Run unit tests for AI Lunar Lander project'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '-k', '--pattern',
        type=str,
        default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )
    parser.add_argument(
        '--module',
        type=str,
        help='Run tests for specific module (e.g., common_utils, terrain_simulation)'
    )
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first test failure'
    )
    
    args = parser.parse_args()
    
    # Configure test verbosity
    verbosity = 2 if args.verbose else 1
    
    # Print header
    print("=" * 80)
    print("AI LUNAR LANDER - UNIT TEST SUITE")
    print("=" * 80)
    print()
    
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Load tests based on arguments
    if args.module:
        # Load tests for specific module
        test_file = f'test_{args.module}.py'
        print(f"Running tests for module: {args.module}")
        print(f"Test file: {test_file}")
        print()
        
        try:
            suite = loader.loadTestsFromName(f'test_{args.module}')
        except Exception as e:
            print(f"Error loading tests for module '{args.module}': {e}")
            print(f"\nAvailable test modules:")
            print("  - common_utils")
            print("  - starship_constants")
            print("  - terrain_simulation")
            print("  - generate_terrain")
            print("  - lunar_lander_env")
            print("  - unified_training")
            return 1
    else:
        # Load all tests matching pattern
        print(f"Discovering tests with pattern: {args.pattern}")
        print()
        
        suite = loader.discover(
            start_dir=test_dir,
            pattern=args.pattern,
            top_level_dir=test_dir
        )
    
    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=args.failfast
    )
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)
    
    # Return exit code
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
