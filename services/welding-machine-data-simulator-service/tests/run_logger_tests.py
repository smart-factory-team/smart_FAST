#!/usr/bin/env python3
"""
Test runner for the AnomalyLogger tests.
This script ensures proper path setup for importing the modules under test.
"""

import sys
import os
import unittest

# Add the service root to Python path
service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, service_root)

if __name__ == '__main__':
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())