#!/usr/bin/env python
"""Test runner for the NeuralChild project.

This script ensures that all required packages are installed and runs the test suite.
"""

import subprocess
import sys
import os
import importlib.util

# Required packages for testing
REQUIRED_PACKAGES = [
    "pytest",
    "torch",
    "pydantic",
    "numpy"
]

def check_and_install_packages():
    """Check if required packages are installed and install them if needed."""
    packages_to_install = []
    
    for package in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(package.replace('-', '_'))
        if spec is None:
            packages_to_install.append(package)
    
    if packages_to_install:
        print(f"Installing required packages: {', '.join(packages_to_install)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
        print("All required packages installed successfully!")
    else:
        print("All required packages are already installed.")

def main():
    """Main function to run the tests."""
    # Check and install required packages
    check_and_install_packages()
    
    # Get the absolute path to the tests directory
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    
    # Run the tests
    print("Running NeuralChild tests...")
    result = subprocess.call([sys.executable, "-m", "pytest", tests_dir, "-v"])
    
    return result

if __name__ == "__main__":
    sys.exit(main()) 