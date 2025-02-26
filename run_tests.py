#!/usr/bin/env python
"""
Run script for Neural Child tests.

This script provides a convenient way to run the test suite for the Neural Child project
with proper environment setup and reporting.
"""

import os
import sys
import logging
import subprocess
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_tests")

def check_dependencies():
    """Check if all required packages for testing are installed."""
    required_packages = [
        "pytest", "pytest-mock", "pytest-cov", "pytest-dash"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing_packages
            ])
            logger.info("All required packages installed successfully.")
        except subprocess.CalledProcessError:
            logger.error("Failed to install required packages.")
            logger.info("Please install them manually using: pip install -r requirements.txt")
            sys.exit(1)

def run_tests():
    """Run the Neural Child test suite."""
    logger.info("Running Neural Child tests...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the project root directory
    os.chdir(project_root)
    
    # Run pytest with coverage
    try:
        subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=mind", "--cov=mother", "--cov=core", "--cov=communication",
            "-v"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Tests stopped by user.")
    
if __name__ == "__main__":
    # Check if all required packages are installed
    logger.info("Checking required packages...")
    check_dependencies()
    
    # Run the tests
    run_tests() 