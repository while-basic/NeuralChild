"""Tests for the Neural Child Dashboard."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dashboard app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neural_child_dashboard", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neural-child-dashboard.py")
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)

class TestDashboard:
    """Test suite for the Neural Child Dashboard."""
    
    def test_dashboard_initialization(self):
        """Test that the dashboard initializes correctly."""
        # Check that the app exists
        assert hasattr(dashboard, 'app')
        
        # Check that the mind and mother exist
        assert hasattr(dashboard, 'mind')
        assert hasattr(dashboard, 'mother')
        
        # Check that the layout exists
        assert hasattr(dashboard.app, 'layout')
    
    def test_dashboard_callbacks(self):
        """Test that the dashboard callbacks are registered."""
        # Check that the callbacks are registered
        assert len(dashboard.app.callback_map) > 0
    
    def test_dashboard_components(self):
        """Test that the dashboard has the expected components."""
        # Check that the layout has children
        assert hasattr(dashboard.app.layout, 'children')
        assert len(dashboard.app.layout.children) > 0
        
        # Check for specific components
        component_ids = []
        
        def collect_ids(component):
            if hasattr(component, 'id') and component.id is not None:
                component_ids.append(component.id)
            if hasattr(component, 'children') and component.children is not None:
                if isinstance(component.children, list):
                    for child in component.children:
                        collect_ids(child)
                else:
                    collect_ids(component.children)
        
        collect_ids(dashboard.app.layout)
        
        # Print all component IDs for debugging
        print("Found component IDs:", component_ids)
        
        # Check for essential components - using more general checks
        assert any('consciousness' in str(id).lower() for id in component_ids), "No consciousness component found"
        assert any('emotion' in str(id).lower() for id in component_ids), "No emotions component found"
        assert any('development' in str(id).lower() for id in component_ids), "No developmental stage component found"
        assert any('energy' in str(id).lower() for id in component_ids), "No energy level component found"
        assert any('input' in str(id).lower() for id in component_ids), "No input component found"
        assert any('button' in str(id).lower() for id in component_ids), "No button component found" 