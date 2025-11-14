"""
vizard_integration.py
Vizard visualization integration for Lunar Lander environment

This module provides Vizard connectivity for real-time visualization of RL training
episodes. It handles:
- vizInterface initialization and configuration
- Port management and connection settings
- Thread-safe message passing
- Automatic cleanup on errors

Usage:
    from vizard_integration import VizardManager
    
    viz = VizardManager(port=5556, enabled=True)
    viz.add_spacecraft(lander, "Starship_HLS")
    viz.render()  # Called each environment step
"""

import threading
import warnings
import os


class VizardManager:
    """
    Manages Vizard visualization for Basilisk simulations.
    
    Handles initialization, configuration, and safe cleanup of vizInterface.
    Provides thread-safe rendering for parallel environments.
    """
    
    def __init__(self, port=5556, enabled=True, name="LunarLander"):
        """
        Initialize Vizard manager.
        
        Args:
            port (int): Vizard connection port (default: 5556)
            enabled (bool): Enable Vizard (disable to save resources)
            name (str): Simulation name in Vizard
        """
        self.port = port
        self.enabled = enabled
        self.name = name
        self.viz = None
        self.lock = threading.Lock()
        self._initialized = False
        
        if enabled:
            self._initialize_vizard()
    
    def _initialize_vizard(self):
        """Initialize vizInterface with error handling."""
        if self._initialized:
            return
        
        try:
            from Basilisk.simulation import vizInterface
            
            self.viz = vizInterface.VizInterface()
            self.viz.pubPortNumber = str(self.port)
            
            self._initialized = True
            print(f"✓ Vizard initialized on port {self.port}")
            print(f"  → Connect with: localhost:{self.port}")
            
        except ImportError as e:
            warnings.warn(
                f"Vizard vizInterface not available: {e}\n"
                "Visualization disabled. Install Basilisk to enable."
            )
            self.enabled = False
        except Exception as e:
            warnings.warn(f"Failed to initialize Vizard: {e}")
            self.enabled = False
    
    def add_to_simulation(self, sim, task_name="task"):
        """
        Add vizInterface to Basilisk simulation task.
        
        Args:
            sim: Basilisk SimulationBaseClass instance
            task_name: Task name to add vizInterface to
        """
        if not self.enabled or self.viz is None:
            return
        
        try:
            with self.lock:
                sim.AddModelToTask(task_name, self.viz)
                print(f"✓ Vizard interface added to simulation task '{task_name}'")
        except Exception as e:
            warnings.warn(f"Failed to add vizInterface to task: {e}")
            self.enabled = False
    
    def render(self):
        """
        Trigger render (no-op if Vizard disabled).
        
        Note: In Basilisk, rendering happens automatically during
        ExecuteSimulation(). This method is here for API consistency.
        """
        if not self.enabled:
            return
        # Vizard updates happen during simulation steps
        pass
    
    def close(self):
        """Clean up Vizard resources."""
        if self.viz is not None:
            try:
                with self.lock:
                    del self.viz
            except Exception as e:
                warnings.warn(f"Error closing Vizard: {e}")
            finally:
                self.viz = None
                self._initialized = False
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


class VizardConfig:
    """Configuration helper for Vizard settings."""
    
    # Port settings
    DEFAULT_PORT = 5556
    PORT_RANGE = (5556, 5566)  # Alternative ports if default unavailable
    
    # Rendering settings
    DEFAULT_ENABLED = True
    AUTO_PORT_SEARCH = True
    
    # Performance settings
    FRAME_RATE = 10  # Hz
    UPDATE_FREQUENCY = 0.1  # seconds
    
    @classmethod
    def get_available_port(cls):
        """
        Find an available port for Vizard.
        
        Returns:
            int: Available port number, or DEFAULT_PORT if none found
        """
        if not cls.AUTO_PORT_SEARCH:
            return cls.DEFAULT_PORT
        
        import socket
        
        for port in range(*cls.PORT_RANGE):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        
        return cls.DEFAULT_PORT
