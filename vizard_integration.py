# vizard_integration.py
# Defensive VizardManager for Basilisk vizInterface.
# Replace your existing vizard_integration.py with this file.

import threading
import warnings
import socket
import time
import contextlib

class VizardManager:
    """
    Robust Vizard manager that:
      - chooses/validates a port
      - initializes Basilisk vizInterface defensively (many attribute fallbacks)
      - adds viz model to a Basilisk task
      - exposes simple diagnostics for port/listen state
    """
    def __init__(self, port=None, enabled=True, name="LunarLander", auto_search=True):
        self.requested_port = int(port) if port is not None else None
        self.enabled = enabled
        self.name = name
        self.viz = None
        self.lock = threading.Lock()
        self._initialized = False
        self.auto_search = auto_search

        if self.enabled:
            self._initialize_vizard()

    # -------------------------
    # Port utilities / probing
    # -------------------------
    @staticmethod
    def _is_port_free(port, host='127.0.0.1'):
        """Return True if port is free on host (TCP)."""
        try:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, int(port)))
                return True
        except OSError:
            return False

    @staticmethod
    def _find_free_port(start=5570, end=5580):
        """Return first free port in inclusive range, or None."""
        for p in range(start, end + 1):
            if VizardManager._is_port_free(p):
                return p
        return None

    # -------------------------
    # Initialization
    # -------------------------
    def _initialize_vizard(self):
        """Attempt to initialize Basilisk vizInterface with many fallbacks and diagnostics."""
        if self._initialized:
            return

        try:
            # Import-only when needed (keeps module lightweight)
            from Basilisk.simulation import vizInterface as _vizmod
        except Exception as e:
            warnings.warn(f"[VizardManager] Basilisk.vizInterface import failed: {e}\n"
                          "➡ Make sure your PYTHONPATH points to Basilisk/dist3 and that Basilisk is built.")
            self.enabled = False
            return

        # Decide port
        port = self.requested_port
        if port is None and self.auto_search:
            port = self._find_free_port(start=5570, end=5585) or 5570
        elif port is None:
            port = 5570

        # Check whether port is free at OS level
        port_free = self._is_port_free(port)
        if not port_free:
            print(f"[VizardManager] Warning: Port {port} is NOT free on localhost.")
            print("  → Another process may be using it (check netstat). Trying to proceed anyway.")

        # Create vizInterface object
        try:
            viz = _vizmod.VizInterface()
        except Exception as e:
            warnings.warn(f"[VizardManager] Failed to construct VizInterface(): {e}")
            self.enabled = False
            return

        # Defensive attribute / method setting:
        # try several plausible attribute/method names (historical variations)
        def safe_set(obj, names, value, call=False):
            """
            Try each name in names:
             - if attribute exists, set it
             - if a method exists and call=True, call it with value
             - return True if something was set/called
            """
            for n in names:
                if hasattr(obj, n):
                    attr = getattr(obj, n)
                    # if callable and call==True, call it
                    if call and callable(attr):
                        try:
                            attr(value)
                            return True
                        except Exception:
                            continue
                    # else try set attribute (if possible)
                    try:
                        setattr(obj, n, value)
                        return True
                    except Exception:
                        # try calling if callable without call flag (rare)
                        try:
                            if callable(attr):
                                attr(value)
                                return True
                        except Exception:
                            continue
            return False

        # Set port in multiple possible ways
        port_set = safe_set(viz, ['pubPortNumber', 'pub_port', 'pubPort', 'pubPortNumberStr', 'pubPortStr'], str(port))
        if not port_set:
            # sometimes an integer is expected
            port_set = safe_set(viz, ['pubPortNumber', 'pub_port', 'pubPort'], port)
        if not port_set:
            # try a setter method if exists
            port_set = safe_set(viz, ['setPubPort', 'setPubPortNumber'], port, call=True)

        # Enable Vizard flag in multiple ways
        enabled_set = safe_set(viz, ['UseVizard', 'useVizard', 'useVizardFlag'], True)
        if not enabled_set:
            enabled_set = safe_set(viz, ['setUseVizard', 'EnableVizard', 'enableVizard'], True, call=True)

        # Some releases expect an explicit "connectToVizard" attr (but it's often readonly) — don't force it
        if hasattr(viz, 'connectToVizard'):
            # do not set if it's protected (can raise ValueError)
            try:
                setattr(viz, 'connectToVizard', True)
            except Exception:
                # ignore, it's often intentionally protected
                pass

        # Keep the object and mark initialized
        self.viz = viz
        self._initialized = True
        self.port = port

        # Print diagnostics
        print(f"✓ Vizard initialized (object created).")
        print(f"  Requested port: {self.requested_port or '(auto)'}  →  Using port: {self.port}")
        print("  Attributes set:")
        print(f"    pubPort-like set? {'yes' if port_set else 'no'}")
        print(f"    UseVizard-like set? {'yes' if enabled_set else 'no'}")
        print("  Tip: If Vizard times out, check netstat and firewall (see instructions).")

        # Quick OS-level listen check: If vizInterface opened a socket immediately it will consume port.
        # Wait briefly and poll port status to see if it is bound by this process.
        time.sleep(0.15)
        if not self._is_port_free(self.port):
            print(f"  Port {self.port} appears to be in use (listening) — good (expected if Viz opened it).")
        else:
            print(f"  Port {self.port} still free after init — vizInterface may not have opened the socket yet.")

    # -------------------------
    # Add to simulation
    # -------------------------
    def add_to_simulation(self, sim, task_name="task"):
        """
        Add vizInterface to Basilisk simulation task.
        """
        if not self.enabled or self.viz is None:
            print("[VizardManager] Viz disabled or not initialized; skipping add_to_simulation.")
            return

        try:
            with self.lock:
                # Basilisk expects sim.AddModelToTask(taskName, model)
                sim.AddModelToTask(task_name, self.viz)
                print(f"✓ Vizard interface added to simulation task '{task_name}'")
        except Exception as e:
            warnings.warn(f"[VizardManager] Failed to add vizInterface to task: {e}")
            self.enabled = False

    def render(self):
        """Placeholder for API compatibility. Basilisk handles rendering during ExecuteSimulation()."""
        if not self.enabled:
            return
        # Optionally call a viz update function if present
        try:
            if hasattr(self.viz, 'render') and callable(self.viz.render):
                self.viz.render()
        except Exception:
            pass

    def close(self):
        """Attempt to clean up the VizInterface object safely."""
        if self.viz is not None:
            try:
                with self.lock:
                    # call a shutdown/close if available
                    if hasattr(self.viz, 'shutdown') and callable(self.viz.shutdown):
                        try:
                            self.viz.shutdown()
                        except Exception:
                            pass
                    if hasattr(self.viz, 'Close') and callable(self.viz.Close):
                        try:
                            self.viz.Close()
                        except Exception:
                            pass
                    # best effort delete
                    del self.viz
            except Exception as e:
                warnings.warn(f"[VizardManager] Error closing Vizard: {e}")
            finally:
                self.viz = None
                self._initialized = False

    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# ---------------------------------------
# Simple helper to show local port status
# ---------------------------------------
def print_local_port_status(port):
    """
    Print very quick local hints for the given port.
    (Non-admin; uses bind-test only.)
    """
    free = VizardManager._is_port_free(port)
    if free:
        print(f"[port status] {port} appears free (no listener detected by bind test).")
    else:
        print(f"[port status] {port} appears IN USE (another process is listening).")
