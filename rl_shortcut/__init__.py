"""Course code. Import the env modules only after AppLauncher has started Isaac Sim."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "checkpoints"
MOTION_DIR = ROOT / "motions"


def shutdown(app):
    """Release the physics scene, then close Isaac Sim.

    With a SimulationContext still alive, app.close() waits forever (Isaac Sim 5.1, headless).
    """
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    if sim is not None:
        sim.clear_all_callbacks()
        sim.clear_instance()
    app.close()
