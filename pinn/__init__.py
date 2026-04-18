# Legacy compatibility
import warnings

warnings.filterwarnings("default", category=DeprecationWarning, module="pinn")

# Old imports still work but warn
from . import problems_definitions, slope_limiters, training

warnings.warn(
    "Direct imports from pinn are deprecated. Use pinn.core, pinn.problems, pinn.experiments instead.",
    DeprecationWarning,
    stacklevel=2,
)
