"""Machine learning in manufacturing: real-plant vs. digital-twin training data for press-station fault detection."""

__version__ = "1.0.0"

from .detector import Evaluation, FaultDetector
from .schema import FAULTS, FEATURE_COLUMNS, SIGNALS, fault_name

__all__ = ["FAULTS", "FEATURE_COLUMNS", "SIGNALS", "Evaluation", "FaultDetector", "__version__", "fault_name"]
