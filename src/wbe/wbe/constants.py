from pathlib import Path

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"

# Data processing
GROUP_VARS = ["sewershed_id", "sample_collect_date"]
