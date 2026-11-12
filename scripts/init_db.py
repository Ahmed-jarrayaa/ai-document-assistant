import sys
from pathlib import Path

# Allow "from app...." imports when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.database.schema import ensure_schema

if __name__ == "__main__":
    ensure_schema()
