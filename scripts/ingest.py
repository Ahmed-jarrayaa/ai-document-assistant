import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.monitoring.telemetry import flush_telemetry, setup_telemetry

setup_telemetry()

from app.services.ingestion import ingest_pdf

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts\\ingest.py documents\\your_file.pdf")
        sys.exit(1)
    t0 = time.perf_counter()
    result = ingest_pdf(sys.argv[1])
    print(f"{result} in {time.perf_counter() - t0:.1f}s")
    flush_telemetry()