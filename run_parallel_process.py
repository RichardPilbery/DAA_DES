from air_ambulance_des.des_parallel_process import (
    removeExistingResults,
    parallelProcessJoblib,
)
from datetime import datetime

if __name__ == "__main__":
    removeExistingResults()
    # parallelProcessJoblib(1, (1*365*24*60), (0*60), datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"), False, False, 1.0, 1.0, True)
    parallelProcessJoblib(
        12,
        (2 * 365 * 24 * 60),
        (0 * 60),
        datetime.strptime("2023-01-01 05:00:00", "%Y-%m-%d %H:%M:%S"),
        False,
        False,
        1.0,
        1.0,
    )

# Testing ----------
# python des_parallel_process.py
