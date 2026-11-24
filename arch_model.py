"""Run the configured arch sensor forecasting benchmark."""
import sys

from pdm.forecasting.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(["--models", "arch", *sys.argv[1:]]))
