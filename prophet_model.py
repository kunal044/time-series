"""Run the configured prophet sensor forecasting benchmark."""
import sys

from pdm.forecasting.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(["--models", "prophet", *sys.argv[1:]]))
