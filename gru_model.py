"""Run the configured gru sensor forecasting benchmark."""
import sys

from pdm.forecasting.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(["--models", "gru", *sys.argv[1:]]))
