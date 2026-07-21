"""Autonomous frontier exploration public entry point."""
from echos_frontier_logic import *
from echos_frontier_runtime import *


if __name__ == "__main__":
    raise SystemExit(0 if verify_frontier_determinism() else 1)
