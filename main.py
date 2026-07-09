import logging
import os
import sys

from src.pipeline import train
from src.run_manager import RunManager
from src.logger import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    run_manager = RunManager()
    checkpoint_dir = run_manager.checkpoint_dir
    run_log_file = run_manager.run_log_file

    shell_log_file = os.path.join(run_manager.run_dir, "shell.log")

    sys.stdout = open(shell_log_file, "a", buffering=1)
    sys.stderr = sys.stdout

    with open(os.path.join(run_manager.run_dir, "pid.txt"), "w") as f:
        f.write(str(os.getpid()))

    setup_logging(run_log_file)

    logger.info(f"Shell stdout/stderr redirected to {shell_log_file}")

    try:
        train(checkpoint_dir)
    except Exception:
        logger.exception("Training crashed with an unhandled exception.")
        raise
    else:
        logger.info("Training finished successfully.")