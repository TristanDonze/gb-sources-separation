from src.pipeline import train
from src.run_manager import RunManager
from src.logger import setup_logging

if __name__ == "__main__":
    run_manager = RunManager()
    checkpoint_dir = run_manager.checkpoint_dir
    run_log_file = run_manager.run_log_file

    setup_logging(run_log_file)

    train(checkpoint_dir)