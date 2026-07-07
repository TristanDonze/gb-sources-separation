import logging
from rich.logging import RichHandler

def setup_logging(run_log_file, level=logging.INFO):
    """Configure logging for the entire project."""
    
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = RichHandler(
        rich_tracebacks=True, 
        markup=True, 
        show_path=False
    )
    console_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)