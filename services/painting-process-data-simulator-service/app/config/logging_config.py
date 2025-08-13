import os
import logging
import sys
from app.config.settings import settings

def setup_logging():
    """Sets up centralized logging."""
    # Ensure log directory exists  
    os.makedirs(settings.log_directory, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{settings.log_directory}/service.log")
        ]
    )
