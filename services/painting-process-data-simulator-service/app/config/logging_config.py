import os
import logging
import sys
from app.config.settings import settings

def setup_logging():
    """Sets up centralized logging."""
    # Ensure log directory exists  
    os.makedirs(settings.log_directory, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(f"{settings.log_directory}/service.log"))
    except OSError as e:
        # fallback to stdout-only logging if file handler can't be created
        logging.warning("FileHandler creation failed; using stdout-only logging. (%s)", e)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
