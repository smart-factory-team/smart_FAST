import logging
from app.config.settings import settings

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulator.log'),
            logging.StreamHandler()
        ]
    )