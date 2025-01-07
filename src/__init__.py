# Initialisation

from .data_processing import process_data
from .feature_eng import extract_features

# Define metadata
__version__ = "0.1.0"
__author__ = "Aditya Rajan"

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("CRISPR-ML package initialised.")
