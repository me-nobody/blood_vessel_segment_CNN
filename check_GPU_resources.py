import torch
import os
import logging


logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")