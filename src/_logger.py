import sys
import loguru

from loguru import logger

logger.remove(0)  # remove default logger sink
logger.add(sys.stderr, level="WARNING")  # initialize stderr sink with higher default level