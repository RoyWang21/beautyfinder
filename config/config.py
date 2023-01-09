# config.py
import logging
import sys
from pathlib import Path
import logging.config
#import mlflow
#from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
LOGS_DIR = Path(BASE_DIR, "logs")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
#mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# Assets
#PROJECTS_URL = (
#    "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
#)
#TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"


#ACCEPTED_TAGS = ["natural-language-processing", "computer-vision", "mlops", "graph-learning"]

#NUM_LAYERS = 1
#LEARNING_RATE = 1e-4
#PATIENCE = 10
#num_epochs = 50

# set up loggings
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
#logger.handlers[0] = RichHandler(markup=True)  # pretty formatting
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Sample messages (note that we use configured `logger` now)
# logger.debug("Used for debugging your code.")
# logger.info("Informative messages from your code.")
# logger.warning("Everything works but there is something to be aware of.")
# logger.error("There's been a mistake with the process.")
# logger.critical("There is something terribly wrong and process may terminate.")

