import logging

class Logger:
    _instance = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logging()
        return cls._instance

    def _setup_logging(self):
        """Configure logging with proper handlers and formatting"""
        root_logger = logging.getLogger()

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create console handler with proper formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)

        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def logger(self):
        """Get the logger"""
        return logging.getLogger(__name__)

# Create singleton instance
logger_instance = Logger()
logger = logger_instance.logger
