import logging
from datetime import datetime


class Logger:
    """
    This module provides a customizable logging utility with colorized console output.
    It logs messages to both a file (named with the current date) and the console.
    """
    def __init__(self):
        today_date = datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"logs/log-{today_date}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.log_file), logging.StreamHandler()],
        )

    @staticmethod
    def _colorize(text, color_code):
        """Helper function to add color to text."""
        return f"\033[{color_code}m{text}\033[0m"

    def log_success(self, message):
        colored_message = self._colorize(f"SUCCESS: {message}", "32")  # Green
        logging.info(colored_message)

    def log_fail(self, message):
        colored_message = self._colorize(f"FAIL: {message}", "31")  # Red
        logging.error(colored_message, exc_info=True)

    def log_info(self, message):
        colored_message = self._colorize(f"INFO: {message}", "37")  # White
        logging.info(colored_message)

    def log_warning(self, message):
        colored_message = self._colorize(f"WARNING: {message}", "33")  # Orange
        logging.warning(colored_message)

