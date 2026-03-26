import sys
from src.utils.logger import logger
 
 
def get_error_message(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] at line [{line_number}]: {str(error)}"
 
 
class LendingClubException(Exception):
    def __init__(self, error: Exception):
        self.error_message = get_error_message(error)
        logger.error(self.error_message)
        super().__init__(self.error_message)
 
    def __str__(self):
        return self.error_message