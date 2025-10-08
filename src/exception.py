import sys
"""
The sys module in Python provides various functions and variables 
that are used to manipulate different parts of the Python runtime environment...

The sys module in Python provides access to system-specific parameters and functions that interact closely with the Python runtime environment.

sys is a bridge between Python and the operating system/runtime.

"""

from src.logger import logging 


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    #exc_tb would provide the info on which document and line number rhe exception has occurred 
    error_message = 'Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):#Calling constructor class for parent constructor
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    def __str__(self): #defines how the object should look when represented as string. Used in print(object) or in str(object)
        return self.error_message

if __name__=='__main__':
    try :
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e,sys)


