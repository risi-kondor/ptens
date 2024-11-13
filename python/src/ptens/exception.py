from abc import abstractmethod

class PtensException(Exception):
    """
    Parent Ptens exception.
    All Ptens exception inherit this class.
    """
    @abstractmethod
    def __str__(self) -> str:
        pass

class GraphCreationError(PtensException):

    def __init__(self, msg:str):
        self.msg = msg
    
    def __str__(self)->str:
        return self.msg
    
