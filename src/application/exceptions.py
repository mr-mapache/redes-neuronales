class ExperimentNotFound(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ModelNotFound(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class StateNotFound(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class TaskNotSupported(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ExperimentAlreadyExists(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return "Experiment already exists"

class ModelNotSupported(Exception):
    def __init__(self, message: str):
        super().__init__(message)