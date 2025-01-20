# errors.py

class MissingDefinitionError(Exception):
    """Exception raised when no definition is found in the dataset."""
    pass

class MissingExampleError(Exception):
    """Exception raised when no example is found in the dataset."""
    pass

class MissingExampleSummaryError(Exception):
    """Exception raised when no summary for an example is found in the dataset."""
    pass

class MissingExampleScoreError(Exception):
    """Exception raised when no score for an example is found in the dataset."""
    pass

class MissingExampleExplanationError(Exception):
    """Exception raised when no explanation for an example is found in the dataset."""
    pass

class DataValidationError(Exception):
    """Exception raised for errors in data validation."""
    pass

# Additional custom exceptions can be added here as needed.
