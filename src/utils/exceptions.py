"""Custom exceptions for NFL projections."""


class NFLProjectionsError(Exception):
    """Base exception for NFL projections."""
    pass


class DataLoadError(NFLProjectionsError):
    """Raised when data loading fails."""
    pass


class InsufficientDataError(NFLProjectionsError):
    """Raised when there's not enough historical data."""
    pass


class ModelNotTrainedError(NFLProjectionsError):
    """Raised when trying to use untrained models."""
    pass


class InvalidInputError(NFLProjectionsError):
    """Raised when user input is invalid."""
    pass