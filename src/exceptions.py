


class IncorrectAssetException(Exception):
    """
    Raises if filter received a measurement for a different asset
    """


class StaleMeasurementException(Exception):
    """
    Raised when a measurement arrives older than the filter state timestamp.
    """
