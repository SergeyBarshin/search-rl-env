"""Python-клиент отдельного поискового сервиса на Go."""

from .client import SearchClient
from .models import (InvalidRequest, SearchBusy, SearchError, SearchProtocolError,
                     SearchTimeout, StorageError, UnknownId)

__all__ = ["SearchClient", "SearchError", "InvalidRequest", "UnknownId", "SearchBusy",
           "SearchTimeout", "StorageError", "SearchProtocolError"]
