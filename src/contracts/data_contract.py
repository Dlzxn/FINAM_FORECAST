from abc import ABC, abstractmethod
from typing import Any


class DataContract(ABC):
    @abstractmethod
    def __len__(self) -> None:
        pass

    def _load_data(self) -> None:
        pass

    def _save_data(self) -> None:
        pass

    def _reset_data(self) -> None:
        pass

    def get_loader(self) -> tuple[Any, Any]:
        pass