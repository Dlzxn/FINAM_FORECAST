from abc import ABC, abstractmethod

class DataContract(ABC):
    @abstractmethod
    def __len__(self) -> None:
        pass

    def _load_data(self) -> None:
        pass

    def _save_data(self) -> None:
        pass