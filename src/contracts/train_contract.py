from abc import ABC, abstractmethod

class TrainContract(ABC):
    @abstractmethod
    def train(self):
        pass