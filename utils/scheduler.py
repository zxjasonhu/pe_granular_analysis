"""
Scheduler functions or Classes
"""


class ConstantLossWeightScheduler:
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def get_weight(self, epoch):
        return self.weight

    def __repr__(self):
        return f"ConstantLossWeightScheduler(weight={self.weight})"


class LinearLossWeightScheduler:
    def __init__(
        self,
        start_epoch: int = 0,
        end_epoch: int = 30,
        start_weight: float = 0.05,
        end_weight: float = 1.0,
    ):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_weight = start_weight
        self.end_weight = end_weight

    def get_weight(self, epoch):
        if epoch < self.start_epoch:
            return self.start_weight
        if epoch > self.end_epoch:
            return self.end_weight
        if self.start_epoch == self.end_epoch:
            return self.end_weight
        return self.start_weight + (self.end_weight - self.start_weight) * (
            epoch - self.start_epoch
        ) / (self.end_epoch - self.start_epoch)

    def __repr__(self):
        return f"LinearLossWeightScheduler(start_epoch={self.start_epoch}, end_epoch={self.end_epoch}, start_weight={self.start_weight}, end_weight={self.end_weight})"


if __name__ == "__main__":
    pass
    start_epoch = 50
    end_epoch = 50
    start_weight = 0.05
    end_weight = 0.95
    scheduler = LinearLossWeightScheduler(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_weight=start_weight,
        end_weight=end_weight,
    )
    for epoch in range(60):
        print(epoch, scheduler.get_weight(epoch))
