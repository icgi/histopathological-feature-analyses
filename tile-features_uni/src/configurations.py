from typing import Optional


class Config:

    def __init__(self, gpu: Optional[int] = None):
        self.batch_size = 32
        self.num_workers = 8
        self.device = "cpu" if gpu is None else f"cuda:{gpu}"
