from typing import List

class _ModelConfig:
    def __init__(self, name: str, params: List[str]):
        self.name        = name
        self.params      = params
        assert len(params) <= 4
        self.hidden_size = 48
