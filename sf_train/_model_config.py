class _ModelConfig:
    def __init__(
        self               ,
        sample_rate = 48000,
        input_size  =     1,
        output_size =     1,
        hidden_size =    64):
        self.sample_rate = sample_rate
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
