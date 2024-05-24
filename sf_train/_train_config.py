class _TrainConfig:
    def __init__(
        self                     ,
        chunk_sec   =      0.5   ,
        batch_size  =     16     ,
        initial_lr  =      0.005 ,
        lr_decay    =      0.0001,
        lr_factor   =      0.5   ,
        lr_patience =      5     ,
        init_len    =    200     ,
        up_fr       =   1000     ,
        vali_chunk  = 100000):
        self.chunk_sec   = chunk_sec
        self.batch_size  = batch_size
        self.initial_lr  = initial_lr
        self.lr_decay    = lr_decay
        self.lr_factor   = lr_factor
        self.lr_patience = lr_patience
        self.init_len    = init_len
        self.up_fr       = up_fr
        self.vali_chunk  = vali_chunk
