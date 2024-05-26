class _TrainConfig:
    def __init__(
        self                 ,
        compile     = True   ,
        batch_size  = 16     ,
        initial_lr  =  0.005 ,
        lr_decay    =  0.0001,
        lr_factor   =  0.5   ,
        lr_patience =  5     ,
        seed        =  0):
        self.compile     = compile
        self.batch_size  = batch_size
        self.initial_lr  = initial_lr
        self.lr_decay    = lr_decay
        self.lr_factor   = lr_factor
        self.lr_patience = lr_patience
        self.seed        = seed
