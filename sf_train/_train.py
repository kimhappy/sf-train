import time
import numpy as np
import torch
from ._model        import _Model
from ._model_config import _ModelConfig
from ._train_config import _TrainConfig
from ._data         import _Data
from ._loss         import _MixLoss, _ESRLoss, _DCLoss

def _best_device() -> torch.device:
    print('Detecting device...', end = '')

    if torch.cuda.is_available():
        print('CUDA available')
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print('MPS available')
        return torch.device('mps')
    else:
        print('CPU available')
        return torch.device('cpu')

class _TrainResult:
    def __init__(
        self               ,
        loss        : float,
        time_elapsed: float):
        self.loss         = loss
        self.time_elapsed = time_elapsed

class _ValiResult:
    def __init__(
        self                    ,
        loss        : float     ,
        processed   : np.ndarray,
        time_elapsed: float):
        self.loss         = loss
        self.processed    = processed
        self.time_elapsed = time_elapsed

class _Train:
    def __init__(
        self                      ,
        model_config: _ModelConfig,
        train_config: _TrainConfig,
        data        : _Data       ,
        device      : torch.device = None):
        assert model_config.input_size == data.num_params + 1

        self.model_config = model_config
        self.train_config = train_config
        self.network      = _Model(
            self.model_config.input_size ,
            self.model_config.output_size,
            self.model_config.hidden_size)
        self.optimizer    = torch.optim.Adam(
            self.network.parameters()                  ,
            lr           = self.train_config.initial_lr,
            weight_decay = self.train_config.lr_decay)
        self.scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer                        ,
            'min'                                 ,
            factor   = self.train_config.lr_factor,
            patience = self.train_config.lr_patience)
        self.loss_fn      = _MixLoss([(_ESRLoss(), 0.75), (_DCLoss(), 0.25)])
        self.device       = device if device else _best_device()
        self.network.to(self.device)

        self.chunk_len = int(self.train_config.chunk_sec * self.model_config.sample_rate)
        self.tis, self.tos, self.vis, self.vos = data._chunkify(self.chunk_len)
        self.tis = self.tis.to(self.device)
        self.tos = self.tos.to(self.device)
        self.vis = self.vis.to(self.device)
        self.vos = self.vos.to(self.device)

    def train(self) -> _TrainResult:
        begin_time = time.time()

        num_chunks     = self.tis.shape[ 0 ]
        num_chunks_cut = num_chunks - (num_chunks % self.train_config.batch_size)
        shuffle        = (torch.randperm(num_chunks)[ : num_chunks_cut ] - 1).reshape(-1, self.train_config.batch_size)
        ep_loss        = 0

        for batch_idxs in shuffle:
            input_batches  = self.tis[ batch_idxs, :, : ]
            target_batches = self.tos[ batch_idxs, :    ]

            self.network(input_batches[ :, 0: self.train_config.init_len, : ])
            self.network.zero_grad()

            num_iter   = (input_batches.shape[ 1 ] - self.train_config.init_len) // self.train_config.up_fr
            batch_loss = 0

            for begin in range(self.train_config.init_len, input_batches.shape[ 1 ], self.train_config.up_fr):
                input_batch   = input_batches [ :, begin: begin + self.train_config.up_fr, : ]
                target_batch  = target_batches[ :, begin: begin + self.train_config.up_fr    ]
                output_batch  = self.network(input_batch)
                loss          = self.loss_fn(output_batch, target_batch)

                loss          .backward()
                self.optimizer.step()
                self.network  .detach_hidden()
                self.network  .zero_grad()

                batch_loss += loss

            ep_loss += batch_loss / num_iter
            self.network.hidden = None

        end_time = time.time()
        return _TrainResult(ep_loss / shuffle.shape[ 0 ], end_time - begin_time)

    def vali(self, reduce_lr = True) -> _ValiResult:
        begin_time = time.time()

        with torch.no_grad():
            output     = torch.empty_like(self.vos)
            num_chunks = self.vos.shape[ 0 ] // self.train_config.vali_chunk
            remainder  = self.vos.shape[ 0 ] %  self.train_config.vali_chunk

            for l in range(num_chunks):
                begin                = l     * self.train_config.vali_chunk
                end                  = begin + self.train_config.vali_chunk
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.network(input_chunk)
                output[ begin: end ] = output_chunk
                self.network.detach_hidden()

            if remainder != 0:
                begin                = num_chunks * self.train_config.vali_chunk
                end                  = begin + remainder
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.network(input_chunk)
                output[ begin: end ] = output_chunk
                self.network.detach_hidden()

            self.network.hidden = None
            loss = self.loss_fn(output, self.vos)

        end_time = time.time()

        if reduce_lr:
            self.scheduler.step(loss)

        return _ValiResult(loss.item(), output.cpu().numpy(), end_time - begin_time)

    def state_dict(self) -> dict:
        return dict(self.network.state_dict())
