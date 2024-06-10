import time
import struct
import numpy as np
import torch
from ._lstm_model   import _LSTMModel
from ._model_config import _ModelConfig
from ._train_config import _TrainConfig
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

class _Trainer:
    VERSION          =      0
    SAMPLE_RATE      =  48000
    TRAIN_INIT_FRAME =    200
    TRAIN_FRAME      =   1000
    VALI_FRAME       = 100000
    CHUNK_SEC        =      0.5

    def __init__(
        self                      ,
        model_config: _ModelConfig,
        train_config: _TrainConfig,
        device      : torch.device = None):
        # Seed setting
        torch.manual_seed(train_config.seed)

        # Model related
        self.name        = model_config.name
        self.params      = model_config.params
        self.hidden_size = model_config.hidden_size

        # Training related
        self.batch_size = train_config.batch_size

        # Model, optimizer, scheduler, loss function, device
        self.network   = _LSTMModel(
            len(self.params) + 1,
            1                   ,
            self.hidden_size)
        self.optimizer = torch.optim.Adam(
            self.network.parameters()             ,
            lr           = train_config.initial_lr,
            weight_decay = train_config.lr_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer                   ,
            'min'                            ,
            factor   = train_config.lr_factor,
            patience = train_config.lr_patience)
        self.loss_fn   = _MixLoss([(_ESRLoss(), 0.75), (_DCLoss(), 0.25)])
        self.device    = device if device else _best_device()
        self.network.to(self.device)

        # Data related
        self.chunk_len = int(_Trainer.CHUNK_SEC * _Trainer.SAMPLE_RATE)
        self.tis       = torch.empty(0, self.chunk_len, len(self.params) + 1).to(self.device)
        self.tos       = torch.empty(0, self.chunk_len                      ).to(self.device)
        self.vis       = torch.empty(0,                 len(self.params) + 1).to(self.device)
        self.vos       = torch.empty(0                                      ).to(self.device)

        # Compile model
        if train_config.compile:
            self.network = torch.compile(self.network)

    def _merge_params(input: np.ndarray, params: np.ndarray) -> np.ndarray:
        expanded = np.expand_dims(input, axis = 1)
        tiled    = np.tile(params, (input.size, 1))
        ret      = np.concatenate((expanded, tiled), axis = 1)
        return ret

    def add_train(self, input: np.ndarray, output: np.ndarray, params: np.ndarray):
        input  = np.array(input , dtype = np.float32)
        output = np.array(output, dtype = np.float32)
        params = np.array(params, dtype = np.float32)

        assert input .ndim == 1
        assert output.ndim == 1
        assert input .size == output.size
        assert input .size >  0
        assert params.ndim == 1
        assert params.size == len(self.params)

        num_chunks = input.shape[ 0 ] // self.chunk_len
        len_cut    = num_chunks       *  self.chunk_len

        input  = torch.from_numpy(_Trainer._merge_params(input , params)[ :len_cut ].reshape(num_chunks, self.chunk_len, -1)).to(self.device)
        output = torch.from_numpy(                       output         [ :len_cut ].reshape(num_chunks, self.chunk_len    )).to(self.device)

        self.tis = torch.cat((self.tis, input ))
        self.tos = torch.cat((self.tos, output))

    def add_vali(self, input: np.ndarray, output: np.ndarray, params: np.ndarray):
        input  = np.array(input , dtype = np.float32)
        output = np.array(output, dtype = np.float32)
        params = np.array(params, dtype = np.float32)

        assert input .ndim == 1
        assert output.ndim == 1
        assert input .size == output.size
        assert input .size >  0
        assert params.ndim == 1
        assert params.size == len(self.params)

        input  = torch.from_numpy(_Trainer._merge_params(input , params)).to(self.device)
        output = torch.from_numpy(                       output         ).to(self.device)

        self.vis = torch.cat((self.vis, input ))
        self.vos = torch.cat((self.vos, output))

    def train(self) -> _TrainResult:
        begin_time = time.time()

        # Prepare shuffle
        num_chunks     = self.tis.shape[ 0 ]
        num_chunks_cut = num_chunks - (num_chunks % self.batch_size)
        shuffle        = (torch.randperm(num_chunks)[ : num_chunks_cut ] - 1).reshape(-1, self.batch_size)
        ep_loss        = 0

        for batch_idxs in shuffle:
            # Prepare batch
            input_batch  = self.tis[ batch_idxs, :, : ]
            target_batch = self.tos[ batch_idxs, :    ]

            # Reset hidden state
            self.network.reset_hidden()
            self.network(input_batch[ :, 0: _Trainer.TRAIN_INIT_FRAME, : ])
            self.network.zero_grad()

            # Training
            batch_loss = 0

            for begin in range(_Trainer.TRAIN_INIT_FRAME, input_batch.shape[ 1 ], _Trainer.TRAIN_FRAME):
                # Get frame
                input_frame  = input_batch [ :, begin: begin + _Trainer.TRAIN_FRAME, : ]
                target_frame = target_batch[ :, begin: begin + _Trainer.TRAIN_FRAME    ]

                # Forward pass
                output_batch = self.network(input_frame)
                loss         = self.loss_fn(output_batch, target_frame)

                # Backpropagation
                loss          .backward()
                self.optimizer.step()
                self.network  .detach_hidden()
                self.network  .zero_grad()
                batch_loss += loss

            # Calculate loss
            num_iter  = (input_batch.shape[ 1 ] - _Trainer.TRAIN_INIT_FRAME) // _Trainer.TRAIN_FRAME
            ep_loss  += batch_loss / num_iter

        end_time = time.time()
        return _TrainResult(ep_loss / shuffle.shape[ 0 ], end_time - begin_time)

    def vali(self) -> _ValiResult:
        begin_time = time.time()

        with torch.no_grad():
            # Prepare output
            output     = torch.empty_like(self.vos)
            num_frames = self.vos.shape[ 0 ] // _Trainer.VALI_FRAME
            remainder  = self.vos.shape[ 0 ] %  _Trainer.VALI_FRAME
            self.network.reset_hidden()

            # Process frames
            for l in range(num_frames):
                begin                = l     * _Trainer.VALI_FRAME
                end                  = begin + _Trainer.VALI_FRAME
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.network(input_chunk)
                output[ begin: end ] = output_chunk
                self.network.detach_hidden()

            # Process remainder
            if remainder != 0:
                begin                = num_frames * _Trainer.VALI_FRAME
                end                  = begin + remainder
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.network(input_chunk)
                output[ begin: end ] = output_chunk
                self.network.detach_hidden()

            # Calculate loss
            loss = self.loss_fn(output, self.vos)

        self.scheduler.step(loss)
        end_time = time.time()

        return _ValiResult(loss.item(), output.cpu().numpy(), end_time - begin_time)

    def save(self, path: str):
        info_bytes = struct.pack('ii', _Trainer.VERSION, len(self.params))

        with open(path, 'wb') as f:
            f.write(info_bytes)

            for param in self.params:
                bs = param.encode('utf-8')
                f.write(struct.pack('i', len(bs)))
                f.write(bs)

            self.network.params().tofile(f)
