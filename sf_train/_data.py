import numpy as np
import torch

class _Data:
    def __init__(self, num_params: int):
        self.num_params    = num_params
        self.train_inputs  = []
        self.train_outputs = []
        self.vali_inputs   = []
        self.vali_outputs  = []

    def _merge(input: np.ndarray, param: np.ndarray) -> np.ndarray:
        expanded = np.expand_dims(input, axis = 1)
        tiled    = np.tile(param, (input.size, 1))
        ret      = np.concatenate((expanded, tiled), axis = 1)
        return ret

    def add_train(self, input: np.ndarray, output: np.ndarray, param: np.ndarray):
        param = np.array(param, dtype = np.float32)
        assert input .ndim == 1
        assert output.ndim == 1
        assert input .size == output.size
        assert input .size >  0
        assert param .ndim == 1
        assert param .size == self.num_params
        self.train_inputs .append(_Data._merge(input, param))
        self.train_outputs.append(output)

    def add_vali(self, input: np.ndarray, output: np.ndarray, param: np.ndarray):
        param = np.array(param, dtype = np.float32)
        assert input .ndim == 1
        assert output.ndim == 1
        assert input .size == output.size
        assert input .size >  0
        assert param .ndim == 1
        assert param .size == self.num_params
        self.vali_inputs .append(_Data._merge(input, param))
        self.vali_outputs.append(output)

    def _chunkify(self, chunk_len: int):
        train_inputs  = None
        train_outputs = None
        vali_inputs   = None
        vali_outputs  = None

        for ti, to in zip(self.train_inputs, self.train_outputs):
            num_chunks    = ti.shape[ 0 ] // chunk_len
            len_cut       = num_chunks * chunk_len
            tis           = torch.from_numpy(ti[ :len_cut ].reshape(num_chunks, chunk_len, -1))
            tos           = torch.from_numpy(to[ :len_cut ].reshape(num_chunks, chunk_len))
            train_inputs  = torch.cat((train_inputs , tis), dim = 0) if train_inputs  else tis
            train_outputs = torch.cat((train_outputs, tos), dim = 0) if train_outputs else tos

        for vi, vo in zip(self.vali_inputs, self.vali_outputs):
            vi           = torch.from_numpy(vi)
            vo           = torch.from_numpy(vo)
            vali_inputs  = torch.cat((vali_inputs , vi), dim = 0) if vali_inputs  else vi
            vali_outputs = torch.cat((vali_outputs, vo), dim = 0) if vali_outputs else vo

        return train_inputs, train_outputs, vali_inputs, vali_outputs
