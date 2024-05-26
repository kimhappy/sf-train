import numpy as np

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
        def _input_chunk(x):
            num_chunks = x.shape[ 0 ] // chunk_len
            len_cut    = num_chunks    * chunk_len
            return x[ :len_cut ].reshape(num_chunks, chunk_len, -1)

        def _output_chunk(x):
            num_chunks = x.shape[ 0 ] // chunk_len
            len_cut    = num_chunks    * chunk_len
            return x[ :len_cut ].reshape(num_chunks, chunk_len)

        train_inputs  = np.concatenate(list(map(_input_chunk , self.train_inputs )))
        train_outputs = np.concatenate(list(map(_output_chunk, self.train_outputs)))
        vali_inputs   = np.concatenate(                        self.vali_inputs    )
        vali_outputs  = np.concatenate(                        self.vali_outputs   )

        return train_inputs, train_outputs, vali_inputs, vali_outputs
