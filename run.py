import sys
import torch
from sf_train      import Data, ModelConfig, TrainConfig, Train
from sf_train.util import read_mono, write_mono

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f'Usage: python3 {sys.argv[ 0 ]} <train_input> <train_target> <val_input> <val_target>')
        sys.exit(1)

    model_config = ModelConfig(
        sample_rate = 48000,
        input_size  = 1    ,
        output_size = 1    ,
        hidden_size = 64)

    train_config = TrainConfig(
        chunk_sec   =      0.5   ,
        batch_size  =     16     ,
        initial_lr  =      0.005 ,
        lr_decay    =      0.0001,
        lr_factor   =      0.5   ,
        lr_patience =      5     ,
        init_len    =    200     ,
        up_fr       =   1000     ,
        vali_chunk  = 100000)

    data = Data(0)

    train_input  = read_mono(sys.argv[ 1 ])
    train_output = read_mono(sys.argv[ 2 ])
    vali_input   = read_mono(sys.argv[ 3 ])
    vali_output  = read_mono(sys.argv[ 4 ])

    data.add_train(train_input , train_output, [])
    data.add_vali (vali_input  , vali_output , [])

    train = Train(model_config, train_config, data)

    max_epochs       = 2000
    vali_cycle       =    2
    early_patience   =   20
    patience_counter =    0
    best_vali_loss   = 1e12

    for epoch in range(1, max_epochs + 1):
        tr = train.train()
        print(f'train epoch: {epoch} | loss: {tr.loss:.6f} | time: {tr.time_elapsed:.2f}s')

        if epoch % vali_cycle != 0:
            continue

        vr = train.vali()
        print(f'vali loss: {vr.loss:.6f} | time: {vr.time_elapsed:.2f}s')

        if vr.loss < best_vali_loss:
            patience_counter = 0
            best_vali_loss   = vr.loss
            torch.save(train.state_dict(), 'best.pt')
            write_mono('best.wav', vr.processed, model_config.sample_rate)
            continue

        patience_counter += 1
        print(f'patience counter: {patience_counter} / {early_patience}')

        if patience_counter > early_patience:
            print(f'validation patience limit reached at epoch {epoch}')
            break
