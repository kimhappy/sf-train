import sys
from sf_train      import Data, ModelConfig, TrainConfig, Train
from sf_train.util import read_mono, write_mono

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print(f'Usage: python3 {sys.argv[ 0 ]} <train_input> <train_target> <val_input> <val_target> <name>')
        sys.exit(1)

    model_config = ModelConfig(
        sample_rate = 48000,
        hidden_size = 64)

    train_config = TrainConfig(
        compile     = True   ,
        batch_size  = 16     ,
        initial_lr  =  0.005 ,
        lr_decay    =  0.0001,
        lr_factor   =  0.5   ,
        lr_patience =  5)

    MAX_EPOCHS     = 2000
    VALI_CYCLE     =    2
    EARLY_PATIENCE =   20

    data = Data(0)

    train_input  = read_mono(sys.argv[ 1 ])
    train_output = read_mono(sys.argv[ 2 ])
    vali_input   = read_mono(sys.argv[ 3 ])
    vali_output  = read_mono(sys.argv[ 4 ])

    data.add_train(train_input , train_output, [])
    data.add_vali (vali_input  , vali_output , [])

    train = Train(model_config, train_config, data)

    patience_counter =    0
    best_vali_loss   = 1e12
    file_name        = f'{model_config.sample_rate}_{data.num_params}_{model_config.hidden_size}_{sys.argv[ 5 ]}'

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = train.train()
        print(f'train epoch: {epoch} | loss: {tr.loss:.6f} | time: {tr.time_elapsed:.2f}s')

        if epoch % VALI_CYCLE != 0:
            continue

        vr = train.vali()
        print(f'vali loss: {vr.loss:.6f} | time: {vr.time_elapsed:.2f}s')

        if vr.loss < best_vali_loss:
            patience_counter = 0
            best_vali_loss   = vr.loss
            train.save(f'{file_name}.snow')
            write_mono(f'{file_name}.wav', vr.processed, model_config.sample_rate)
            continue

        patience_counter += 1
        print(f'patience counter: {patience_counter} / {EARLY_PATIENCE}')

        if patience_counter > EARLY_PATIENCE:
            print(f'validation patience limit reached at epoch {epoch}')
            break
