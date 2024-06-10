import sys
from sf_train      import ModelConfig, TrainConfig, Trainer
from sf_train.util import read_mono, write_mono

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print(f'Usage: python3 {sys.argv[ 0 ]} <name> <train_input> <train_target> <val_input> <val_target>')
        sys.exit(1)

    model_name   = sys.argv[ 1 ]
    train_input  = read_mono(sys.argv[ 2 ])
    train_output = read_mono(sys.argv[ 3 ])
    vali_input   = read_mono(sys.argv[ 4 ])
    vali_output  = read_mono(sys.argv[ 5 ])

    model_config = ModelConfig(model_name, [])
    train_config = TrainConfig(
        compile     = True   ,
        batch_size  = 16     ,
        initial_lr  =  0.005 ,
        lr_decay    =  0.0001,
        lr_factor   =  0.5   ,
        lr_patience =  5)
    trainer = Trainer(model_config, train_config)

    trainer.add_train(train_input, train_output, [])
    trainer.add_vali (vali_input , vali_output , [])

    MAX_EPOCHS     = 2000
    VALI_CYCLE     =    2
    EARLY_PATIENCE =   20

    patience_counter =    0
    best_vali_loss   = 1e12

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = trainer.train()
        print(f'train epoch: {epoch} | loss: {tr.loss:.6f} | time: {tr.time_elapsed:.2f}s')

        if epoch % VALI_CYCLE != 0:
            continue

        vr = trainer.vali()
        print(f'vali loss: {vr.loss:.6f} | time: {vr.time_elapsed:.2f}s')

        if vr.loss < best_vali_loss:
            patience_counter = 0
            best_vali_loss   = vr.loss
            trainer.save(f'{trainer.name}.snow')
            write_mono(f'{trainer.name}.wav', vr.processed, Trainer.SAMPLE_RATE)
            print(f'new best vali loss: {best_vali_loss:.6f}')
            continue

        patience_counter += 1
        print(f'patience counter: {patience_counter} / {EARLY_PATIENCE}')

        if patience_counter > EARLY_PATIENCE:
            print(f'validation patience limit reached at epoch {epoch}')
            break
