import os
import time
import torch
import pytorch_lightning as pl
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

from tools.utils_github import setup_arguments, setup_seed
from models.model_github import Alignment, TrainLanguageModel, TrainLanguageModelOneSample

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


def main():

    args, logger = setup_arguments()
    setup_seed(args['seed'])
    seed_everything(args['seed'])
    if args['is_save_checkpoint']:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=False,
            monitor=args['monitor_metric'],
            mode=args['monitor_mode'],
            save_last=True,
            save_weights_only=False,
            dirpath=args['checkpoint_dir'],
            filename='best_model'
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=0,
            verbose=False,
            monitor=args['monitor_metric'],
            mode=args['monitor_mode'],
            save_last=False,
            save_weights_only=False,
        )
    earlystop_callback = pl.callbacks.EarlyStopping(
        monitor=args["monitor_metric"],
        patience=15,
        verbose=False, mode=args['monitor_mode']
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_callback, earlystop_callback]

    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
    tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]'])

    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    logger.info(params)
    # print(params)

    # Trainer
    # Training Hyper-Parameters
    check_val_every_n_epoch = 1
    if args['phase'] == 'inference':
        check_val_every_n_epoch = None
    strategy = 'auto'
    if args['num_gpus'] > 1:
        strategy = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=2,
        benchmark=False,
        callbacks=callbacks,
        check_val_every_n_epoch=check_val_every_n_epoch,
        strategy=strategy,
        devices=args["num_gpus"],
        deterministic=True,
        max_epochs=args['epochs'],
        logger=None,
        log_every_n_steps=500,
        enable_model_summary=True,
        profiler="simple",
    )

    if args['task'] == 'pretraining':
        model = Alignment(args, tokenizer, logger)
    elif args['task'] == 'report-generation-gpt2':  # ['train-language-model']
        model = TrainLanguageModel(args, tokenizer, logger)
    else:
        raise ValueError('not implemented!')

    if args['phase'] != 'inference':
        if args['load'] is not None:
            if args['resume'] is not None:
                trainer.fit(model=model, ckpt_path=args['resume'])
            else:
                cur_model_state = model.state_dict()
                pre_model_state = torch.load(args['load'])['state_dict']
                valid_state = {k: v for k, v in pre_model_state.items() if
                               k in cur_model_state and v.shape == cur_model_state[k].shape}
                invalid_state = {k for k in pre_model_state.keys() if k not in valid_state}
                print(f"missing {invalid_state}")
                cur_model_state.update(valid_state)
                model.load_state_dict(cur_model_state)
                trainer.fit(model=model)
        else:
            if args['resume'] is not None:
                trainer.fit(model=model, ckpt_path=args['resume'])
            else:
                trainer.fit(model=model)
    else:   # test
        start = time.time()
        trainer.test(model=model, ckpt_path=args['test_ckpt_path'])
        end = time.time()
        print(f'Inference time: {end-start:.6f} seconds')


if __name__ == '__main__':
    main()
