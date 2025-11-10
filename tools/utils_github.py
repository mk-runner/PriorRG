import argparse
import datetime
import random
import os
import yaml
import torch
import threading
import numpy as np
import multiprocessing
from dateutil import tz


def str2bool(value):
    if value.lower() in ['yes', 'true', 't', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_arguments():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    # basic configuration
    parse.add_argument('--task', type=str, default='pretraining',
                       choices=['pretraining', 'report-generation-gpt2', 'report-generation-single-sample'],
                       help='the task to run. gpt2 is DistilGPT2. ')
    parse.add_argument('--phase', type=str, default='inference', choices=['finetune', 'inference'],
                       help='Is this task for fine-tuning or inference mode?')
    # data configuration
    parse.add_argument('--data_name', type=str, choices=['mimic_cxr', 'mimic_abn'], default='mimic_cxr')
    parse.add_argument('--ann_path', type=str, help='annotation for radiology reports',
                       default='/home/miao/data/dataset/MIMIC-CXR/five_work_mimic_cxr_annotation_v2_similar_case.json',
                       )
    parse.add_argument('--view_position_dict', type=str, help='the dictory of view positions',
                       default='/home/miao/data/dataset/MIMIC-CXR/five_work_mimic_cxr_view_position_v1.1.json'
                       )
    parse.add_argument('--images_dir', type=str, default='/home/miao/data/dataset/MIMIC-CXR/files/',
                       help='the directory of images')
    parse.add_argument('--max_length', type=int, default=100, help='the maximum number of generated tokens')
    parse.add_argument('--encoder_max_length', type=int, default=300,
                       help='The maximum number of tokens encoded by the text encoder to avoid excessive memory consumption')
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
    parse.add_argument('--patience', type=int, default=10, help='used for learning rate')
    parse.add_argument('--is_save_checkpoint', type=str2bool, default='no', help='whether save checkpoint')
    parse.add_argument('--online_checkpoint', type=str2bool, default='no', help='whether using online checkpoint')
    # parse.add_argument('--is_prior_scan', type=str2bool, default='yes', help='whether using prior scan')
    # parse.add_argument('--is_layer_fusion', type=str2bool, default='yes', help='whether obtaining high-granularity visual features')
    # knowledge
    # parse.add_argument('--is_indication', type=str2bool, default='yes', help='whether using indications')
    # parse.add_argument('--is_history', type=str2bool, default='yes', help='whether using history')
    parse.add_argument('--ckpt_zoo_dir', type=str,
                       default='/home/miao/data/dataset/checkpoints/',
                       help='if using local checkpoint, this variable must be provided')
    parse.add_argument('--text_encoder_num_blocks', type=int, default=6)
    parse.add_argument('--temporal_fusion_num_blocks', type=int, default=3)
    parse.add_argument('--perceiver_num_blocks', type=int, default=3)
    parse.add_argument('--num_heads', type=int, default=8)
    parse.add_argument('--num_latents', type=int, default=128, help='the number of latents used in Perceiver.')

    # trainer configuration-
    parse.add_argument('--pt_lr', type=float, default=5.0e-6, help='learning rate of pretraining module.')  # 5.0e-5
    parse.add_argument('--ft_lr', type=float, default=5.0e-5, help='learning rate of non-pretraining module, '
                                                                   'e.g., language generator')  # 5.0e-5
    parse.add_argument('--temp', type=float, default=0.5,
                       help='temperature parameter for instance-wise alignment')  # 5.0e-5
    parse.add_argument('--monitor_metric', type=str, default='RCB',
                       help='the metric is used to selecting best models. '
                            'align phase is all_loss, while fine-tuning is RCB (bleu4+f1-chexpert+f1-radgraph)')
    # choices={all_metrics, metrics, RC, RB, RCB}
    parse.add_argument('--hidden_size', type=int, default=768, help='the dimension of unified space between multimodal')
    parse.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.',
                       # default='script/results/mimic_cxr/finetune/v0915_ft-fs_2024_09_17_10/checkpoint/last-1.0329.ckpt'
                       )
    parse.add_argument('--load', type=str, help='whether to load the pre-trained model.',
                       # default='script/results/mimic_cxr/align/v0207-align_2025_02_07_22-best/checkpoint/best_model.ckpt'
                       # default='script/results/mimic_cxr/train-language-model/v0307-all-have_2025_03_12_10/checkpoint/best_model.ckpt'
                       )
    parse.add_argument('--test_ckpt_path', type=str, help='checkpoint for test',
                       # default='script/results/mimic_cxr/train-language-model/v0307-all-have_2025_03_18_09-best/checkpoint/best_model.ckpt'
                       # default='/home/miao/data/Code/PriorRG/script/results/mimic_cxr/align/v0207-align_2025_02_07_22-best/checkpoint/best_model.ckpt'
                       )
    parse.add_argument('--project_name', type=str, default='PriorRG', help='the name of the project')
    # ========= metrics checkpoint config =====#
    parse.add_argument('--chexbert_path', type=str, default='chexbert.pth', help='checkpoint for f1-chexbert')
    parse.add_argument('--bert_path', type=str, default='google-bert/bert-base-uncased', help='checkpoint for f1-chexbert')
    parse.add_argument('--radgraph_path', type=str, default='radgraph', help='checkpoint for f1-radgraph')
    # ========= backbone checkpoint config =====#
    parse.add_argument('--cxr_bert_path', type=str, default='microsoft/BiomedVLP-CXR-BERT-specialized',
                       help='checkpoint for text-encoder')
    parse.add_argument('--rad_dino_path', type=str, default='microsoft/rad-dino',
                       help='checkpoint of image encoder')
    parse.add_argument('--llama_path', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       help='checkpoint of language generator')
    parse.add_argument('--distilgpt2_path', type=str, default='distilbert/distilgpt2',
                       help='checkpoint of language generator')
    # ========= implementation config =====#
    parse.add_argument('--seed', type=int, default=9233, help='random seed')
    parse.add_argument('--num_beams', type=int, default=3, help='beam size for language generation')
    # ========= record results ============#
    parse.add_argument('--version', type=str, default='v1',
                       help='the name of version')
    parse.add_argument('--print_step', type=int, default=500, help='the frequency of print')
    # =============finish=====================#
    args = parse.parse_args()
    args = vars(args)
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H")
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['project_name'] = f'{args["project_name"]}/{args["data_name"]}/{args["task"]}/{args["version"]}_{extension}'
    os.makedirs(args['project_name'], exist_ok=True)

    # config logger
    logger = SetLogger(f'{args["project_name"]}/log_{extension}.log', 'a')

    # determine absolute path for checkpoints
    if not args['online_checkpoint']:
        candi_list = ['chexbert_path', 'radgraph_path', "bert_path", 'cxr_bert_path',
                      "distilgpt2_path", "rad_dino_path", 'llama_path']
    else:  # for clinical efficacy metrics
        candi_list = ['chexbert_path', 'radgraph_path']

    for candi in candi_list:
        if args[candi] is None:
            continue
        args[candi] = os.path.join(args['ckpt_zoo_dir'], args[candi])
    # determine the monitor_mode
    args['monitor_mode'] = 'max'
    if args['task'] == 'pretraining':  # pretrain
        args['monitor_mode'] = 'min'
        args['monitor_metric'] = 'val_epoch_loss'

    checkpoint_dir = os.path.join(args['project_name'], 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    args['checkpoint_dir'] = checkpoint_dir
    args['time'] = extension
    # save parameters
    config_dir = f"{args['project_name']}/configs"
    os.makedirs(config_dir, exist_ok=True)
    file_name = f"{config_dir}/config_{extension}.yaml"
    print(f'parameters is saved in {file_name}')
    with open(file_name, 'w') as file:
        yaml.dump(args, file, default_flow_style=False)
    return args, logger


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SetLogger:
    def __init__(self, filepath, mode='a', lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi-process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            raise ValueError("Mode must be 'w' or 'a'")
        self.mode = mode
        self.lock = lock or (multiprocessing.Lock() if 'multiprocessing' in globals() else threading.Lock())

        try:
            self.file = open(self.filepath, self.mode)
        except Exception as e:
            print(f"Failed to open log file: {e}")
            raise

    def info(self, message):
        """
        Log an info message to the file.
        :param message: The message to log
        """
        with self.lock:
            try:
                self.file.write(message + '\n')
                self.file.flush()
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def __del__(self):
        """Ensure that the file is closed when the logger is destroyed."""
        try:
            if not self.file.closed:
                self.file.close()
        except Exception as e:
            print(f"Failed to close log file: {e}")
