import argparse
import datetime
import os
import random
import re
import time

import numpy as np
import torch
import yaml
from dateutil import tz
import pytorch_lightning as pl
from PIL import Image
from transformers import GPT2TokenizerFast
from pytorch_lightning import seed_everything

# from tools.utils_github import setup_arguments, setup_seed
from tools.metrics.metrics import compute_all_scores
from models.model_github import TrainLanguageModelOneSample

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_float32_matmul_precision('medium')


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
                       choices=['report-generation-single-sample'],
                       help='the task to run. gpt2 is DistilGPT2. ')
    parse.add_argument('--phase', type=str, default='inference', choices=['inference'],
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
    parse.add_argument('--test_ckpt_path', type=str, help='checkpoint for test',
                       default='script/results/mimic_cxr/train-language-model/v0307-all-have_2025_03_18_09-best/checkpoint/best_model.ckpt'
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
    return args


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


def obtain_aligned_reference_reports(tokenizer, text, max_length):
    inputs = tokenizer(text, padding=True, max_length=max_length,
                       truncation=True, return_tensors='pt')
    ref_reports = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
    # delete illegal characters
    ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
    return ref_reports


def build_arguments_model():
    """
    only support inference mode for report generation
    """
    args = setup_arguments()
    setup_seed(args['seed'])
    seed_everything(args['seed'])

    tokenizer = GPT2TokenizerFast.from_pretrained(args['distilgpt2_path'])
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]', 'cls_token': '[CLS]'})
    tokenizer.add_tokens(['[INDICATION]', '[HISTORY]', '[Similar Cases]', '[FINDINGS]'])

    # save the config
    params = ''
    for key, value in args.items():
        params += f'{key}:\t{value}\n'
    print(params)

    # Trainer
    model = TrainLanguageModelOneSample(args, tokenizer)

    cur_model_state = model.state_dict()
    pre_model_state = torch.load(args['test_ckpt_path'])['state_dict']
    valid_state = {k: v for k, v in pre_model_state.items() if
                   k in cur_model_state and v.shape == cur_model_state[k].shape}
    invalid_state = {k for k in pre_model_state.keys() if k not in valid_state}
    print(f"missing {invalid_state}")
    cur_model_state.update(valid_state)
    model.load_state_dict(cur_model_state)

    return model, args


def structured_user_data(image_processor, sep_token, device, current_study, prior_study):
    cur_context = current_study['indication_pure'].strip() + f' {sep_token} ' + current_study['history_pure']
    cur_images = Image.open(current_study['image_path'])
    cur_images = image_processor(cur_images, return_tensors='pt').pixel_values.to(device)
    if prior_study is not None:
        pri_image = Image.open(prior_study['image_path'])
        pri_image = image_processor(pri_image, return_tensors='pt').pixel_values.to(device)
        prior_study = {
            'image': pri_image,
            'view_position': [prior_study['view_position']],
            'pri_idx': [0],
            'no_pri_idx': []
        }
    else:
        prior_study = None
    item = {
        'image_ids': [current_study['id']],
        'current_study': {
            'image': cur_images,
            'view_position': [current_study['view_position']],
        },
        'clinical_context': [cur_context],
        'prior_study': prior_study,
    }
    return item



def main():

    #===================user-defined input================
    # *******v1: current_study(Mandatory) + clinical context(optional) + prior study(optional)******
    current_study = {
        'id': 'b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c',
        'image_path': 'data-demo/b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c.jpg',
        'view_position': 'AP',  # also can be 'unk'
        'indication_pure': '',
        'history_pure': 'woman with respiratory failure. Evaluate fluid status.',
    }
    prior_study = {
        'image_path': 'data-demo/4f4c142-ff4415c6-17466d42-d7531983-33acac69.jpg',
        'view_position': 'AP'
    }
    # # *******v2: current_study(Mandatory) + not clinical context(optional) + prior study (optional)******
    # current_study = {
    #     'id': 'b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c',
    #     'image_path': 'data-demo/b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c.jpg',
    #     'view_position': 'AP',
    #     'indication_pure': '',
    #     'history_pure': '',
    # }
    # prior_study = {
    #     'image_path': 'data-demo/4f4c142-ff4415c6-17466d42-d7531983-33acac69.jpg',
    #     'view_position': 'AP'
    # }
    #
    # # *******v3: current_study(Mandatory) + clinical context(optional) + not prior study(optional)******
    # current_study = {
    #     'id': 'b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c',
    #     'image_path': 'data-demo/b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c.jpg',
    #     'view_position': 'AP',  # also can be 'unk'
    #     'indication_pure': '',
    #     'history_pure': 'woman with respiratory failure. Evaluate fluid status.',
    # }
    # prior_study = None
    #
    # # *******v4: current_study(Mandatory) + not clinical context(optional) + not prior study(optional)******
    # current_study = {
    #     'id': 'b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c',
    #     'image_path': 'data-demo/b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c.jpg',
    #     'view_position': 'AP',  # also can be 'unk'
    #     'indication_pure': '',
    #     'history_pure': '',
    # }
    # prior_study = None

    # reference_report = None  # (alternative)
    reference_report = 'Comparison is made to the prior study from ___. The feeding tube, left IJ catheter and endotracheal tube are unchanged in position. There is persistent cardiomegaly. There is unchanged left retrocardiac opacity. There are no signs for overt pulmonary edema. There is a small right-sided pleural effusion as well. Overall, these findings are stable.'

    model, args = build_arguments_model()
    model = model.to(args['device'])
    model.eval()
    image_processor, sep_token = model.image_processor, model.tokenizer.sep_token
    item = structured_user_data(image_processor, sep_token, args['device'], current_study, prior_study)
    generated_report = model(item)
    print(f'current image id is {current_study["id"]}, generated report is {generated_report}')
    # aligned performance: reference report is truncated into a user-defined length

    if reference_report is not None:
        aligned_reference_report = obtain_aligned_reference_reports(model.tokenizer, reference_report, args['max_length'])
        aligned_scores = compute_all_scores(generated_report, aligned_reference_report, args)
        print("aligned version performance: ", aligned_scores)

        # not aligned performance
        not_aligned_scores = compute_all_scores(generated_report, [reference_report], args)
        print("not aligned version performance: ", not_aligned_scores)

        # v1 version output (has context and prior study):
        # current image id is b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c, generated report is ['Comparison is made to prior study from ___. The endotracheal tube, feeding tube, and left IJ central line are unchanged in position. There is unchanged cardiomegaly. There is a persistent left retrocardiac opacity and left-sided pleural effusion which is stable. There are no signs for overt pulmonary edema. No pneumothoraces are identified.']
        # aligned version performance:  {'Radgraph-partial': 0.8333333333333333, 'Radgraph-simple': 0.8333333333333333, 'Radgraph-complete': 0.7272727272727272, 'chexbert_5_micro_f1': 0.8, 'chexbert_5_macro_f1': 0.4, 'chexbert_all_micro_p': 0.8, 'chexbert_all_micro_r': 1.0, 'chexbert_all_micro_f1': 0.8888888888888888, 'chexbert_all_macro_p': 0.2857142857142857, 'chexbert_all_macro_r': 0.2857142857142857, 'chexbert_all_macro_f1': 0.2857142857142857, 'BLEU_1': 0.8032786885114217, 'BLEU_2': 0.674678632250609, 'BLEU_3': 0.5699687529215061, 'BLEU_4': 0.489609023557076, 'METEOR': 0.44062046564969604, 'ROUGE_L': 0.6753390534182121, 'CIDer': 0.0}
        # not aligned version performance:  {'Radgraph-partial': 0.8333333333333333, 'Radgraph-simple': 0.8333333333333333, 'Radgraph-complete': 0.7272727272727272, 'chexbert_5_micro_f1': 0.8, 'chexbert_5_macro_f1': 0.4, 'chexbert_all_micro_p': 0.8, 'chexbert_all_micro_r': 1.0, 'chexbert_all_micro_f1': 0.8888888888888888, 'chexbert_all_macro_p': 0.2857142857142857, 'chexbert_all_macro_r': 0.2857142857142857, 'chexbert_all_macro_f1': 0.2857142857142857, 'BLEU_1': 0.8032786885114217, 'BLEU_2': 0.674678632250609, 'BLEU_3': 0.5699687529215061, 'BLEU_4': 0.489609023557076, 'METEOR': 0.44062046564969604, 'ROUGE_L': 0.6753390534182121, 'CIDer': 0.0}

        # v2 version output (not context, has prior study):
        # current image id is b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c, generated report is ['Indwelling support and monitoring devices are unchanged in position, and cardiomediastinal contours are stable in appearance. A moderate-sized left pleural effusion appears slightly larger than on the prior study, and a small right pleural effusion appears slightly increased compared to the prior study. There is no evidence of pneumothorax.']
        # aligned version performance:  {'Radgraph-partial': 0.34285714285714286, 'Radgraph-simple': 0.34285714285714286, 'Radgraph-complete': 0.20000000000000004, 'chexbert_5_micro_f1': 0.5, 'chexbert_5_macro_f1': 0.2, 'chexbert_all_micro_p': 0.4, 'chexbert_all_micro_r': 0.6666666666666666, 'chexbert_all_micro_f1': 0.5, 'chexbert_all_macro_p': 0.14285714285714285, 'chexbert_all_macro_r': 0.14285714285714285, 'chexbert_all_macro_f1': 0.14285714285714285, 'BLEU_1': 0.3442622950763236, 'BLEU_2': 0.23953506878624473, 'BLEU_3': 0.15727102948884353, 'BLEU_4': 0.09049603446800758, 'METEOR': 0.1852403372627515, 'ROUGE_L': 0.21548425080953781, 'CIDer': 0.0}
        # not aligned version performance:  {'Radgraph-partial': 0.34285714285714286, 'Radgraph-simple': 0.34285714285714286, 'Radgraph-complete': 0.20000000000000004, 'chexbert_5_micro_f1': 0.5, 'chexbert_5_macro_f1': 0.2, 'chexbert_all_micro_p': 0.4, 'chexbert_all_micro_r': 0.6666666666666666, 'chexbert_all_micro_f1': 0.5, 'chexbert_all_macro_p': 0.14285714285714285, 'chexbert_all_macro_r': 0.14285714285714285, 'chexbert_all_macro_f1': 0.14285714285714285, 'BLEU_1': 0.3442622950763236, 'BLEU_2': 0.23953506878624473, 'BLEU_3': 0.15727102948884353, 'BLEU_4': 0.09049603446800758, 'METEOR': 0.1852403372627515, 'ROUGE_L': 0.21548425080953781, 'CIDer': 0.0}

        # v3 version output (has context, not prior study):
        # current image id is b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c, generated report is ['Comparison is made to prior study from ___. The endotracheal tube, feeding tube, and left IJ central line are unchanged in position. There is unchanged cardiomegaly. There is a persistent left retrocardiac opacity and left-sided pleural effusion. There are no signs for overt pulmonary edema. No pneumothoraces are identified.']
        # aligned version performance:  {'Radgraph-partial': 0.8571428571428571, 'Radgraph-simple': 0.8571428571428571, 'Radgraph-complete': 0.761904761904762, 'chexbert_5_micro_f1': 0.8, 'chexbert_5_macro_f1': 0.4, 'chexbert_all_micro_p': 0.8, 'chexbert_all_micro_r': 1.0, 'chexbert_all_micro_f1': 0.8888888888888888, 'chexbert_all_macro_p': 0.2857142857142857, 'chexbert_all_macro_r': 0.2857142857142857, 'chexbert_all_macro_f1': 0.2857142857142857, 'BLEU_1': 0.7704918032660576, 'BLEU_2': 0.6509765677728108, 'BLEU_3': 0.5565403380608707, 'BLEU_4': 0.4809319320145883, 'METEOR': 0.4452507258976649, 'ROUGE_L': 0.6961483594864479, 'CIDer': 0.0}
        # not aligned version performance:  {'Radgraph-partial': 0.8571428571428571, 'Radgraph-simple': 0.8571428571428571, 'Radgraph-complete': 0.761904761904762, 'chexbert_5_micro_f1': 0.8, 'chexbert_5_macro_f1': 0.4, 'chexbert_all_micro_p': 0.8, 'chexbert_all_micro_r': 1.0, 'chexbert_all_micro_f1': 0.8888888888888888, 'chexbert_all_macro_p': 0.2857142857142857, 'chexbert_all_macro_r': 0.2857142857142857, 'chexbert_all_macro_f1': 0.2857142857142857, 'BLEU_1': 0.7704918032660576, 'BLEU_2': 0.6509765677728108, 'BLEU_3': 0.5565403380608707, 'BLEU_4': 0.4809319320145883, 'METEOR': 0.4452507258976649, 'ROUGE_L': 0.6961483594864479, 'CIDer': 0.0}

        # v4 version output (not context, not prior study)
        # current image id is b83e699f-f3106ae1-2e81b3c2-289d9017-3ddb459c, generated report is ['Indwelling support and monitoring devices are unchanged in position, and cardiomediastinal contours are stable in appearance. A moderate-sized left pleural effusion is again demonstrated with adjacent left lower lobe opacity, which may be due to atelectasis and / or consolidation. Right lung is clear except for minimal linear atelectasis at the right base.']
        # aligned version performance:  {'Radgraph-partial': 0.2790697674418605, 'Radgraph-simple': 0.2790697674418605, 'Radgraph-complete': 0.12, 'chexbert_5_micro_f1': 0.3333333333333333, 'chexbert_5_macro_f1': 0.2, 'chexbert_all_micro_p': 0.6, 'chexbert_all_micro_r': 0.5, 'chexbert_all_micro_f1': 0.5454545454545454, 'chexbert_all_macro_p': 0.21428571428571427, 'chexbert_all_macro_r': 0.21428571428571427, 'chexbert_all_macro_f1': 0.21428571428571427, 'BLEU_1': 0.29508196720827734, 'BLEU_2': 0.14025737466133686, 'BLEU_3': 0.06934254863485513, 'BLEU_4': 8.707487334984397e-06, 'METEOR': 0.1559153991775291, 'ROUGE_L': 0.17226772098277324, 'CIDer': 0.0}
        # not aligned version performance:  {'Radgraph-partial': 0.2790697674418605, 'Radgraph-simple': 0.2790697674418605, 'Radgraph-complete': 0.12, 'chexbert_5_micro_f1': 0.3333333333333333, 'chexbert_5_macro_f1': 0.2, 'chexbert_all_micro_p': 0.6, 'chexbert_all_micro_r': 0.5, 'chexbert_all_micro_f1': 0.5454545454545454, 'chexbert_all_macro_p': 0.21428571428571427, 'chexbert_all_macro_r': 0.21428571428571427, 'chexbert_all_macro_f1': 0.21428571428571427, 'BLEU_1': 0.29508196720827734, 'BLEU_2': 0.14025737466133686, 'BLEU_3': 0.06934254863485513, 'BLEU_4': 8.707487334984397e-06, 'METEOR': 0.1559153991775291, 'ROUGE_L': 0.17226772098277324, 'CIDer': 0.0}

if __name__ == '__main__':
    main()
