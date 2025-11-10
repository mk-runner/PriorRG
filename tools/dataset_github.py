import copy
import math
import random
import re
import os
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class AlignDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, tokenizer=None):
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        bos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
        for item in ann:
            # delete the sample that has no clinical findings
            if len(item['findings_factual_serialization']) == 0:
                continue
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            self.examples.append({
                'id': f'{item["subject_id"]}_{item["study_id"]}_{item["id"]}',
                'anchor_scan': item['anchor_scan'],  # {view_position: **, image_path: **}
                # 'auxiliary_references': item['auxiliary_references'],
                'align_report': '[CLS] [FINDINGS] ' + report,  # for alignment
                'ref_report': f'{bos_token} ' + report + f' {eos_token}',  # for calculating language modeling loss
                'prior_study': item['prior_study'],
                'knowledge': {
                    'indication': '[INDICATION] ' + indication,
                    'history': '[HISTORY] ' + history
                }
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sample = (example['id'], example['anchor_scan'], example['align_report'], example['ref_report'],
                  example['prior_study'], example['knowledge'])
        return sample


class AlignCollateFn:
    def __init__(self, args, processor, sep_token):
        self.args = args
        self.processor = processor
        self.sep_token = sep_token

    def __call__(self, data):
        # note that images is tuple(list, list)
        # image_ids, anchor_scans, reports, prior_studies = zip(*data)
        b_ids, b_cur_imgs, b_reports, b_ref_reports, b_pri_imgs, b_knowledge = zip(*data)
        cur_images, pri_images, patient_ids, reports = [], [], [], []
        cur_view_position, pri_view_position, has_pri_idx, no_pri_idx = [], [], [], []
        image_ids, knowledge_list, reference_reports = [], [], []
        for i, (ids, anchor_scan, report, ref_report, prior_study, knowledge) in enumerate(zip(b_ids, b_cur_imgs, b_reports, b_ref_reports, b_pri_imgs, b_knowledge)):
            image_ids.append(ids)
            sub2stu_id = '_'.join(ids.split("_")[:-1])
            patient_ids.append(sub2stu_id)  # subject-id + study-id
            # obtain the clinical prompt (indication + [SEP] + history)
            cur_knowledge = knowledge['indication'].strip() + f' {self.sep_token} ' + knowledge['history']
            knowledge_list.append(cur_knowledge)
            # obtain radiology reports (findings section)
            reports.append(report)
            reference_reports.append(ref_report)
            # obtain anchor scan and view position
            image_path, vp = anchor_scan['image_path'][0], anchor_scan['view_position'][0]
            image = Image.open(os.path.join(self.args['images_dir'], image_path))
            image = self.processor(image, return_tensors='pt').pixel_values
            cur_images.append(image)
            cur_view_position.append(vp)
            # obtain prior scan and view position
            if prior_study is None:
                no_pri_idx.append(i)
            else:
                has_pri_idx.append(i)
                for k, v in prior_study.items():  # only use latest study
                    if k != 'latest_study':
                        continue
                    ## prior scan
                    image = Image.open(os.path.join(self.args['images_dir'], v['image_path']))
                    image = self.processor(image, return_tensors='pt').pixel_values
                    pri_images.append(image)
                    pri_view_position.append(v['view_position'])

        cur_images = torch.cat(cur_images, dim=0)  # (batch_size, 3, 518, 518)
        patient_ids = np.array(patient_ids)

        if len(pri_images) != 0:
            pri_images = torch.cat(pri_images, dim=0)
            prior_study = {
                'image': pri_images,
                'view_position': pri_view_position,
                'pri_idx': has_pri_idx,
                'no_pri_idx': no_pri_idx
            }
        else:
            prior_study = None
        item = {
            'image_ids': image_ids,
            'patient_ids': patient_ids,
            'report': reports,  # for cross-modal alignment
            'reference_report': reference_reports,  # for language modeling loss
            'current_study': {
                'image': cur_images,
                'view_position': cur_view_position,
            },
            'clinical_context': knowledge_list,
            'prior_study': prior_study
        }
        return item


class FinetuneDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, tokenizer):
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        bos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
        self.examples = []
        for item in ann:
            if len(item['findings_factual_serialization']) == 0:
                continue  # delete invalid findings (which has not clinical meaning.)
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']
            # similar_case = item["similar_case"]["findings"].strip() if args['is_similar_case'] else ""
            self.examples.append({
                'id': f'{item["subject_id"]}_{item["study_id"]}_{item["id"]}',
                'anchor_scan': item['anchor_scan'],  # {view_position: **, image_path: **}
                'ref_report': f'{bos_token} ' + report + f' {eos_token}',  # for calculating language modeling loss
                'prior_study': item['prior_study'],
                'knowledge': {
                    'indication': '[INDICATION] ' + indication,
                    'history': '[HISTORY] ' + history,
                }
            })
        # self.examples = self.examples[:500]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sample = (example['id'], example['anchor_scan'], example['ref_report'],
                  example['prior_study'], example['knowledge'])
        return sample


class FinetuneCollateFn:
    def __init__(self, args, processor, sep_token):
        self.processor = processor
        self.args = args
        self.sep_token = sep_token

    def __call__(self, data):
        b_ids, b_cur_imgs, b_ref_reports, b_pri_imgs, b_knowledge = zip(*data)
        cur_images, pri_images = [], []
        cur_view_position, pri_view_position, has_pri_idx, no_pri_idx = [], [], [], []
        image_ids, reference_reports = [], []
        contexts = []
        for i, (ids, anchor_scan, ref_report, prior_study, knowledge) in enumerate(zip(b_ids, b_cur_imgs, b_ref_reports, b_pri_imgs, b_knowledge)):
            image_ids.append(ids)
            # obtain the clinical context (indication + [SEP] + history)
            cur_context = knowledge['indication'].strip() + f' {self.sep_token} ' + knowledge['history']
            contexts.append(cur_context)
            # obtain radiology reports (findings section)
            reference_reports.append(ref_report)
            # obtain anchor scan and view position
            image_path, vp = anchor_scan['image_path'][0], anchor_scan['view_position'][0]
            image = Image.open(os.path.join(self.args['images_dir'], image_path))
            image = self.processor(image, return_tensors='pt').pixel_values
            cur_images.append(image)
            cur_view_position.append(vp)
            # obtain prior scan and view position
            if prior_study is None:
                no_pri_idx.append(i)
            else:
                has_pri_idx.append(i)
                for k, v in prior_study.items():  # only use latest study
                    if k != 'latest_study':
                        continue
                    ## prior scan
                    image = Image.open(os.path.join(self.args['images_dir'], v['image_path']))
                    image = self.processor(image, return_tensors='pt').pixel_values
                    pri_images.append(image)
                    pri_view_position.append(v['view_position'])

        cur_images = torch.cat(cur_images, dim=0)  # (batch_size, 3, 518, 518)

        # ============determine prior scan based on user-defined command
        if len(pri_images) != 0:
            pri_images = torch.cat(pri_images, dim=0)
            prior_study = {
                'image': pri_images,
                'view_position': pri_view_position,
                'pri_idx': has_pri_idx,
                'no_pri_idx': no_pri_idx
            }
        else:
            prior_study = None

        item = {
            'image_ids': image_ids,
            'reference_report': reference_reports,  # for language modeling loss
            'current_study': {
                'image': cur_images,
                'view_position': cur_view_position,
            },
            'clinical_context': contexts,
            'prior_study': prior_study,
        }
        return item


class FinetuneDatasetLLM(Dataset):  # finetune and inference phase
    def __init__(self, args, split, tokenizer):
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        bos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
        self.examples = []
        for item in ann:
            if len(item['findings_factual_serialization']) == 0:
                continue  # delete invalid findings (which has not clinical meaning.)
            report = item['findings'].strip()
            indication = item['indication_pure']
            history = item['history_pure']

            self.examples.append({
                'id': f'{item["subject_id"]}_{item["study_id"]}_{item["id"]}',
                'anchor_scan': item['anchor_scan'],  # {view_position: **, image_path: **}
                'ref_report': f'{bos_token} ' + report + f' {eos_token}',  # for calculating language modeling loss
                'prior_study': item['prior_study'],
                'knowledge': {
                    'indication': '[INDICATION] ' + indication,
                    'history': '[HISTORY] ' + history,
                },
                'similar_case': item["similar_case"]["findings"].strip(),
            })
        # self.examples = self.examples[:500]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sample = (example['id'], example['anchor_scan'], example['ref_report'],
                  example['prior_study'], example['knowledge'], example['similar_case'])
        return sample


class FinetuneCollateFnLMM:
    def __init__(self, args, processor, sep_token):
        self.processor = processor
        self.args = args
        self.sep_token = sep_token

    def __call__(self, data):
        b_ids, b_cur_imgs, b_ref_reports, b_pri_imgs, b_knowledge, b_similar = zip(*data)
        cur_images, pri_images = [], []
        cur_view_position, pri_view_position, has_pri_idx, no_pri_idx = [], [], [], []
        image_ids, reference_reports = [], []
        contexts = []
        for i, (ids, anchor_scan, ref_report, prior_study, knowledge) in enumerate(zip(b_ids, b_cur_imgs, b_ref_reports, b_pri_imgs, b_knowledge)):
            image_ids.append(ids)
            # obtain the clinical context (indication + [SEP] + history)
            cur_context = knowledge['indication'].strip() + f' {self.sep_token} ' + knowledge['history']
            contexts.append(cur_context)
            # obtain radiology reports (findings section)
            reference_reports.append(ref_report)
            # obtain anchor scan and view position
            image_path, vp = anchor_scan['image_path'][0], anchor_scan['view_position'][0]
            image = Image.open(os.path.join(self.args['images_dir'], image_path))
            image = self.processor(image, return_tensors='pt').pixel_values
            cur_images.append(image)
            cur_view_position.append(vp)
            # obtain prior scan and view position
            if prior_study is None:
                no_pri_idx.append(i)
            else:
                has_pri_idx.append(i)
                for k, v in prior_study.items():  # only use latest study
                    if k != 'latest_study':
                        continue
                    ## prior scan
                    image = Image.open(os.path.join(self.args['images_dir'], v['image_path']))
                    image = self.processor(image, return_tensors='pt').pixel_values
                    pri_images.append(image)
                    pri_view_position.append(v['view_position'])

        cur_images = torch.cat(cur_images, dim=0)  # (batch_size, 3, 518, 518)

        # ============determine prior scan based on user-defined command
        if len(pri_images) != 0:
            pri_images = torch.cat(pri_images, dim=0)
            prior_study = {
                'image': pri_images,
                'view_position': pri_view_position,
                'pri_idx': has_pri_idx,
                'no_pri_idx': no_pri_idx
            }
        else:
            prior_study = None

        item = {
            'image_ids': image_ids,
            'reference_report': reference_reports,  # for language modeling loss
            'current_study': {
                'image': cur_images,
                'view_position': cur_view_position,
            },
            'clinical_context': contexts,
            'prior_study': prior_study,
            'similar_case': b_similar
        }
        return item






