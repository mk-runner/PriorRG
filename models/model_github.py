import copy
import json
import math
import re
from typing import Dict

import torch
import numpy as np
import torchmetrics
import transformers
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities import rank_zero_only
from transformers.configuration_utils import PretrainedConfig
from transformers import GPT2TokenizerFast, AutoModel, AutoConfig, AutoImageProcessor

from models.bert_model import TemporalFusion
from models.perceiver_pytorch import Perceiver
from tools.metrics.chexbert import RadGraphMetrics, F1CheXbertMetrics
from tools.metrics.coco import COCOCaptionMetrics
from tools.metrics.report_logger import ReportLogger
from tools.dataset_github import (AlignDataset, FinetuneDataset,
                                  AlignCollateFn, FinetuneCollateFn)


class Alignment(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            tokenizer: GPT2TokenizerFast,
            logger,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.val_min_losses = {
            "epoch": -1,
            'loss': 1000
        }  # loss = instance_loss

        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }

        self.train_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args['device']),
        }
        self.val_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args['device']),
        }
        self.test_loss_metric = {
            'loss': torchmetrics.MeanMetric().to(args['device']),
        }

        # Image Encoder (frozen):
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.config.output_hidden_states = True
        image_dim = self.image_encoder.config.hidden_size
        self.freeze_parameters(self.image_encoder)

        # Text Encoder
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head
        self.image_projection = VisualProjectionHead(image_dim, args['hidden_size'] // 2, args['hidden_size'])
        self.text_projection = ProjectionHead(text_dim, args['hidden_size'] // 2, args['hidden_size'])

        # vp_pos_embed for view_position
        self.vp2id = json.load(open(args['view_position_dict']))
        self.vp_pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=True)
        # temp_pos_embed for temporal information
        self.temp_pos_embed = nn.Parameter(torch.randn(2, 1, args['hidden_size']), requires_grad=True)
        # define temperature hyper-parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        self.layer_norm = nn.LayerNorm(args['hidden_size'])

        # temporal_fusion module
        self.temporal_fusion = TemporalFusion(args['hidden_size'], args['temporal_fusion_num_blocks'],
                                              heads=args['num_heads'], dim_head=args['hidden_size'] // 4,
                                              mlp_dim=args['hidden_size'])

        # fusion high-level visual features, low-high-level visual features, knowledge
        self.perceiver = Perceiver(
            byte_dim=args['hidden_size'],  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['num_latents'],  # number of latents
            latent_dim=args['hidden_size'],  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = AlignDataset(self.args, 'train', self.tokenizer)
            self.val_set = AlignDataset(self.args, 'val', self.tokenizer)
            print(
                "No. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  # fit
            self.test_set = AlignDataset(self.args, 'test', self.tokenizer)
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        collate_fn = AlignCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        all_parameters = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            all_parameters.append(param)
        optimiser = torch.optim.AdamW(all_parameters, lr=self.args['pt_lr'])
        lr_scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                         factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1  # the frequency of check
            }
        }

    def tokenization(self, text, device):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False,
                                max_length=self.args['max_length'], truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        return inputs

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        del patient_ids

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2)

        # calculate the InfoNCE loss
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * global_image_embed @ global_text_embed.t()
        logits_per_text = logits_per_image.t()
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        global_instance_loss = (loss_image + loss_text) / 2.0
        return global_instance_loss

    def image_encoder_forward(self, images):
        with torch.no_grad():
            outputs = self.image_encoder(images)
            last_hidden_state = outputs['last_hidden_state']
            hidden_states = torch.stack(outputs['hidden_states'][1:], dim=1)  # the first token is [cls] token
        return hidden_states, last_hidden_state

    def obtain_spatio_temporal_visual_features(self, current_study, prior_study=None):
        # ===================obtain spatio-temporal visual features =====================
        _, last_hidden_state = self.image_encoder_forward(current_study['image'])

        # add view position embeddings
        image_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in current_study['view_position']]
        cur_vis_feat = torch.cat(image_pos_embed, dim=0) + last_hidden_state
        # image projection
        cur_vis_feat = self.image_projection(cur_vis_feat)  # first sublayer is a layer norm
        # add temporal position embeddings
        cur_temporal_embed = self.temp_pos_embed[0].repeat(cur_vis_feat.shape[0], 1, 1)
        cur_vis_feat = cur_vis_feat + cur_temporal_embed
        # fusion prior study using temporal_fusion network
        spatio_temp_feat = torch.empty_like(cur_vis_feat).to(cur_vis_feat)
        if prior_study is not None:
            # fill the spatio_temp_feat using prior scan
            _, pri_last_hidden_state = self.image_encoder_forward(prior_study['image'])
            # view position embedding
            pri_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in prior_study['view_position']]
            pri_last_hidden_state = torch.cat(pri_pos_embed, dim=0) + pri_last_hidden_state
            # image projection
            pri_last_hidden_state = self.image_projection(pri_last_hidden_state)
            # temporal position embedding
            pri_temporal_embed = self.temp_pos_embed[1].repeat(pri_last_hidden_state.shape[0], 1, 1)
            pri_last_hidden_state = pri_temporal_embed + pri_last_hidden_state
            # temporal image fusion
            has_pri_idx = prior_study['pri_idx']
            # temporal_fusion (the first and lasy sublayer have layer-norm)
            temp_visual_features = self.temporal_fusion(cur_vis_feat[has_pri_idx], pri_last_hidden_state)
            spatio_temp_feat[has_pri_idx] = temp_visual_features
            # fill the pri_img_embed using the learnable embed
            no_pri_idx = prior_study['no_pri_idx']
        else:
            no_pri_idx = list(range(cur_vis_feat.shape[0]))
        spatio_temp_feat[no_pri_idx] = self.layer_norm(cur_vis_feat[no_pri_idx])
        return spatio_temp_feat

    def obtain_textual_features(self, reports, device, return_attention_mask=False):
        inputs = self.tokenization(reports, device=device)
        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num + 1, 768)
        if not return_attention_mask:
            return text_embed
        else:
            return text_embed, inputs['attention_mask']

    def forward(self, current_study, reports, reference_reports, patient_ids, context, prior_study=None, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        # =================== extract uni-modal features =======================
        # obtain spatio-temporal visual features
        spatio_temp_feat = self.obtain_spatio_temporal_visual_features(current_study, prior_study)
        # obtain context information
        context_embed = self.obtain_textual_features(context, spatio_temp_feat.device)

        # =============== fusion multimodal information using Perceiver (compact)=============
        context_latents = self.perceiver(context_embed)
        spatio_temp_latents = self.perceiver(spatio_temp_feat, latent=context_latents)
        encoder_outputs = torch.cat([context_latents, spatio_temp_latents], dim=1)

        # ================ instance-level contrastive loss ================================
        # extract report features using [CLS] token (for cross-modal alignment)
        text_embed = self.obtain_textual_features(reports, spatio_temp_feat.device)
        # ===========instance-level contrastive loss====
        image_cls_embed = torch.mean(encoder_outputs, dim=1)
        instance_loss = self.global_alignment_loss(image_cls_embed, text_embed[:, 0, :], patient_ids)

        return {
            'loss': instance_loss
        }

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='train')

        self.log_dict({f'train_step_{k}': v for k, v in loss_dict.items()}, on_step=True, on_epoch=False,
                      batch_size=len(reports), prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        for key, loss in loss_dict.items():
            self.train_loss_metric[f"{key}"].update(loss.detach())
        # Update and log scores for each validation metric:
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='val')

        # Logging:
        self.log_dict({f'val_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports), prog_bar=False, sync_dist=True)

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        for key, loss in loss_dict.items():
            self.val_loss_metric[f"{key}"].update(loss)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        image_ids, patient_ids, reports = batch['image_ids'], batch['patient_ids'], batch['report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        reference_reports = batch['reference_report']
        loss_dict = self(current_study, reports, reference_reports, patient_ids, context, prior_study, mode='test')

        # Logging:
        self.log_dict({f'test_step_{k}': v for k, v in loss_dict.items()}, on_epoch=False, on_step=True,
                      batch_size=len(reports), prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            cur_loss_item = ''
            with torch.no_grad():
                cur_loss_item += ', '.join([f"{k} = {round(v.detach().item(), 2)}" for k, v in loss_dict.items()])
            self.log_once(f"Epoch {self.current_epoch}, testing step {batch_idx}/{self.trainer.num_test_batches[0]}, "
                          f"{cur_loss_item}")
        for key, loss in loss_dict.items():
            if f"{key}" in self.test_loss_metric:
                self.test_loss_metric[f"{key}"].update(loss)

    def on_train_epoch_end(self):
        cur_all_loss = {}
        for key, metric in self.train_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'train_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True,
                      on_step=False, prog_bar=False)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.val_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'val_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False,
                      prog_bar=False)

        if cur_all_loss['loss'] < self.val_min_losses["loss"]:
            self.val_min_losses = {**cur_all_loss, "epoch": self.current_epoch}

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        best_loss_item = ', '.join([f"{k} = {v}" for k, v in self.val_min_losses.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current val loss:"
            f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}\n"
            f"best validation loss: {best_loss_item}\n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """
        cur_all_loss = {}
        for key, metric in self.test_loss_metric.items():
            avg_metric = metric.compute()
            metric.reset()
            cur_all_loss[key] = avg_metric
        self.log_dict({f'test_epoch_{k}': v for k, v in cur_all_loss.items()}, on_epoch=True, on_step=False,
                      prog_bar=False)

        cur_loss_item = ', '.join([f"{k} = {round(v.item(), 2)}" for k, v in cur_all_loss.items()])
        self.log_once(
            "###############################################################\n"

            f"Epoch {self.current_epoch}, test is over, current loss:"
            f"{cur_loss_item}\n"
        )


class TrainLanguageModel(pl.LightningModule):
    def __init__(
            self,
            args: Dict,
            tokenizer: GPT2TokenizerFast,
            logger,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mylog = logger
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.val_best_scores = {
            "best_epoch": -1,
            "best_monitor_metric": -1.0,
        }

        self.train_loss_metric = torchmetrics.MeanMetric()
        self.val_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"])
        self.test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "rouge", "meteor"], save=False)

        self.val_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=16,
            exp_dir=args['project_name'],
        )
        self.test_f1chexbert_metrics = F1CheXbertMetrics(
            chexbert_path=args['chexbert_path'],
            model_path=args['bert_path'],
            mbatch_size=16,
            exp_dir=args['project_name'],
        )
        # Radgraph metrics:
        self.val_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=16,
            exp_dir=args['project_name'],
        )
        self.test_radgraph_metrics = RadGraphMetrics(
            radgraph_path=self.args['radgraph_path'],
            mbatch_size=16,
            exp_dir=args['project_name'],
        )
        # Report logging:
        self.val_report_logger = ReportLogger(exp_dir=args['project_name'], split='val_reports')
        self.test_report_logger = ReportLogger(exp_dir=args['project_name'], split='test_reports')

        # Image Encoder (frozen):
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.config.output_hidden_states = True
        image_dim = self.image_encoder.config.hidden_size
        image_num_layers = self.image_encoder.config.num_hidden_layers
        self.freeze_parameters(self.image_encoder)

        # attention-enhanced fusion network from multi-layer hidden states
        self.layer_fusion = LayerwiseFusion(channel=image_num_layers)

        # Text Encoder to encode context data
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head for each encoder (training)
        self.image_projection = VisualProjectionHead(image_dim, args['hidden_size'] // 2, args['hidden_size'])
        self.text_projection = ProjectionHead(text_dim, args['hidden_size'] // 2, args['hidden_size'])

        # vp_pos_embed for view_position (frozen these parameters in stage 2)
        self.vp2id = json.load(open(args['view_position_dict']))
        self.vp_pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=False)
        # temp_pos_embed for temporal information
        self.temp_pos_embed = nn.Parameter(torch.randn(2, 1, args['hidden_size']), requires_grad=False)
        # define temperature hyper-parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)
        self.layer_norm = nn.LayerNorm(args['hidden_size'])
        self.layer_norm_fusion = nn.LayerNorm(args['hidden_size'])  # LN low-level and high-level visual features

        # temporal_fusion module
        self.temporal_fusion = TemporalFusion(args['hidden_size'], args['temporal_fusion_num_blocks'],
                                              heads=args['num_heads'], dim_head=args['hidden_size'] // 4,
                                              mlp_dim=args['hidden_size'])

        # fusion high-level visual features, low-high-level visual features, knowledge
        self.perceiver = Perceiver(
            byte_dim=args['hidden_size'],  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['num_latents'],  # number of latents
            latent_dim=args['hidden_size'],  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

        self.text_decoder = self.build_text_decoder()

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args['distilgpt2_path'])
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        decoder = transformers.GPT2LMHeadModel(config=config)
        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(len(self.tokenizer))

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def forward(self, *args, **kwargs):
                pass

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        return Decoder()

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    @rank_zero_only
    def log_once(self, message):
        self.mylog.info(message)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # construct train_set and val_set
            self.train_set = FinetuneDataset(self.args, 'train', self.tokenizer)
            self.val_set = FinetuneDataset(self.args, 'test', self.tokenizer)
            print(
                "No. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )
            self.log_once("No. of training & validation examples: {} & {}.".format(
                self.train_set.__len__(), self.val_set.__len__()))
        if stage == "test" or stage is None:  # fit
            self.test_set = FinetuneDataset(self.args, 'test', self.tokenizer)
            print("No. of test examples: {}.".format(self.test_set.__len__()))
            self.log_once("No. of test examples: {}.".format(self.test_set.__len__()))

    def train_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.train_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.val_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        collate_fn = FinetuneCollateFn(self.args, self.image_processor, self.tokenizer.sep_token)
        return DataLoader(
            self.test_set,
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        ft_parameters, pt_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'text_decoder' in name:
                ft_parameters.append(param)
            else:
                pt_parameters.append(param)
        optimiser = torch.optim.AdamW(
            [{'params': pt_parameters, 'lr': self.args['pt_lr']},
             {'params': ft_parameters, 'lr': self.args['ft_lr']}])

        lr_scheduler = ReduceLROnPlateau(optimiser, mode=self.args['monitor_mode'],
                                         factor=0.1, patience=self.args['patience'])
        return {
            "optimizer": optimiser,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': self.args['monitor_metric'],
                'frequency': 1  # the frequency of check
            }
        }

    def tokenization(self, text, device, max_length):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False,
                                max_length=max_length, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        return inputs

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # delete illegal characters
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def obtain_decoder_input_ids(self, inputs):
        decoder_input_ids = inputs['input_ids']
        decoder_attention_mask = inputs['attention_mask'][:, :-1]  # string + [eos]
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        return decoder_input_ids, decoder_attention_mask, label_ids

    def image_encoder_forward(self, images):
        with torch.no_grad():
            outputs = self.image_encoder(images)
            last_hidden_state = outputs['last_hidden_state']
            hidden_states = torch.stack(outputs['hidden_states'][1:], dim=1)  # the first token is [cls] token
        return hidden_states, last_hidden_state

    def obtain_joint_visual_features_forward(self, current_study, prior_study=None):
        # obtain joint_visual features (joint visual features and spatio-temporal visual features)
        # ===================image encoder forward==============================
        hidden_states, last_hidden_state = self.image_encoder_forward(current_study['image'])
        # ===================combine low-level and high-level visual features ===========
        # obtain view_position and temporal embedding
        cur_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in current_study['view_position']]
        cur_pos_embed = torch.cat(cur_pos_embed, dim=0)
        cur_temporal_embed = self.temp_pos_embed[0].repeat(last_hidden_state.shape[0], 1, 1)

        fused_hidden_states = self.layer_fusion(hidden_states)
        # add view position embeddings
        fused_hidden_states = cur_pos_embed + fused_hidden_states
        # image projection
        fused_hidden_states = self.image_projection(fused_hidden_states)  # joint visual features (ln+head)

        # add temporal position embeddings (current view)
        fused_hidden_states = self.layer_norm_fusion(fused_hidden_states + cur_temporal_embed)  # (temp+ln)
        # ===================obtain spatio-temporal visual features =====================
        # add view position embeddings
        cur_vis_feat = cur_pos_embed + last_hidden_state
        # image projection
        cur_vis_feat = self.image_projection(cur_vis_feat)
        # add temporal position embeddings
        cur_vis_feat = cur_vis_feat + cur_temporal_embed

        # fusion prior study using temporal_fusion network
        spatio_temp_feat = torch.empty_like(cur_vis_feat).to(cur_vis_feat)
        if prior_study is not None:
            # fill the spatio_temp_feat using prior scan
            _, pri_last_hidden_state = self.image_encoder_forward(prior_study['image'])
            # view position embedding
            pri_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in prior_study['view_position']]
            pri_last_hidden_state = torch.cat(pri_pos_embed, dim=0) + pri_last_hidden_state
            # image projection
            pri_last_hidden_state = self.image_projection(pri_last_hidden_state)
            # temporal position embedding
            pri_temporal_embed = self.temp_pos_embed[1].repeat(pri_last_hidden_state.shape[0], 1, 1)
            pri_last_hidden_state = pri_temporal_embed + pri_last_hidden_state
            # temporal image fusion
            has_pri_idx = prior_study['pri_idx']
            # temporal_fusion (the first and lasy sublayer have layer-norm)
            temp_visual_features = self.temporal_fusion(cur_vis_feat[has_pri_idx], pri_last_hidden_state)
            spatio_temp_feat[has_pri_idx] = temp_visual_features

            # fill the pri_img_embed using the learnable embed
            no_pri_idx = prior_study['no_pri_idx']
        else:
            no_pri_idx = list(range(cur_vis_feat.shape[0]))
        spatio_temp_feat[no_pri_idx] = self.layer_norm(cur_vis_feat[no_pri_idx])
        return fused_hidden_states, spatio_temp_feat

    def obtain_textual_features(self, reports, device, return_attention_mask=False):
        inputs = self.tokenization(reports, device=device, max_length=self.args['encoder_max_length'])
        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num + 1, 768)
        if not return_attention_mask:
            return text_embed
        else:
            return text_embed, inputs['attention_mask']

    def forward(self, current_study, context, reference_reports=None, prior_study=None, mode='train'):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        # =================== extract uni-modal features =======================
        # obtain joint visual features and spatio-temporal visual features
        joint_feat, spatio_temp_feat = self.obtain_joint_visual_features_forward(current_study, prior_study)
        # obtain context information
        context_embed = self.obtain_textual_features(context, spatio_temp_feat.device)

        # =============== fusion multimodal information using Perceiver (compact)=============
        context_latents = self.perceiver(context_embed)
        spatio_temp_latents = self.perceiver(spatio_temp_feat, latent=context_latents)

        # detail features, context -> global (include prior image) -> fine-grained
        joint_latents = self.perceiver(joint_feat, latent=spatio_temp_latents)
        # inject similar cases into joint_latents
        encoder_outputs = torch.cat([context_latents, spatio_temp_latents, joint_latents], dim=1)

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=encoder_outputs)

        # ===========language modeling loss or generating reports =================
        if mode == 'train':
            report_inputs = self.tokenization(reference_reports, device=spatio_temp_feat.device,
                                              max_length=self.args['max_length'])
            decoder_input_ids, decoder_attention_mask, labels_ids = self.obtain_decoder_input_ids(report_inputs)
            outputs = self.text_decoder.encoder_decoder(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                labels=labels_ids
            )
            return outputs['loss']
        else:
            outputs = self.generate(encoder_outputs)
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # delete illegal characters
            generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
            return generated_reports

    def generate(self, encoder_outputs):
        """
        Autoregressive generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        outputs = self.text_decoder.encoder_decoder.generate(
            max_length=self.args['max_length'],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.args['num_beams'],
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        # Inference:
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        loss = self(current_study, context, reference_reports, prior_study, mode='train')

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_training_batches:
            cur_loss_item = ''
            # cur_loss_item += ', '.join([f"{k} = {round(v.detach().cpu().item(), 2)}" for k, v in loss_dict.items()])
            cur_loss_item += f"{round(loss.detach().cpu().item(), 3)}"
            self.log_once(
                f"Epoch {self.current_epoch}, training step {batch_idx}/{self.trainer.num_training_batches}, "
                f"{cur_loss_item}, lr: {self.optimizers().param_groups[0]['lr']}")

        # update loss through mean_metric
        self.train_loss_metric.update(loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        generated_reports = self(current_study, context, prior_study=prior_study, mode='val')

        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reference_reports)  # remove special tokens

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_val_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, validation step {batch_idx}/{self.trainer.num_val_batches[0]}")

        # # Log reports:
        self.val_report_logger.update(generated_reports, dicom_ids=image_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.val_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.val_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """
        image_ids, reference_reports = batch['image_ids'], batch['reference_report']
        current_study, prior_study, context = batch['current_study'], batch['prior_study'], batch['clinical_context']
        generated_reports = self(current_study, context, prior_study=prior_study, mode='test')

        generated_reports = [text if len(text) > 0 else "..." for text in generated_reports]
        reference_reports = self.obtain_reference_reports(reference_reports)  # remove special tokens

        if batch_idx % self.args['print_step'] == 0 or batch_idx + 1 == self.trainer.num_test_batches[0]:
            self.log_once(
                f"Epoch {self.current_epoch}, test step {batch_idx}/{self.trainer.num_test_batches[0]}")

        # # Log reports:
        self.test_report_logger.update(generated_reports, dicom_ids=image_ids, reference_reports=reference_reports)

        # # Evaluate:
        self.test_f1chexbert_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_coco_metrics.update(generated_reports, reference_reports, ids=image_ids)
        self.test_radgraph_metrics.update(generated_reports, reference_reports, ids=image_ids)

    def on_train_epoch_end(self):
        epoch_loss = self.train_loss_metric.compute()
        self.train_loss_metric.reset()
        self.log_once(
            f"Epoch {self.current_epoch}, Training is over, "
            f"training epoch loss = {epoch_loss}, lr: {self.optimizers().param_groups[0]['lr']}"
            "\n###############################################################"
        )

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        #
        scores = {}
        # F1-radgraph
        output = self.val_radgraph_metrics.compute()
        scores.update(output)
        self.val_radgraph_metrics.reset()

        # chexbert
        output = self.val_f1chexbert_metrics.compute()
        scores.update(output)
        self.val_f1chexbert_metrics.reset()

        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)

        if scores[self.args['monitor_metric']] > self.val_best_scores['best_monitor_metric']:
            self.val_best_scores = {
                "best_epoch": self.current_epoch,
                'best_monitor_metric': scores[self.args['monitor_metric']]
            }

        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"Epoch {self.current_epoch}, Validation is over, current metrics:\n"
            f"best validation epoch: {self.val_best_scores['best_epoch']}, "
            f"best val_metrics: {self.args['monitor_metric']} = {self.val_best_scores['best_monitor_metric']}\n"
            f"{metrics_item} \n"
        )

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.log(1)
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}
        output = self.test_radgraph_metrics.compute()
        scores.update(output)
        self.test_radgraph_metrics.reset()

        output = self.test_f1chexbert_metrics.compute()
        scores.update(output)
        self.test_f1chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        scores['RB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4']
        scores['RC'] = scores['F1-Radgraph-partial'] + scores['chexbert_all_micro_f1']
        scores['RCB'] = scores['F1-Radgraph-partial'] + scores['chen_bleu_4'] + scores['chexbert_all_micro_f1']

        print('\n')
        print(scores)

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        metrics_item = '\n'.join([f'{k}: {v}' for k, v in scores.items()])
        self.log_once(
            "###############################################################\n"
            f"test is over, current metrics:"
            f"{metrics_item} \n"
        )


class TrainLanguageModelOneSample(nn.Module):
    def __init__(
            self,
            args: Dict,
            tokenizer: GPT2TokenizerFast,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        # Image Encoder (frozen):
        self.image_processor = AutoImageProcessor.from_pretrained(args['rad_dino_path'])
        self.image_encoder = AutoModel.from_pretrained(args['rad_dino_path'])
        self.image_encoder.config.output_hidden_states = True
        image_dim = self.image_encoder.config.hidden_size
        image_num_layers = self.image_encoder.config.num_hidden_layers
        self.freeze_parameters(self.image_encoder)

        # attention-enhanced fusion network from multi-layer hidden states
        self.layer_fusion = LayerwiseFusion(channel=image_num_layers)

        # Text Encoder to encode context data
        self.text_encoder = self.build_text_encoder()
        text_dim = self.text_encoder.config.hidden_size
        self.text_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # projection head for each encoder (training)
        self.image_projection = VisualProjectionHead(image_dim, args['hidden_size'] // 2, args['hidden_size'])
        self.text_projection = ProjectionHead(text_dim, args['hidden_size'] // 2, args['hidden_size'])

        # vp_pos_embed for view_position (frozen these parameters in stage 2)
        self.vp2id = json.load(open(args['view_position_dict']))
        self.vp_pos_embed = nn.Parameter(torch.randn(len(self.vp2id), 1, image_dim), requires_grad=False)
        # temp_pos_embed for temporal information
        self.temp_pos_embed = nn.Parameter(torch.randn(2, 1, args['hidden_size']), requires_grad=False)
        # define temperature hyper-parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)
        self.layer_norm = nn.LayerNorm(args['hidden_size'])
        self.layer_norm_fusion = nn.LayerNorm(args['hidden_size'])  # LN low-level and high-level visual features

        # temporal_fusion module
        self.temporal_fusion = TemporalFusion(args['hidden_size'], args['temporal_fusion_num_blocks'],
                                              heads=args['num_heads'], dim_head=args['hidden_size'] // 4,
                                              mlp_dim=args['hidden_size'])

        # fusion high-level visual features, low-high-level visual features, knowledge
        self.perceiver = Perceiver(
            byte_dim=args['hidden_size'],  # byte array dimension
            depth=args['perceiver_num_blocks'],
            # depth of net. depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents=args['num_latents'],  # number of latents
            latent_dim=args['hidden_size'],  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn=1  # number of self attention blocks per cross attention
        )

        self.text_decoder = self.build_text_decoder()

    def build_text_encoder(self):
        enc_config = AutoConfig.from_pretrained(self.args['cxr_bert_path'], trust_remote_code=True)
        enc_config.vocab_size = len(self.tokenizer)
        enc_config.eos_token_id = self.tokenizer.eos_token_id
        enc_config.bos_token_id = self.tokenizer.bos_token_id
        enc_config.pad_token_id = self.tokenizer.pad_token_id
        enc_config.num_hidden_layers = self.args['text_encoder_num_blocks']
        enc_config.max_length = 200
        return AutoModel.from_pretrained(
            self.args['cxr_bert_path'],
            config=enc_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True)

    def build_text_decoder(self):
        config = transformers.GPT2Config.from_pretrained(self.args['distilgpt2_path'])
        config.add_cross_attention = True
        config.is_decoder = True
        config.vocab_size = len(self.tokenizer)
        decoder = transformers.GPT2LMHeadModel(config=config)
        # Resize GPT2 embedding to include padding and beginning of sentence token:
        decoder.resize_token_embeddings(len(self.tokenizer))

        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def forward(self, *args, **kwargs):
                pass

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        return Decoder()

    def freeze_parameters(self, model):
        for para in model.parameters():
            para.requires_grad = False

    def tokenization(self, text, device, max_length):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt', return_token_type_ids=False,
                                max_length=max_length, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        return inputs

    def obtain_reference_reports(self, text):
        inputs = self.tokenizer(text, padding=True, max_length=self.args['max_length'],
                                truncation=True, return_tensors='pt')
        ref_reports = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        # delete illegal characters
        ref_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in ref_reports]
        return ref_reports

    def obtain_decoder_input_ids(self, inputs):
        decoder_input_ids = inputs['input_ids']
        decoder_attention_mask = inputs['attention_mask'][:, :-1]  # string + [eos]
        label_ids = decoder_input_ids[:, 1:].detach().clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        decoder_input_ids = decoder_input_ids[:, :-1]
        decoder_input_ids[decoder_input_ids == self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id
        return decoder_input_ids, decoder_attention_mask, label_ids

    def image_encoder_forward(self, images):
        with torch.no_grad():
            outputs = self.image_encoder(images)
            last_hidden_state = outputs['last_hidden_state']
            hidden_states = torch.stack(outputs['hidden_states'][1:], dim=1)  # the first token is [cls] token
        return hidden_states, last_hidden_state

    def obtain_joint_visual_features_forward(self, current_study, prior_study=None):
        # obtain joint_visual features (joint visual features and spatio-temporal visual features)
        # ===================image encoder forward==============================
        hidden_states, last_hidden_state = self.image_encoder_forward(current_study['image'])
        # ===================combine low-level and high-level visual features ===========
        # obtain view_position and temporal embedding
        cur_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in current_study['view_position']]
        cur_pos_embed = torch.cat(cur_pos_embed, dim=0)
        cur_temporal_embed = self.temp_pos_embed[0].repeat(last_hidden_state.shape[0], 1, 1)

        fused_hidden_states = self.layer_fusion(hidden_states)
        # add view position embeddings
        fused_hidden_states = cur_pos_embed + fused_hidden_states
        # image projection
        fused_hidden_states = self.image_projection(fused_hidden_states)  # joint visual features (ln+head)

        # add temporal position embeddings (current view)
        fused_hidden_states = self.layer_norm_fusion(fused_hidden_states + cur_temporal_embed)  # (temp+ln)
        # ===================obtain spatio-temporal visual features =====================
        # add view position embeddings
        cur_vis_feat = cur_pos_embed + last_hidden_state
        # image projection
        cur_vis_feat = self.image_projection(cur_vis_feat)
        # add temporal position embeddings
        cur_vis_feat = cur_vis_feat + cur_temporal_embed

        # fusion prior study using temporal_fusion network
        spatio_temp_feat = torch.empty_like(cur_vis_feat).to(cur_vis_feat)
        if prior_study is not None:
            # fill the spatio_temp_feat using prior scan
            _, pri_last_hidden_state = self.image_encoder_forward(prior_study['image'])
            # view position embedding
            pri_pos_embed = [self.vp_pos_embed[self.vp2id[vp]].unsqueeze(0) for vp in prior_study['view_position']]
            pri_last_hidden_state = torch.cat(pri_pos_embed, dim=0) + pri_last_hidden_state
            # image projection
            pri_last_hidden_state = self.image_projection(pri_last_hidden_state)
            # temporal position embedding
            pri_temporal_embed = self.temp_pos_embed[1].repeat(pri_last_hidden_state.shape[0], 1, 1)
            pri_last_hidden_state = pri_temporal_embed + pri_last_hidden_state
            # temporal image fusion
            has_pri_idx = prior_study['pri_idx']
            # temporal_fusion (the first and lasy sublayer have layer-norm)
            temp_visual_features = self.temporal_fusion(cur_vis_feat[has_pri_idx], pri_last_hidden_state)
            spatio_temp_feat[has_pri_idx] = temp_visual_features

            # fill the pri_img_embed using the learnable embed
            no_pri_idx = prior_study['no_pri_idx']
        else:
            no_pri_idx = list(range(cur_vis_feat.shape[0]))
        spatio_temp_feat[no_pri_idx] = self.layer_norm(cur_vis_feat[no_pri_idx])
        return fused_hidden_states, spatio_temp_feat

    def obtain_textual_features(self, reports, device, return_attention_mask=False):
        inputs = self.tokenization(reports, device=device, max_length=self.args['encoder_max_length'])
        # obtain the text_encoder forward
        text_embed = self.text_encoder(**inputs)
        text_embed = self.text_projection(text_embed['last_hidden_state'])  # (b, token_num + 1, 768)
        if not return_attention_mask:
            return text_embed
        else:
            return text_embed, inputs['attention_mask']

    def forward(self, item):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """

        # =================== extract uni-modal features =======================
        # obtain joint visual features and spatio-temporal visual features
        joint_feat, spatio_temp_feat = self.obtain_joint_visual_features_forward(item['current_study'], item['prior_study'])
        # obtain context information
        context_embed = self.obtain_textual_features(item['clinical_context'], spatio_temp_feat.device)

        # =============== fusion multimodal information using Perceiver (compact)=============
        context_latents = self.perceiver(context_embed)
        spatio_temp_latents = self.perceiver(spatio_temp_feat, latent=context_latents)

        # detail features, context -> global (include prior image) -> fine-grained
        joint_latents = self.perceiver(joint_feat, latent=spatio_temp_latents)
        # inject similar cases into joint_latents
        encoder_outputs = torch.cat([context_latents, spatio_temp_latents, joint_latents], dim=1)

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=encoder_outputs)

        # ===========generating reports =================
        outputs = self.generate(encoder_outputs)
        generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # delete illegal characters
        generated_reports = [re.sub(r'[^\x20-\x7E]', '', report.strip()) for report in generated_reports]
        return generated_reports

    def generate(self, encoder_outputs):
        """
        Autoregressive generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        outputs = self.text_decoder.encoder_decoder.generate(
            max_length=self.args['max_length'],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.args['num_beams'],
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_result = torch.mean(x, dim=(2, 3), keepdim=True)
        max_result = torch.amax(x, dim=(2, 3), keepdim=True)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=12, reduction=2, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class LayerwiseFusion(nn.Module):
    def __init__(self, channel=12, reduction=4):
        super().__init__()
        self.cbam = CBAMBlock(channel=channel, reduction=reduction)
        self.projection = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.cbam(x)
        x = self.projection(x)
        return x.squeeze(dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class VisualProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)
