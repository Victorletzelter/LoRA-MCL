#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import logging
import torch


pylog = logging.getLogger(__name__)

######
import sys
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.join(os.environ['PROJECT_ROOT'], 'metrics'))
from aac_metrics_custom.functional import mh_evaluate
from utils_diversity import eval_div_stats

class MH_AllMetrics:
    def __init__(self, is_tokenized, tokenizer, num_hypotheses, num_return_sequences, settings):
        self.is_tokenized = is_tokenized
        self.tokenizer = tokenizer
        self.num_hypotheses = num_hypotheses
        self.num_return_sequences = num_return_sequences
        self.settings = settings

    def __call__(self, outputs):
        return self.mh_aac_metrics(outputs, self.tokenizer, self.num_hypotheses)
    
    def mh_decode_gt_captions(self, outputs, tokenizer):

        gt_captions = []

        all_gt_captions = []

        for i_ex in range(outputs['predictions'].shape[0]): # Loop over all examples times number of annotations per example
            
            # Management of the labels. They are added to a list until we reach the end of the file.
            gt_ = tokenizer.decode(outputs['label_ids'][i_ex,:])
            gt_captions.append(gt_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))

            if i_ex == len(outputs['filenames'])-1 or outputs['filenames'][i_ex+1] != outputs['filenames'][i_ex] : # Last example for current audio
                all_gt_captions.append(gt_captions)
                gt_captions = []  

        return all_gt_captions

    def mh_process_predictions(self, outputs, tokenizer, num_hypotheses):

        all_pred_captions = {}

        for hypothesis_idx in range(num_hypotheses):
            all_pred_captions[f'{hypothesis_idx}'] = []

        for i_ex in range(outputs['predictions'].shape[0]): # Loop over all examples times number of annotations per example
            # Management of the predictions: Only the predictions associated with the first duplicate of the file are added
            if i_ex == 0 or outputs['filenames'][i_ex-1] != outputs['filenames'][i_ex] :  
                for hypothesis_idx in range(num_hypotheses): # Loop over the hypotheses
                    pred_ = tokenizer.decode(outputs['predictions'][i_ex,hypothesis_idx,:])
                    all_pred_captions[f'{hypothesis_idx}'].append(pred_.replace('<|pad|>', '').replace('<|endoftext|>', '').replace('</s>', '').replace('<s>', '').replace('<pad>', ''))

        return all_pred_captions

    def mh_aac_metrics(self, outputs, tokenizer, num_hypotheses):  

        if self.is_tokenized is True : 
            all_gt_captions = self.mh_decode_gt_captions(outputs=outputs, tokenizer=tokenizer) 
            all_pred_captions = self.mh_process_predictions(outputs=outputs, tokenizer=tokenizer, num_hypotheses=num_hypotheses) # all_pred_captions is a dictionary with keys '0', '1', '2', ... and values being lists of predictions of length outputs['predictions'].shape[0]/5
        else:
            all_gt_captions = outputs['GT']
            all_pred_captions = outputs['decoded_predictions']
            subset = outputs['datasubset'].split('_')[1]
            dataset_name = outputs['datasubset'].split('_')[0]

        total_metrics = {}

        if 'compute_diversity_metrics' in self.settings['model']['lm']['generation'] and self.settings['model']['lm']['generation']['compute_diversity_metrics'] is True and (num_hypotheses > 1 or self.num_return_sequences > 1) :
            
            preds_n = self.compute_idcaps(outputs)
            if (num_hypotheses > 1 or self.num_return_sequences > 1):
                Vocab, div_1, div_2, mBLeu_4, scrperimg, adiv_1, adiv_2 = eval_div_stats(preds_n, compute_mbleu=True)
            else :
                Vocab, div_1, div_2, mBLeu_4, scrperimg, adiv_1, adiv_2 = eval_div_stats(preds_n, compute_mbleu=False)

            total_metrics['diversity_1'] = div_1
            total_metrics['diversity_2'] = div_2
            total_metrics['mbleu_4'] = mBLeu_4
            total_metrics['vocabulary_size'] = Vocab

        # Computation for the metrics as in https://github.com/Labbeti/aac-metrics
        if self.settings['model']['lm']['generation']['compute_aac_scores'] is True : 
            corpus_scores, sentences_scores, full_scores = mh_evaluate(candidates=all_pred_captions,
                                                                        mult_references=all_gt_captions, 
                                                                        metrics = ["bert_score","bleu","bleu_1","bleu_2","bleu_3","bleu_4","cider_d","fense","meteor","rouge_l","sbert_sim","spice","spider","spider_fl","vocab"],
                                                                        # metrics=["spider"],
                                                                          return_full_scores=self.settings['model']['lm']['generation']['return_full_scores'],
                                                                            cache_path=self.settings['paths']['cache_path']) 

            if self.settings['model']['lm']['generation']['return_full_scores'] is True :
                for key in full_scores :
                    total_metrics[key+'_full_scores'] = full_scores[key]

            for key in corpus_scores :
                total_metrics['aac_oracle_'+key] = corpus_scores[key]
            
        return {
            key.lower(): value for key, value in total_metrics.items()
        }, all_gt_captions, all_pred_captions

    def compute_idcaps(self, outputs) :     
        captured_files = []
        preds_n = dict()
        for i, filename in enumerate(outputs['filenames']):
            if filename in captured_files:
                continue
            preds_n[filename] = []
            captured_files.append(filename)
            num_hypotheses = len(outputs['decoded_predictions'])
            pred_captions = [outputs['decoded_predictions'][f'{hypothesis_idx}'][i] for hypothesis_idx in range(num_hypotheses)]
            # pred_captions = outputs['predictions'][i] # List of n predicted captions.
            for decoded_sentence in pred_captions:

                preds_n[filename] = preds_n[filename] +[{
                            "audio_id": filename,
                            "caption": decoded_sentence
                        }]
        return preds_n