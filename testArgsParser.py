# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:21:23 2020

@author: yangy
"""

import argparse


def get_args():
    
    parser = argparse.ArgumentParser(description='Argument Parser for Downstream Tasks of the S3PLR project.')
    
    # required
    parser.add_argument('--run',  choices=['phone_linear', 'phone_1hidden', 'phone_concat', 'speaker_frame', 'speaker_utterance'], help='select task.', required=True)

    # upstream settings
    parser.add_argument('--ckpt', default='', type=str, help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--upstream', choices=['dual_transformer', 'transformer', 'apc', 'baseline'], default='baseline', help='Whether to use upstream models for speech representation or fine-tune.', required=False)
    parser.add_argument('--input_dim', default=0, type=int, help='Input dimension used to initialize transformer models', required=False)
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine tune the transformer model with downstream task.', required=False)
    parser.add_argument('--weighted_sum', action='store_true', help='Whether to use weighted sum on the transformer model with downstream task.', required=False)
    parser.add_argument('--dual_mode',choices=['phone', 'speaker', 'phone speaker'], default='phone', help='Whether to use weighted sum on the transformer model with downstream task.', required=False)
    
    # Options
    parser.add_argument('--name', default=None, type=str, help='Name of current experiment.', required=False)
    parser.add_argument('--config', default='config/downstream.yaml', type=str, help='Path to downstream experiment config.', required=False)
    parser.add_argument('--phone_set', choices=['cpc_phone', 'montreal_phone'], default='cpc_phone', help='Phone set for phone classification tasks', required=False)
    parser.add_argument('--expdir', default='', type=str, help='Path to store experiment result, if empty then default is used.', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--pi', default='3.14', type=float, help='pi.')

    # parse
    args = parser.parse_args()
    # set additional attributes
    setattr(args, 'gpu', not args.cpu)
    setattr(args, 'task', args.phone_set if 'phone' in args.run else 'speaker')
    
    return args


def main():
    args = get_args()
    print(args)
    
if __name__ == '__main__':
    main()