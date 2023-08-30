# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class tokenized_dataset:
    dataset: str = "tokenized_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/tokenized_data"
    remote_data_path: str = "s3://716533421362-spx-data/phenom-llm-data/streaming-data/pup"
    input_length: int = 4096
    remote_streams: list = ['s3://716533421362-spx-data/phenom-llm-data/streaming-data/annotation-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/crm-profiles',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/hiring-status-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/idea-cand-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/jobs-summary-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/linkup-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/misc',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/profile-matching-data',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/pup',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/resume-freetext',
        's3://716533421362-spx-data/phenom-llm-data/streaming-data/resume-html',
        # 's3://716533421362-spx-data/phenom-llm-data/streaming-data/job-parser-data',
        # 's3://716533421362-spx-data/phenom-llm-data/streaming-data/fitscore-data'
        ]
