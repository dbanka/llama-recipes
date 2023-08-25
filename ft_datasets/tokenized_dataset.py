from streaming import StreamingDataset
from typing import Callable, Any, Tuple


def get_tokenized_dataset(dataset_config, tokenizer, split="train"):
    # Create streaming dataset
    if split == "train":
        dataset = StreamingDataset(local=dataset_config.data_path + "/train",
                                   remote=dataset_config.remote_data_path + "/train", shuffle=False)

    else:
        dataset = StreamingDataset(local=dataset_config.data_path + "/test",
                                   remote=dataset_config.remote_data_path + "/test", shuffle=False)

    return dataset
