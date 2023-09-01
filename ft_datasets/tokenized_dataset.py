from streaming import Stream, StreamingDataset
from typing import Callable, Any, Tuple

def create_stream_config(dataset_config, split):
    stream_prop = 1 / len(dataset_config.remote_streams)
    print(
            f"Creating '{split}' data stream from following sources\n"\
            f"{', '.join(dataset_config.remote_streams)}"
            )
    print(f"Proportion per stream - {stream_prop}")
    return [
        Stream(remote=stream + "/" + split, 
               local=dataset_config.data_path + "/" + split + f"/stream_{i}", 
               proportion=stream_prop) 
        for i, stream in enumerate(dataset_config.remote_streams)
        ]

def get_tokenized_dataset(dataset_config, tokenizer, split="train"):
    # Create streaming dataset
    if split == "train":
        dataset = StreamingDataset(
            local=f"{dataset_config.data_path}/{split}",
            remote=f"{dataset_config.remote_data_path}/combined-{split}-data-stream",
            shuffle=True,
            shuffle_seed=42,
            cache_limit='100gb'
        )
    else:
        dataset = StreamingDataset(
            local=f"{dataset_config.data_path}/{split}",
            remote=f"{dataset_config.remote_data_path}/combined-{split}-4096-data-stream",
            shuffle=False,
            cache_limit='1gb'
        )
    return dataset
