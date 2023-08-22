from streaming import StreamingDataset
from typing import Callable, Any, Tuple


class TokenisedDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 shuffle_seed:int,
                 shuffle_algo:str,
                 ) -> None:
        super().__init__(local=local, remote=remote, shuffle=shuffle, shuffle_seed=shuffle_seed,
                         shuffle_algo= shuffle_algo)


    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
        obj['input_ids'] = obj['input_ids'].tolist()
        obj['attention_mask'] = obj['attention_mask'].tolist()
        obj['labels'] = obj['labels'].tolist()
        return obj


def get_tokenized_dataset(dataset_config, tokenizer, split="train"):
    # Create streaming dataset
    if split == "train":
        dataset = TokenisedDataset(local=dataset_config.data_path + "/train",
                                   remote=dataset_config.remote_data_path + "/train", shuffle=True,
                                   shuffle_seed=42, shuffle_algo= "py1s")

    else:
        dataset = TokenisedDataset(local=dataset_config.data_path + "/test",
                                   remote=dataset_config.remote_data_path + "/test", shuffle=False)

    return dataset
