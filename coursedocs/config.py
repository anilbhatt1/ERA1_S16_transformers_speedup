from pathlib import Path

def get_config():
    return {
        "batch-size": 2048*2, #s16
        "num_epochs": 20,
        "lr": 10**-3,   #s16
        "seq_len": 160, #s16
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr", #s16
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": False,   #s16
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)