from typing import List

MODEL_FOLDER    = "results/models/"
DEFAULT_NAME    = "dqn-model"
MODEL_EXT       = ".zip"


def get_name_from_path(path: str):
    filename = path.split("/")[-1]

    parts = filename.split(".")

    if len(parts) == 1:
        return parts[0]
    return "".join(parts[:-1])


def get_model_path(argv: List[str]) -> str:
    if len(argv) == 2 and argv[1].endswith(MODEL_EXT):
        return argv[1]
    
    return MODEL_FOLDER + DEFAULT_NAME + MODEL_EXT
