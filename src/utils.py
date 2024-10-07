#importa librerias estandars
import os
import sys
from collections import OrderedDict

#importa librerias para normalizacion/calculos/entrenamiento de modelos
import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn
import datetime
import json
from sklearn.model_selection import train_test_split

# Carga el state_dict de un modelo desde una URL o un archivo local y lo devuelve
def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(download_path, map_location=torch.device("cpu"))
    return state_dict


# Restaura el estado de entrenamiento desde el checkpoint más reciente, si existe
def resume_train_state(
    model,
    path: str,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + "/" + "model_store" + "/" + path
        dirs = [base_path + "/" + f.name for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(
            key=os.path.getctime
        )  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.print(f"try to load {dirs[-1]} train stage")
        model = load_pretrain_model(dirs[-1] + "/pytorch_model.bin", model, accelerator)
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(
            f"Load state training success ！Start from {starting_epoch} epoch"
        )
        return model, starting_epoch, step, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Load training status failed ！")
        return model, 0, 0, 0


# Carga un modelo preentrenado desde el path dado y lo asigna al modelo proporcionado
def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print(f"Successfully loaded the training model！")
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load the training model！")
        return model

# Establece una semilla común para asegurar la reproducibilidad en CPU, GPU y numpy
def same_seeds(seed):
    # Establece la estrategia de compartición de memoria para multiprocessing en PyTorch
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Establece la semilla para las operaciones aleatorias en PyTorch en la CPU
    torch.manual_seed(seed) 
    # Si hay GPU disponible, establece la semilla para las operaciones aleatorias en CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # Semilla para una sola GPU
        torch.cuda.manual_seed_all(seed)     # Semilla para todas las GPUs si hay más de una
    # Establece la semilla para la generación de números aleatorios en numpy
    np.random.seed(seed)
    # Activa la búsqueda de las configuraciones más rápidas para las GPU
    torch.backends.cudnn.benchmark = True
     
    # torch.backends.cudnn.deterministic = True



# Inicializa los pesos de las capas lineales y LayerNorm de un modelo
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)  # Inicializa los pesos de las capas lineales
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Inicializa el sesgo de las capas lineales
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)      # Inicializa el sesgo de LayerNorm
        nn.init.constant_(m.weight, 1.0)  # Inicializa los pesos de LayerNorm


# Redirige la salida estándar y de error a la consola y a un archivo de registro si se especifica un directorio de logs
class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open('log_file.txt', 'w', encoding='utf-8')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

        # Solo redirige sys.stdout y sys.stderr si se ha creado log_file correctamente
        if self.log_file is not None:
            sys.stdout = self
            sys.stderr = self

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        if self.console:
            sys.stdout = self.console  # Restaurar sys.stdout
            sys.stderr = sys.__stderr__  # Restaurar sys.stderr original
        if self.log_file is not None:
            self.log_file.close()

