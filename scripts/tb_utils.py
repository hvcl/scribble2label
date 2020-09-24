from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def init_tb_logger(directory, log_file_name):
    log_path = Path(directory, log_file_name)
    tb_logger = SummaryWriter(log_dir=log_path)
    return tb_logger
