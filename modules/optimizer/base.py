from modules.workflow import *
from modules.datas import *
from modules.utils import *
from modules.logging import TqdmLogger

class Optimizer:

    def __init__(
        self,
        train_dataset: Dataset,
        res_dir: str
    ) -> None:
        self.train_dataset = train_dataset
        optimizer_cls = self.__class__.__name__.lower()

        os.makedirs(res_dir, exist_ok = True)
        self.logger = TqdmLogger(f'{train_dataset.name}_{optimizer_cls}_opt', res_dir)
        self.scores_path = os.path.join(res_dir, f'{self.train_dataset.name}_scores.jsonl')
        self.opt_cost = 0
        