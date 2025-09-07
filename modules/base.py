from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '..')

from modules.models import Base_LLM

T = TypeVar('T')

class GenericList(Generic[T]):

    def __init__(self, items: List[T]): self._items = items
    def __len__(self) -> int: return len(self._items)
    def __getitem__(self, idx: int) -> T: return self._items[idx]
    def __iter__(self): yield from self._items
    def append(self, item: T) -> None: self._items.append(item)
    def pop(self, idx: int = -1) -> T: return self._items.pop(idx)
    def to_json(self) -> list: return [_item.__dict__ for _item in self._items]
    def __eq__(self, other: 'GenericList[T]'): return all(a == b for a, b in zip(self._items, other._items))
    def __add__(self, other: 'GenericList[T]'): return GenericList(self._items + other._items)

@dataclass
class EvalScore:
    criterion: str
    score: int | float
    reason: str

class EvalScores(GenericList[EvalScore]):
    source: Base_LLM = None

    def __init__(self, items: List[EvalScore] | List[Dict[str, int | float]]):
        if len(items) == 0:
            self._items = []
        elif isinstance(items[0], dict):
            self._items = [EvalScore(**score) for score in items]
        elif isinstance(items[0], EvalScore):
            self._items = items.copy()
        else:
            raise TypeError('Invalid score type.')

    def sum(self) -> float:
        return sum([score.score for score in self._items])

    def get_score(self, criterion: str) -> Optional[EvalScore]:
        for score in self._items:
            if score.criterion == criterion:
                return score
        return None

    def update(self, other: 'EvalScores') -> None:
        criterion_idxs = {scores.criterion: idx for idx, scores in enumerate(self._items)}
        for scores in other:
            if scores.criterion in criterion_idxs:
                self._items[criterion_idxs[scores.criterion]] = scores
            else:
                self.append(scores)

MessagesState = Literal['system', 'user', 'assistant', 'scored']
@dataclass
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

class Messages(GenericList[Message]):
    source: Base_LLM = None
    scores: EvalScores = None
    meta_data: dict = {}
    cost: Dict[str, float] = {}

    def __init__(self, items: List[Message] | List[Dict[str, str]]):
        if len(items) == 0:
            self._items = []
        elif isinstance(items[0], dict):
            self._items = [Message(**message) for message in items]
        elif isinstance(items[0], Message):
            self._items = items.copy()
        else:
            raise TypeError('Invalid message type.')

    @property
    def state(self) -> MessagesState:
        last_role = self._items[-1].role
        if last_role == 'assistant' and self.scores is not None:
            return 'scored'
        else:
            return last_role
    
    def append(self, message: Message) -> None:
        if self.state == 'scored':
            self.scores = None
        if (self.state == 'system' and message.role != 'user') or \
            (self.state == 'user' and message.role != 'assistant') or \
            (self.state == 'assistant' and message.role != 'user'):
            raise ValueError(f'Failed to append {message.__dict__}, state={self.state}')
        self._items.append(message)
    
    def pop(self, idx: int = -1) -> Message:
        if self.state == 'scored':
            self.scores = None
        msg = self._items.pop(idx)
        return msg
    
    def copy(self) -> 'Messages':
        return deepcopy(self)