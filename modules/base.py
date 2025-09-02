from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Generic, TypeVar, ClassVar, Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '.')

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
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

class Messages(GenericList[Message]):
    source: Base_LLM = None
    ROLE_TO_CLASS: ClassVar[dict] = {}

    def __init__(self, messages: List[Message]):
        self._items = messages.copy()
        self._auto_convert = True
        self._check_and_convert()
    
    def _check_and_convert(self) -> None:
        if not self._auto_convert or not self._items:
            return
        last_role = self._items[-1].role
        target_class = self.ROLE_TO_CLASS.get(last_role)
        if target_class and not isinstance(self, target_class):
            self.__class__ = target_class
    
    def append(self, message: Message) -> None:
        self._items.append(message)
        self._check_and_convert()
    
    def pop(self, idx: int = -1) -> Message:
        msg = self._items.pop(idx)
        self._check_and_convert()
        return msg

class SystemMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

class UserMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

class AssistantMessages(Messages):
    def __init__(self, messages: List[Message]):
        super().__init__(messages)

Messages.ROLE_TO_CLASS = {
    'system': SystemMessages,
    'user': UserMessages,
    'assistant': AssistantMessages
}

@dataclass
class EvalScore:
    criterion: str
    score: int | float
    reason: str

class EvalScores(GenericList[EvalScore]):
    source: Base_LLM = None
    messages: Messages = None

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