from tqdm import tqdm
from copy import deepcopy
import random
import functools
from typing import Literal, List, Dict, Set, Generic, TypeVar, Optional, Any, get_args
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '..')

T = TypeVar('T')

class GenericList(Generic[T]):
    def __init__(self, items: List[T] | List[Dict[str, Any]]):
        self._item_type = self.__orig_bases__[0].__args__[0]
        if len(items) == 0: self._items = []
        elif isinstance(items[0], dict): self._items = [self._item_type(**item) for item in items]
        elif isinstance(items[0], self._item_type): self._items = items.copy()
        else: raise TypeError(f'Invalid item type: {items[0].__class__.__name__}.')
    def __len__(self) -> int: return len(self._items)
    def __getitem__(self, idx: int) -> T: return self._items[idx]
    def __iter__(self): yield from self._items
    def append(self, item: T) -> None: self._items.append(item)
    def pop(self, idx: int = -1) -> T: return self._items.pop(idx)
    def to_json(self) -> list: return [_item.__dict__ for _item in self._items]
    def __eq__(self, other: 'GenericList[T]'): return all(a == b for a, b in zip(self._items, other._items))
    def __add__(self, other: 'GenericList[T]'): return GenericList(self._items + other._items)

@dataclass
class Scenario:
    task: str
    description: str

@dataclass
class Criterion:
    name: str
    description: str
    levels: List[str]

class Criteria(GenericList[Criterion]):
    
    def __getitem__(self, key: int | str) -> Optional[Criterion]:
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            for item in self._items:
                if item.name == key:
                    return item
            return None
        else:
            raise TypeError(f'Invalid key type: {key}.')

@dataclass
class MetaData:
    id: str
    task: str
    scenario: Scenario
    criteria: Criteria

@dataclass
class EvalScore:
    criterion: str
    score: int | float
    reason: str

class EvalScores(GenericList[EvalScore]):
    source: str = None
        
    @property
    def names(self) -> List[str]:
        return [score.criterion for score in self._items]

    def sum(self) -> float:
        return sum([score.score for score in self._items])

    def get_score(self, criterion_name: str) -> Optional[EvalScore]:
        for score in self._items:
            if score.criterion == criterion_name:
                return score
        return None
    
    def __getitem__(self, key: int | str) -> Optional[EvalScore]:
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            for item in self._items:
                if item.criterion == key:
                    return item
            return None
        else:
            raise TypeError(f'Invalid key type: {key}.')

    def update(self, other: 'EvalScores') -> None:
        criterion_idxs = {scores.criterion: idx for idx, scores in enumerate(self._items)}
        for scores in other:
            if scores.criterion in criterion_idxs:
                self._items[criterion_idxs[scores.criterion]] = scores
            else:
                self.append(scores)
    
    def deepcopy(self) -> 'EvalScores':
        return deepcopy(self)

MessagesState = Literal['system', 'user', 'assistant', 'scored']
@dataclass
class Message:
    role: Literal['system', 'user', 'assistant']
    content: str

class Messages(GenericList[Message]):
    metadata: MetaData
    source: str = None
    scores: EvalScores = None
    cost: Dict[str, float] = {}

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
        message = self._items.pop(idx)
        return message
    
    def deepcopy(self) -> 'Messages':
        return deepcopy(self)