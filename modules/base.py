from tqdm import tqdm
from copy import deepcopy
import json
import functools
from typing import Literal, List, Dict, Set, Generic, TypeVar, Optional, Any, Tuple
from dataclasses import dataclass, fields, is_dataclass

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
    def __eq__(self, other: 'GenericList[T]') -> bool: return all(a == b for a, b in zip(self._items, other._items))
    def __add__(self, other: 'GenericList[T]') -> 'GenericList[T]': return GenericList(self._items + other._items)
    def append(self, item: T) -> None: self._items.append(item)
    def pop(self, idx: int = -1) -> T: return self._items.pop(idx)
    def to_list(self) -> list:
        return [_item.__dict__ for _item in self._items]
    def to_dict(self) -> dict:
        data = {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
        data['_items'] = self.to_list()
        return data
    @classmethod
    def from_dict(cls, data: dict) -> 'GenericList[T]':
        list_obj = cls(data.get('_items'))
        for key, value in data: setattr(list_obj, key, value)
        return list_obj
    def to_md(self, indent: int = 0) -> str:
        return '\n'.join([
            '  ' * indent + f'- [{idx}]\n{item.to_md(indent + 1)}'
            for idx, item in enumerate(self._items)
        ])

class DataClassMixin:
    
    def to_md(self, indent: int = 0) -> str:
        indent_space = '  ' * indent
        result = []
        for field in fields(self):
            value = getattr(self, field.name)
            if is_dataclass(value):
                result.append(f'{indent_space}- {field.name}:')
                result.append(value.to_md(indent + 1))
            elif isinstance(value, (list, tuple)):
                result.append(f'{indent_space}- {field.name}:')
                for i, item in enumerate(value):
                    if is_dataclass(item):
                        result.append(f'{indent_space}  - ')
                        result.append(item.to_md(indent + 1))
                    else:
                        result.append(f'{indent_space}  - {self._format_value(item)}')
            elif isinstance(value, dict):
                result.append(f'{indent_space}- {field.name}:')
                for k, v in value.items():
                    if is_dataclass(v):
                        result.append(f'{indent_space}  - {k}:')
                        result.append(v.to_md(indent + 1))
                    else:
                        result.append(f'{indent_space}  - {k}: {self._format_value(v)}')
            else:
                result.append(f'{indent_space}- {field.name}: {self._format_value(value)}')
        return '\n'.join(result)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return 'null'
        elif isinstance(value, (str, int, float, bool)):
            return str(value)
        else:
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)

@dataclass
class Scenario(DataClassMixin):
    task: str
    description: str

@dataclass
class Criterion(DataClassMixin):
    name: str
    description: str
    rules: List[str]

class Criteria(GenericList[Criterion]):

    @property
    def names(self) -> List[str]:
        return [criterion.name for criterion in self._items]
    
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
class MetaData(DataClassMixin):
    id: str
    language: Literal['zh', 'en']
    task: str
    scenario: Scenario
    criteria: Criteria

@dataclass
class EvalScore(DataClassMixin):
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
class Message(DataClassMixin):
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