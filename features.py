from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class Feature(ABC):
    name: str
    task: str

    def __init__(self, name: str, task: str = "regression"):
        self.name = name
        self.task = task

    @abstractmethod
    def extract(self, state: str) -> float:
        raise NotImplementedError

    def label_value(self, label_dict: Dict[str, Any], state: str) -> float:
        if label_dict is None:
            return self.extract(state)
        if self.name in label_dict:
            value = label_dict[self.name]
            if value is None:
                return self.extract(state)
            return float(value)
        return self.extract(state)


class VariableCountFeature(Feature):
    def __init__(self):
        super().__init__(name="variable_count", task="regression")

    def extract(self, state: str) -> float:
        if not state or "⊢" not in state:
            return 0.0

        local_context = state.split("⊢")[0].strip()
        if not local_context:
            return 0.0

        count = 0
        lines = local_context.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('case'):
                continue

            if ':' in line:
                names_part = line.split(':')[0].strip()
                clean_names = re.sub(r'[\{\}\[\]]', '', names_part).strip()
                vars_on_line = clean_names.split()
                count += len(vars_on_line)

        return float(count)


class BinaryLogicFeature(Feature):
    def __init__(self, name: str, predicate: Callable[[str], bool]):
        super().__init__(name=name, task="binary")
        self.predicate = predicate

    def extract(self, state: str) -> float:
        return 1.0 if self.predicate(state) else 0.0


def has_quantifier(state: str) -> bool:
    return '∀' in state or '∃' in state


def has_equality(state: str) -> bool:
    if "⊢" not in state:
        return False
    goal = state.split("⊢", 1)[1]
    return '=' in goal


def has_inductive_structure(state: str) -> bool:
    if not state:
        return False
    state_lower = state.lower()
    return bool(
        re.search(r"\binductive\b", state_lower)
        or re.search(r"\bcase(s)?\b", state_lower)
        or re.search(r"\binduction\b", state_lower)
    )


FEATURES: List[Feature] = [
    VariableCountFeature(),
    BinaryLogicFeature("quantifier_presence", has_quantifier),
    BinaryLogicFeature("equality_vs_inequality", has_equality),
    BinaryLogicFeature("inductive_structure", has_inductive_structure),
]
