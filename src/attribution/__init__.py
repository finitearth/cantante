"""Attribution package exports."""

from src.attribution.absolute import AbsoluteAttributer
from src.attribution.relative import RelativeAttributer
from src.attribution.base import BaseAttributer, build_agent_var_mapping
from src.attribution.naive import IdentityAttributer

ALL_ATTRIBUTERS = {
    "naive": IdentityAttributer,
    "intelligent": RelativeAttributer,
    "unconstrained": AbsoluteAttributer,
}


__all__ = [
    "BaseAttributer",
    "IdentityAttributer",
    "RelativeAttributer",
    "AbsoluteAttributer",
    "build_agent_var_mapping",
    "ALL_ATTRIBUTERS",
]
