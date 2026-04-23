from .chemqa import ChemQARunner
from .single_llm import SingleLLMRunner


def build_runner(*, runner_kind: str, chemqa_runner_cls=ChemQARunner, single_llm_runner_cls=SingleLLMRunner, **kwargs):
    if runner_kind == "chemqa":
        return chemqa_runner_cls(**kwargs)
    if runner_kind == "single_llm":
        return single_llm_runner_cls(**kwargs)
    raise ValueError(f"Unsupported runner kind: {runner_kind}")


__all__ = [
    "ChemQARunner",
    "SingleLLMRunner",
    "build_runner",
]
