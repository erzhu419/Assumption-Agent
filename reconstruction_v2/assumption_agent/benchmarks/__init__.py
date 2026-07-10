__all__ = [
    "SkillLearnBenchAdapter",
    "SkillLearnProgramCompiler",
    "SkillLearnCounterfactualRunner",
    "SkillLearnBackendPool",
    "SkillLearnEvolutionHarness",
    "SkillLearnPrebuiltImageCache",
    "SkillLearnResidualMiner",
    "SkillLearnSubprocessBackend",
    "SkillLearnTrialObservation",
    "SkillLearnTrialRequest",
    "TrialVariant",
]


def __getattr__(name: str):
    if name == "SkillLearnBenchAdapter":
        from .skilllearnbench import SkillLearnBenchAdapter

        return SkillLearnBenchAdapter
    if name == "SkillLearnProgramCompiler":
        from .skilllearn_compiler import SkillLearnProgramCompiler

        return SkillLearnProgramCompiler
    if name in {
        "SkillLearnCounterfactualRunner",
        "SkillLearnBackendPool",
        "SkillLearnEvolutionHarness",
        "SkillLearnPrebuiltImageCache",
        "SkillLearnResidualMiner",
        "SkillLearnSubprocessBackend",
        "SkillLearnTrialObservation",
        "SkillLearnTrialRequest",
        "TrialVariant",
    }:
        from . import skilllearn_lifecycle

        return getattr(skilllearn_lifecycle, name)
    raise AttributeError(name)
