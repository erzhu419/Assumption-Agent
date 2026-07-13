__all__ = [
    "SkillLearnBenchAdapter",
    "SkillLearnProgramCompiler",
    "BaselineArmEvidenceReplayCache",
    "SkillLearnCounterfactualRunner",
    "SkillLearnBackendPool",
    "SkillLearnEvolutionHarness",
    "SkillLearnPrebuiltImageCache",
    "SkillLearnProviderCircuit",
    "SkillLearnResidualMiner",
    "SkillLearnSubprocessBackend",
    "SkillLearnTrialObservation",
    "SkillLearnTrialRequest",
    "TrainingEvidenceReplayCache",
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
        "BaselineArmEvidenceReplayCache",
        "SkillLearnCounterfactualRunner",
        "SkillLearnBackendPool",
        "SkillLearnEvolutionHarness",
        "SkillLearnPrebuiltImageCache",
        "SkillLearnProviderCircuit",
        "SkillLearnResidualMiner",
        "SkillLearnSubprocessBackend",
        "SkillLearnTrialObservation",
        "SkillLearnTrialRequest",
        "TrainingEvidenceReplayCache",
        "TrialVariant",
    }:
        from . import skilllearn_lifecycle

        return getattr(skilllearn_lifecycle, name)
    raise AttributeError(name)
