"""Text-only access to the official DiscoveryWorld environment.

Run one live laboratory per process. The agent receives public UI JSON and
official action results; the separate ``score`` entry point is for evaluation.
No environment observations are reconstructed from hidden object properties.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib
import os
from pathlib import Path
import sys


class DiscoveryWorldLab:
    def __init__(self, task: dict, runtime_path, workspace):
        self.task = deepcopy(task)
        self.runtime_path = Path(runtime_path).expanduser().resolve()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not (self.runtime_path / "discoveryworld" / "DiscoveryWorldAPI.py").is_file():
            raise FileNotFoundError(f"DiscoveryWorld runtime missing from {self.runtime_path}")
        sys.path.insert(0, str(self.runtime_path))
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        # Scenario setup prints internal generation details. They are not an
        # observation and must not enter the agent's textual action feedback.
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            module = importlib.import_module("discoveryworld.DiscoveryWorldAPI")
            if not Path(module.__file__).resolve().is_relative_to(self.runtime_path):
                raise RuntimeError("DiscoveryWorld was already imported from a different runtime")
            self._api = module.DiscoveryWorldAPI(threadID=os.getpid())
            # loadScenario clears PNGs here. Isolate that operation from other
            # episodes; text observations below never produce any frame files.
            self._api.FRAME_DIR = str(self.workspace / "discoveryworld_frames")
            loaded = self._api.loadScenario(scenarioName=task["scenario_name"],
                difficultyStr=task["difficulty"], randomSeed=task["random_seed"], numUserAgents=1)
            if not loaded:
                raise ValueError(f"DiscoveryWorld could not load {task['scenario_name']} / {task['difficulty']}")
            self._refresh_observation()
        # The official loader performs one initialization tick, which is not
        # an action purchased by this agent.
        self.initial_world_steps = self._observation["world_steps"]
        self.environment_steps = 0
        self.agent_actions = 0

    def _refresh_observation(self):
        ui = self._api.ui[0]
        # The vendor's JSON USE path can enter dialog without updating its UI
        # cache. Apply the same public-menu refresh as TALK / keyboard USE;
        # clear the cache when a choice has ended the conversation.
        if ui.currentAgent.isInDialog():
            text, options = ui.currentAgent.getAgentInDialogWith().dialogTree.getCurrentDialog()
            ui.dialogToDisplay = {"dialogText": text, "dialogOptions": options}
        else:
            ui.dialogToDisplay = None
        # Render once: this consumes one queued extended message, without PNGs.
        self._observation = ui.renderJSON()
        self._api.taskProgress = self._observation["taskProgress"]
        self._api.steps = self._observation["world_steps"]

    def public_observation(self) -> dict:
        """Return the last public observation without consuming messages again."""
        return deepcopy(self._observation)

    def capabilities(self) -> dict:
        actions = deepcopy(self._api.listKnownActions())
        accessible = {obj["uuid"] for field in ("accessibleEnvironmentObjects", "inventoryObjects")
                      for obj in self._observation.get(field, [])}
        interaction_targets = []
        if "TALK" in actions:
            # The public UI labels nearby agent roles as "<name> uuid <id>".
            # Equipment can have this role too; use the published role instead
            # of guessing from names or inspecting hidden object attributes.
            for label in self._observation.get("nearbyAgents", {}).get("list_of_agents", {}):
                name, _, identifier = label.rpartition(" uuid ")
                uuid = int(identifier)
                interaction_targets.append({"uuid": uuid, "name": name, "role": "agent",
                    "action": {"action": "TALK", "arg1": uuid}, "accessible": uuid in accessible})
        return {"actions": actions,
                "action_format": self._api.additionalActionDescriptionString(),
                "teleport_locations": deepcopy(self._api.listTeleportLocationsDict()),
                "interaction_targets": interaction_targets,
                "dialog": "When dialog_box.is_in_dialog is true, submit {'chosen_dialog_option_int': N} "
                          "using a listed option number instead of a normal action."}

    def act(self, action: dict) -> dict:
        """Submit one official action and advance the world exactly once."""
        if not isinstance(action, dict):
            raise ValueError("DiscoveryWorld actions must be JSON objects")
        before = self._observation["world_steps"]
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
            feedback = self._api.performAgentAction(agentIdx=0, actionJSON=deepcopy(action))
            tick = self._api.tick()
            self._refresh_observation()
        self.agent_actions += 1
        after = self._observation["world_steps"]
        self.environment_steps = after - self.initial_world_steps
        return {"action": deepcopy(action), "feedback": feedback, "tick": tick,
                "observation": self.public_observation(), "environment_steps": self.environment_steps,
                "step_cost": after - before, "agent_actions": self.agent_actions}

    def score(self) -> dict:
        """Evaluation only: official completion and procedural-score summaries."""
        cards = self._api.getTaskScorecard() or []
        summaries = [{name: deepcopy(card[name]) for name in (
            "taskName", "completed", "completedSuccessfully",
            "scoreNormalized", "score", "maxScore") if name in card} for card in cards]
        return {"tasks": summaries,
                "completed": bool(cards) and all(card["completed"] for card in cards),
                "completedSuccessfully": bool(cards) and all(card["completedSuccessfully"] for card in cards),
                "scoreNormalized": sum(card["scoreNormalized"] for card in cards) / len(cards) if cards else 0.,
                "environment_steps": self.environment_steps, "agent_actions": self.agent_actions,
                "world_steps": self._observation["world_steps"]}

    def close(self):
        if self._api is not None:
            self._api = None
            import pygame
            pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
