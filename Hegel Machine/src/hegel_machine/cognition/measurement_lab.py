"""A continuous process with selectable instruments and a local process backend.

Each probe starts a new experiment from the same initial physical state. The
response is always observed; auxiliary channels are measured only when chosen.
Reusing an experiment's probe id and controls replays that experiment, while a
new probe id creates another realization of the unobserved perturbation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys

from .models import Observation, Probe, TaskSpec


@dataclass(frozen=True)
class MeasurementChannel:
    name: str
    cost_per_row: int
    description: str


CHANNELS = (
    MeasurementChannel("sensor_a", 1, "Read the first auxiliary process instrument."),
    MeasurementChannel("sensor_b", 4, "Read the second auxiliary process instrument."),
)


@dataclass
class MeasurementBatch:
    observations: list[Observation]
    observation_cost: int
    measurement_cost: int
    channels: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(_read_observations(payload["observations"]), payload["observation_cost"],
                   payload["measurement_cost"], tuple(payload["channels"]))


def _read_observations(rows):
    return [Observation(dict(row["features"]), row["target"],
                        tuple(dict(point) for point in row.get("history", ()))) for row in rows]


def _selected(channels):
    selected = tuple(dict.fromkeys(channels))
    unknown = set(selected) - {channel.name for channel in CHANNELS}
    if unknown:
        raise ValueError(f"unknown measurement channels: {sorted(unknown)}")
    return selected


def _public_spec(task_index, channels=()):
    if task_index not in range(4):
        raise ValueError("measurement task index must be 0, 1, 2 or 3")
    selected = _selected(channels)
    return TaskSpec(
        f"measurement_{task_index + 1:02d}",
        "Apply a control sequence and predict the measured process response.",
        ("u", *selected),
        (Probe("initial", tuple({"u": value} for value in (-.8, -.2, .3, .9))),), (),
        feature_bounds={"u": (-1., 1.)}, max_probe_length=64)


class MeasurementLab:
    channels = CHANNELS

    def __init__(self, task_index=0, seed=113):
        self.task_index, self.seed = task_index, seed
        self.spec = _public_spec(task_index)
        self._random = Random(seed + 1009 * task_index)
        self._experiments = {}
        self._state = 0.
        self._exposure = 0
        self.observation_cost = 0
        self.measurement_cost = 0
        self.acquisitions = []

    def spec_for(self, channels=()):
        return _public_spec(self.task_index, channels)

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": self.measurement_cost,
                "queries": len(self.acquisitions)}

    def reset(self):
        """Reset the physical state; previously paid measurements remain recorded."""
        self._state = 0.

    def _experiment(self, probe):
        if probe.cost > self.spec.max_probe_length:
            raise ValueError("probe exceeds the process's maximum experiment length")
        controls = []
        for point in probe.sequence:
            if set(point) != {"u"} or not -1. <= point["u"] <= 1.:
                raise ValueError("the controllable input is u in [-1, 1]")
            controls.append(float(point["u"]))
        key = probe.probe_id, tuple(controls)
        self.reset()
        if key in self._experiments:
            rows, self._state = self._experiments[key]
            return rows
        # These parameters describe the apparatus, not its learning interface.
        decay, coupling, disturbance = ((.65, .45, .8), (.84, .35, .6), (.93, -.3, .45),
                                         (.65, .45, .8))[self.task_index]
        rows = []
        for control in controls:
            perturbation = self._random.uniform(-1., 1.)
            unrelated = self._random.uniform(-1., 1.)
            if self.task_index == 3:
                # An initially dormant second drive activates after sustained
                # operation. Its instrument reads the actual drive, including
                # zero while dormant; experiment replay preserves that regime.
                unrelated = unrelated if self._exposure >= 48 else 0.
            self._state = decay * self._state + coupling * control + disturbance * perturbation
            if self.task_index == 3:
                self._state += .6 * unrelated
            self._exposure += 1
            target = (self._state ** 2 + .1 * control if self.task_index == 1 else
                      self._state + (.15 * control if self.task_index == 2 else 0.))
            rows.append(({"u": control, "sensor_a": perturbation, "sensor_b": unrelated}, target))
        self._experiments[key] = rows, self._state
        return rows

    def acquire(self, probe, channels=()):
        selected = _selected(channels)
        raw = self._experiment(probe)
        history, observations = [], []
        for readings, target in raw:
            features = {"u": readings["u"], **{name: readings[name] for name in selected}}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
        measurement_cost = len(observations) * sum(channel.cost_per_row for channel in self.channels
                                                  if channel.name in selected)
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "channels": list(selected),
                                  "sequence": list(probe.sequence),
                                  "observation_cost": len(observations), "measurement_cost": measurement_cost})
        return MeasurementBatch(observations, len(observations), measurement_cost, selected)

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        """Generate a separate evaluation stream without changing this apparatus."""
        separate = MeasurementLab(self.task_index, seed + 1000003)
        separate._exposure = self._exposure
        controls = Random(seed)
        rows = []
        for index in range(trajectories):
            probe = Probe(f"evaluation_{index}", tuple({"u": controls.uniform(-1., 1.)} for _ in range(length)))
            rows.extend(separate.observe(probe, channels))
        return rows

    def restore(self, acquisitions):
        """Rebuild this simulator from previously executed public commands.

        Replayed readings remain historical evidence and retain their original
        costs. This does not expose or add any measurements to the researcher.
        """
        self.__init__(self.task_index, self.seed)
        for item in acquisitions:
            self.acquire(Probe(item["probe_id"], tuple(item["sequence"])), item["channels"])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ProcessMeasurementLab:
    """The apparatus and its state live in an actual local Python subprocess."""

    channels = CHANNELS

    def __init__(self, task_index=0, seed=113):
        self.task_index, self.seed = task_index, seed
        self.spec = _public_spec(task_index)
        self.acquisitions = []
        self.observation_cost = 0
        self.measurement_cost = 0
        process_environment = dict(os.environ)
        source_directory = str(Path(__file__).resolve().parents[2])
        process_environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (source_directory, process_environment.get("PYTHONPATH", "")) if item)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "hegel_machine.cognition.measurement_lab", "serve",
             "--task-index", str(task_index), "--seed", str(seed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=process_environment)

    @property
    def process_id(self):
        return self._process.pid

    @property
    def costs(self):
        return {"observation_cost": self.observation_cost, "measurement_cost": self.measurement_cost,
                "queries": len(self.acquisitions)}

    def spec_for(self, channels=()):
        return _public_spec(self.task_index, channels)

    def _request(self, operation, **payload):
        self._process.stdin.write(json.dumps({"operation": operation, **payload}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"measurement process ended: {self._process.stderr.read().strip()}")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"])
        return response["result"]

    def acquire(self, probe, channels=()):
        selected = _selected(channels)
        payload = self._request("observe", probe=asdict(probe), channels=list(selected))
        batch = MeasurementBatch.from_dict(payload)
        self.observation_cost += batch.observation_cost
        self.measurement_cost += batch.measurement_cost
        self.acquisitions.append({"probe_id": probe.probe_id, "channels": list(selected),
                                  "sequence": list(probe.sequence),
                                  "observation_cost": batch.observation_cost, "measurement_cost": batch.measurement_cost})
        return batch

    def observe(self, probe, channels=()):
        return self.acquire(probe, channels).observations

    def reset(self):
        self._request("reset")

    def restore(self, acquisitions):
        costs = self._request("restore", acquisitions=acquisitions)
        self.acquisitions = list(acquisitions)
        self.observation_cost, self.measurement_cost = costs["observation_cost"], costs["measurement_cost"]

    def evaluation_rows(self, channels=(), *, seed=1009, length=64, trajectories=4):
        return _read_observations(self._request("evaluate", channels=list(_selected(channels)),
                                              seed=seed, length=length, trajectories=trajectories))

    def close(self):
        if self._process.poll() is None:
            self._request("close")
            self._process.wait(timeout=5)
        self._process.stdin.close()
        self._process.stdout.close()
        self._process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def make_lab(task_index=0, seed=113, *, backend="local"):
    if backend == "local":
        return MeasurementLab(task_index, seed)
    if backend == "process":
        return ProcessMeasurementLab(task_index, seed)
    raise ValueError("measurement backend must be local or process")


def _serve(task_index, seed):
    lab = MeasurementLab(task_index, seed)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "observe":
                probe = request["probe"]
                result = lab.acquire(Probe(probe["probe_id"], tuple(probe["sequence"])),
                                     request.get("channels", ())).to_dict()
            elif operation == "reset":
                lab.reset()
                result = {"status": "reset"}
            elif operation == "restore":
                lab.restore(request["acquisitions"])
                result = lab.costs
            elif operation == "evaluate":
                result = [asdict(row) for row in lab.evaluation_rows(request.get("channels", ()),
                    seed=request.get("seed", 1009), length=request.get("length", 64),
                    trajectories=request.get("trajectories", 4))]
            elif operation == "close":
                print(json.dumps({"result": {"status": "closed"}}), flush=True)
                break
            else:
                raise ValueError(f"unknown apparatus operation: {operation}")
            response = {"result": result}
        except (KeyError, TypeError, ValueError) as error:
            response = {"error": str(error)}
        print(json.dumps(response), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the selectable-instrument process apparatus.")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=113)
    args = parser.parse_args(argv)
    _serve(args.task_index, args.seed)


if __name__ == "__main__":
    main()
