"""Asynchronous local experiments whose tickets survive a research restart."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

from .measurement_lab import CHANNELS, MeasurementBatch, make_lab
from .models import Probe
from .checkpoint_storage import write_text_incrementally


def _write_result(path, result):
    write_text_incrementally(path, json.dumps(result))


class DelayedFeedback:
    """Reserve parallel experiments, then settle and deliver each outcome once.

    A dispatched experiment costs one attempt, including failure/cancellation.
    Observation and instrument costs are paid for successful delivered batches;
    unsuccessful jobs release their reserved rows and instrument units.
    """

    def __init__(self, directory, *, observation_budget=256, measurement_budget=1024,
                 experiment_budget=32, state=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._processes = {}
        self.state = deepcopy(state) if state is not None else {
            "directory": str(self.directory), "limits": {"observation_cost": observation_budget,
                "measurement_cost": measurement_budget, "experiment_cost": experiment_budget},
            "costs": {"observation_cost": 0, "measurement_cost": 0, "experiment_cost": 0},
            "tickets": {}, "next_ticket": 0}

    def to_dict(self):
        return deepcopy(self.state)

    @classmethod
    def from_dict(cls, state):
        return cls(state["directory"], state=state)

    @property
    def costs(self):
        return dict(self.state["costs"])

    @property
    def reserved(self):
        pending = [ticket["reservation"] for ticket in self.state["tickets"].values() if not ticket["settled"]]
        return {name: sum(item[name] for item in pending) for name in self.state["costs"]}

    def submit(self, probe, *, task_index=0, seed=313, channels=(), delay_seconds=0., source=None):
        """Start a measurement or suite apparatus in its own local process."""
        if delay_seconds < 0:
            raise ValueError("experiment delay must be nonnegative")
        channels = tuple(dict.fromkeys(channels))
        rates = {channel.name: channel.cost_per_row for channel in CHANNELS}
        if set(channels) - set(rates):
            raise ValueError("unknown measurement channel")
        source = source or {"kind": "measurement", "task_index": task_index, "seed": seed}
        if source["kind"] == "suite" and channels:
            raise ValueError("suite tasks have no selectable instruments")
        reservation = {"observation_cost": probe.cost,
                       "measurement_cost": probe.cost * sum(rates[name] for name in channels),
                       "experiment_cost": 1}
        reserved = self.reserved
        if any(self.costs[name] + reserved[name] + cost > self.state["limits"][name]
               for name, cost in reservation.items()):
            raise ValueError("experiment exceeds remaining resources after pending reservations")
        ticket_id = f"experiment_{self.state['next_ticket']:05d}"
        self.state["next_ticket"] += 1
        request_path = self.directory / f"{ticket_id}.request.json"
        result_path = self.directory / f"{ticket_id}.result.json"
        request_path.write_text(json.dumps({"source": source, "probe": asdict(probe), "channels": channels,
                                           "delay_seconds": delay_seconds}), encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH", ""))))
        error_path = self.directory / f"{ticket_id}.stderr"
        with error_path.open("w", encoding="utf-8") as errors:
            process = subprocess.Popen([sys.executable, "-m", "hegel_machine.cognition.delayed_feedback",
                                        str(request_path), str(result_path)],
                                       stdout=subprocess.DEVNULL, stderr=errors,
                                       start_new_session=True, env=environment)
        self._processes[ticket_id] = process
        self.state["tickets"][ticket_id] = {
            "ticket_id": ticket_id, "process_id": process.pid, "source": source, "probe_id": probe.probe_id,
            "reservation": reservation, "result_path": str(result_path), "error_path": str(error_path), "status": "running",
            "settled": False, "delivered": False}
        return ticket_id

    def poll(self, ticket_id=None):
        if ticket_id is None:
            return {identifier: self.poll(identifier) for identifier in self.state["tickets"]}
        ticket = self.state["tickets"][ticket_id]
        path = Path(ticket["result_path"])
        if not ticket["settled"] and not path.exists():
            if ticket_id in self._processes:
                returncode = self._processes[ticket_id].poll()
            else:
                try:
                    pid, status = os.waitpid(ticket["process_id"], os.WNOHANG)
                    returncode = os.waitstatus_to_exitcode(status) if pid else None
                except ChildProcessError:
                    try:
                        os.kill(ticket["process_id"], 0)
                        returncode = None
                    except ProcessLookupError:
                        returncode = -1
            if returncode is not None and not path.exists():
                _write_result(path, {"status": "failed", "observations": [],
                    "error": Path(ticket["error_path"]).read_text() or f"apparatus exited with code {returncode}",
                    "costs": {"observation_cost": 0, "measurement_cost": 0, "experiment_cost": 1}})
        if not ticket["settled"] and path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            ticket["status"] = result["status"]
            ticket["costs"] = result["costs"]
            for name, cost in result["costs"].items():
                self.state["costs"][name] += cost
            ticket["settled"] = True
            if ticket_id in self._processes:
                self._processes[ticket_id].poll()
        return deepcopy(ticket)

    def collect(self, ticket_id):
        ticket = self.poll(ticket_id)
        if not ticket["settled"] or ticket["delivered"]:
            return None
        result = json.loads(Path(ticket["result_path"]).read_text(encoding="utf-8"))
        self.state["tickets"][ticket_id]["delivered"] = True
        return {"ticket_id": ticket_id, **result}

    def cancel(self, ticket_id):
        ticket = self.poll(ticket_id)
        if ticket["settled"]:
            return ticket
        # Cancellation is cooperative: the apparatus checks this marker before
        # acquisition. If acquisition already completed, its result wins.
        Path(ticket["result_path"]).with_suffix(".cancel").touch()
        self.state["tickets"][ticket_id]["status"] = "cancelling"
        return deepcopy(self.state["tickets"][ticket_id])


def _worker(request_path, result_path):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    cancelled = result_path.with_suffix(".cancel")
    remaining = request["delay_seconds"]
    while remaining > 0 and not cancelled.exists():
        duration = min(.02, remaining)
        sleep(duration)
        remaining -= duration
    costs = {"observation_cost": 0, "measurement_cost": 0, "experiment_cost": 1}
    result = {"status": "cancelled", "observations": [], "costs": costs}
    if not cancelled.exists():
        try:
            source = request["source"]
            raw = request["probe"]
            probe = Probe(raw["probe_id"], tuple(raw["sequence"]))
            if source["kind"] == "measurement":
                with make_lab(source["task_index"], source["seed"]) as laboratory:
                    batch = laboratory.acquire(probe, request["channels"])
            elif source["kind"] == "suite":
                from .environments import make_suite
                environment = make_suite(source["seed"], partition=source.get("partition", "development"))[source["task_index"]]
                rows = environment.observe(probe)
                batch = MeasurementBatch(rows, len(rows), 0, ())
            else:
                raise ValueError(f"unknown apparatus source: {source['kind']}")
            costs.update(observation_cost=batch.observation_cost, measurement_cost=batch.measurement_cost)
            result.update(status="succeeded", observations=[asdict(row) for row in batch.observations],
                          channels=list(batch.channels))
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
    _write_result(result_path, result)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
