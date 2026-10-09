"""Locate historical work and execute its current native research entry point."""

import argparse
from collections import Counter
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys


def historical_capabilities(*, version=None, group=None, query=None, compact=False):
    """Find retained implementations and the interfaces that now consume them.

    A query is a case-insensitive conjunction of whitespace-separated terms.
    Route terms also find their historical versions, so callers do not need to
    know which release first implemented the capability they want to reuse.
    """
    catalogue = json.loads(files(__package__).joinpath("historical_capabilities.json").read_text())
    versions = catalogue["versions"]
    if version:
        version = version.removeprefix("v")
        if version.endswith(".0"):
            version = version[:-2]
        versions = [entry for entry in versions if entry["version"] == version]
        if not versions:
            raise ValueError(f"historical version not found: {version}")
    if group:
        versions = [entry for entry in versions if group in entry["capability_groups"]]
        if not versions:
            raise ValueError(f"capability group not found: {group}")
    routes = catalogue.get("reuse_routes", [])
    group_titles = {item["id"]: item["title"] for item in catalogue["capability_groups"]}
    terms = (query or "").casefold().split()
    selected = []
    for entry in versions:
        linked_routes = [route for route in routes
                         if entry["version"] in route["historical_versions"]]
        searchable = json.dumps({
            "entry": entry,
            "groups": [group_titles[group_id] for group_id in entry["capability_groups"]],
            "routes": [{"title": route["title"], "keywords": route["keywords"]}
                       for route in linked_routes],
        }, ensure_ascii=False).casefold()
        if all(term in searchable for term in terms):
            selected.append({**entry, "reuse_route_ids": [route["id"] for route in linked_routes]})
    versions = selected
    selected_versions = {entry["version"] for entry in versions}
    routes = [route for route in routes
              if selected_versions.intersection(route["historical_versions"])]
    dispositions = dict(Counter(entry["disposition"] for entry in versions))
    groups = catalogue["capability_groups"]
    if compact:
        group_counts = Counter(group_id for entry in versions for group_id in entry["capability_groups"])
        groups = [{"id": item["id"], "title": item["title"], "count": group_counts[item["id"]]}
                  for item in groups if group_counts[item["id"]]]
        if terms:
            direct_routes = [route for route in routes if all(
                term in json.dumps({"title": route["title"], "keywords": route["keywords"]},
                                   ensure_ascii=False).casefold() for term in terms)]
            if direct_routes:
                routes = direct_routes
        routes = [{**route, "historical_versions": [version for version in route["historical_versions"]
                                                    if version in selected_versions]} for route in routes]
        fields = ("version", "title", "disposition", "source_report", "native_commands",
                  "replacement_versions", "adoption_note", "integration_scope", "reuse_route_ids")
        versions = [{key: entry[key] for key in fields if key in entry} for entry in versions]
    return {"scope": catalogue["scope"], "disposition_meaning": catalogue["disposition_meaning"],
            "capability_groups": groups, "versions": versions,
            "reuse_routes": routes, "count": len(versions), "dispositions": dispositions}


def execute_historical(arguments, *, version=None, group=None):
    """Call the current CLI with the operator's concrete task and memory arguments."""
    if not version and not group:
        raise ValueError("choose a historical version or capability group before execution")
    selected = historical_capabilities(version=version, group=group)
    allowed = {command for entry in selected["versions"] for command in entry["native_commands"]}
    if not arguments or arguments[0] not in allowed:
        raise ValueError(f"choose the native command for this capability: {', '.join(sorted(allowed))}")
    if arguments[0] == "formal":
        from ..cli import main
        return main(arguments[1:])
    if arguments[0].startswith("script:"):
        root = Path(__file__).resolve().parents[3]
        script = root / "scripts" / (arguments[0].split(":", 1)[1] + ".py")
        return subprocess.run([sys.executable, str(script), *arguments[1:]], cwd=root).returncode
    from .cli import add_parser, dispatch
    parser = argparse.ArgumentParser(prog="hegel-machine")
    add_parser(parser.add_subparsers(dest="command", required=True))
    return dispatch(parser.parse_args(["cognition", *arguments]))


def dispatch_historical(args):
    if args.execute is not None:
        return execute_historical(args.execute, version=args.version, group=args.group)
    print(json.dumps(historical_capabilities(version=args.version, group=args.group,
                                            query=args.query, compact=args.compact),
                     ensure_ascii=False, indent=2))
    return 0
