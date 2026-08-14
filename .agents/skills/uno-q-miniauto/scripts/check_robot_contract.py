#!/usr/bin/env python3
"""Statically validate the UNO Q miniAuto firmware/Python Bridge contract."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


BASE_RPCS = {
    "drive",
    "stop",
    "read_sensors",
    "servo",
    "buzz",
    "led",
    "drive_raw",
    "health",
}


def default_repo_root() -> Path:
    # repo/.agents/skills/uno-q-miniauto/scripts/this_file.py
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root(),
        help="Repository root (defaults to the root containing this skill)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unwrapped firmware providers as errors",
    )
    return parser.parse_args()


def read_required(root: Path, relative: str, errors: list[str]) -> str:
    path = root / relative
    if not path.is_file():
        errors.append(f"missing required file: {relative}")
        return ""
    return path.read_text(encoding="utf-8")


def quoted_names(pattern: str, source: str) -> set[str]:
    return set(re.findall(pattern, source))


def top_level_string_dict_keys(tree: ast.Module, variable_name: str) -> set[str]:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in targets):
            continue
        value = statement.value
        if not isinstance(value, ast.Dict):
            return set()
        return {
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    sketch = read_required(root, "sketch/sketch.ino", errors)
    client = read_required(root, "python/robot_client.py", errors)
    main_py = read_required(root, "python/main.py", errors)
    sketch_yaml = read_required(root, "sketch/sketch.yaml", errors)
    read_required(root, "app.yaml", errors)

    python_sources: dict[str, str] = {}
    python_dir = root / "python"
    if python_dir.is_dir():
        for path in sorted(python_dir.glob("*.py")):
            python_sources[str(path.relative_to(root))] = path.read_text(encoding="utf-8")

    providers = quoted_names(r'Bridge\.provide_safe\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', sketch)
    calls = set().union(
        *(quoted_names(r'Bridge\.call\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', source) for source in python_sources.values())
    )

    missing_baseline = BASE_RPCS - providers
    if missing_baseline:
        errors.append("missing baseline Bridge provider(s): " + ", ".join(sorted(missing_baseline)))

    missing_wrappers = BASE_RPCS - calls
    if missing_wrappers:
        errors.append("baseline RPC(s) not called by MiniAutoRobot: " + ", ".join(sorted(missing_wrappers)))

    client_without_provider = calls - providers
    if client_without_provider:
        errors.append("Python Bridge call(s) without firmware provider: " + ", ".join(sorted(client_without_provider)))

    providers_without_client = providers - calls
    if providers_without_client:
        message = "firmware provider(s) without MiniAutoRobot wrapper: " + ", ".join(sorted(providers_without_client))
        (errors if args.strict else warnings).append(message)

    for relative, source in python_sources.items():
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            errors.append(f"{relative} syntax error at line {exc.lineno}: {exc.msg}")
            continue

        dynamic_call_lines = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr == "call"
                and isinstance(owner, ast.Name)
                and owner.id == "Bridge"
                and node.args
                and not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str))
            ):
                dynamic_call_lines.append(node.lineno)
        if dynamic_call_lines:
            warnings.append(
                f"{relative} has dynamic Bridge.call at line(s) "
                + ", ".join(str(line) for line in dynamic_call_lines)
                + "; verify provider names manually"
            )

        if relative == "python/capture.py":
            capture_methods = top_level_string_dict_keys(tree, "LABELS")
            missing_capture_providers = capture_methods - providers
            if missing_capture_providers:
                warnings.append(
                    "python/capture.py expects capture provider(s) not registered by firmware: "
                    + ", ".join(sorted(missing_capture_providers))
                )

    if "platform: arduino:zephyr" not in sketch_yaml:
        errors.append("sketch/sketch.yaml does not declare platform arduino:zephyr")
    if "Arduino_RouterBridge" not in sketch_yaml:
        errors.append("sketch/sketch.yaml does not declare Arduino_RouterBridge")
    if "ARDUINO_ARCH_ZEPHYR" not in sketch:
        errors.append("sketch no longer asserts the UNO Q Zephyr target")
    if "finally:" not in main_py or "robot.stop()" not in main_py:
        warnings.append("python/main.py may not guarantee robot.stop() during shutdown")

    print(f"Repository: {root}")
    print("Firmware providers: " + (", ".join(sorted(providers)) or "none"))
    print("Python Bridge calls: " + (", ".join(sorted(calls)) or "none"))

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors:
        print(f"FAIL: {len(errors)} error(s), {len(warnings)} warning(s)", file=sys.stderr)
        return 1

    print(f"PASS: Bridge contract and static configuration checks ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
