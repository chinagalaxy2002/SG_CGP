"""Stop one native-binding run after N validations and launch a fresh coefficient run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC_TAG = "val/MR-mAP-Full_Avg"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def report(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def write_state(path: Path, **values: Any) -> None:
    state: Dict[str, Any] = {"updated_at": timestamp(), **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validation_status(tensorboard_dir: Path) -> Tuple[int, Optional[int], Optional[float]]:
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if METRIC_TAG not in accumulator.Tags().get("scalars", []):
        return 0, None, None
    events = accumulator.Scalars(METRIC_TAG)
    if not events:
        return 0, None, None
    return len(events), int(events[-1].step), float(events[-1].value)


def process_command(pid: int) -> Optional[str]:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except FileNotFoundError:
        return None


def verified_process_alive(pid: int, command_token: str) -> bool:
    command = process_command(pid)
    if command is None:
        return False
    if command_token not in command:
        raise RuntimeError(
            f"PID {pid} exists but no longer matches required token {command_token!r}: {command}"
        )
    return True


def wait_for_exit(pid: int, command_token: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not verified_process_alive(pid, command_token):
            return True
        time.sleep(1.0)
    return not verified_process_alive(pid, command_token)


def stop_old_process(pid: int, command_token: str) -> None:
    if not verified_process_alive(pid, command_token):
        raise RuntimeError(f"PID {pid} exited before the handoff boundary")

    report(f"Sending SIGINT to verified coefficient=0.2 process PID {pid}")
    os.kill(pid, signal.SIGINT)
    if wait_for_exit(pid, command_token, timeout=120.0):
        return

    report(f"PID {pid} did not exit after SIGINT; sending SIGTERM")
    os.kill(pid, signal.SIGTERM)
    if wait_for_exit(pid, command_token, timeout=60.0):
        return

    report(f"PID {pid} did not exit after SIGTERM; sending SIGKILL to that exact PID")
    os.kill(pid, signal.SIGKILL)
    if not wait_for_exit(pid, command_token, timeout=30.0):
        raise RuntimeError(f"PID {pid} is still alive after SIGKILL")


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def launch_new_run(args: argparse.Namespace) -> str:
    if tmux_session_exists(args.new_session):
        raise RuntimeError(f"Refusing to replace existing tmux session {args.new_session!r}")

    args.new_console_log.parent.mkdir(parents=True, exist_ok=True)
    launcher = args.repo_root / "code" / "sg_native_binding_validation_lab" / "run_qvhighlights.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"Missing launcher: {launcher}")
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Missing baseline checkpoint: {args.baseline_checkpoint}")

    environment = {
        "TARGET_GPU": str(args.gpu),
        "NATIVE_BIND_COEF": str(args.new_coefficient),
        "BATCH_SIZE": str(args.batch_size),
        "BASELINE_CHECKPOINT": str(args.baseline_checkpoint),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
    command = (
        f"cd {shlex.quote(str(args.repo_root))} && "
        f"env {assignments} bash {shlex.quote(str(launcher))} "
        f">> {shlex.quote(str(args.new_console_log))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.new_session, "bash", "-lc", command],
        check=True,
    )
    report(f"Started tmux session {args.new_session!r}: {command}")
    return command


def wait_for_new_training(args: argparse.Namespace, timeout: float = 180.0) -> int:
    required = f"native_binding.coefficient={args.new_coefficient}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["pgrep", "-af", "code.sg_native_binding_validation_lab.train_native_binding"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if required in line:
                return int(line.split(maxsplit=1)[0])
        if not tmux_session_exists(args.new_session):
            raise RuntimeError(
                f"tmux session {args.new_session!r} exited before coefficient={args.new_coefficient} started"
            )
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for coefficient={args.new_coefficient} training process")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tensorboard-dir", type=Path, required=True)
    parser.add_argument("--initial-validation-count", type=int, required=True)
    parser.add_argument("--additional-validations", type=int, default=10)
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--old-command-token", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--new-session", required=True)
    parser.add_argument("--new-console-log", type=Path, required=True)
    parser.add_argument("--new-coefficient", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--check-once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_count = args.initial_validation_count + args.additional_validations
    common = {
        "initial_validation_count": args.initial_validation_count,
        "additional_validations": args.additional_validations,
        "target_validation_count": target_count,
        "old_pid": args.old_pid,
        "new_coefficient": args.new_coefficient,
        "new_session": args.new_session,
        "new_console_log": str(args.new_console_log),
        "source_tensorboard_dir": str(args.source_tensorboard_dir),
        "baseline_checkpoint": str(args.baseline_checkpoint),
    }

    try:
        if not args.source_tensorboard_dir.is_dir():
            raise FileNotFoundError(f"Missing TensorBoard directory: {args.source_tensorboard_dir}")
        if args.additional_validations <= 0:
            raise ValueError("additional-validations must be positive")
        if args.new_coefficient <= 0:
            raise ValueError("new-coefficient must be positive")
        if not verified_process_alive(args.old_pid, args.old_command_token):
            raise RuntimeError(f"PID {args.old_pid} is not running at controller startup")

        count, last_step, last_value = validation_status(args.source_tensorboard_dir)
        if count < args.initial_validation_count:
            raise RuntimeError(
                f"Persisted validation count regressed: observed {count}, expected at least "
                f"{args.initial_validation_count}"
            )
        write_state(
            args.state_path,
            status="waiting",
            observed_validation_count=count,
            last_validation_step=last_step,
            last_validation_score=last_value,
            **common,
        )
        report(
            f"Watching validation count {count} -> {target_count}; old PID={args.old_pid}, "
            f"next coefficient={args.new_coefficient}"
        )
        if args.check_once:
            report("Check-only mode complete; no signal or launch action taken")
            return 0

        previous_count = count
        while count < target_count:
            time.sleep(args.poll_seconds)
            if not verified_process_alive(args.old_pid, args.old_command_token):
                raise RuntimeError(
                    f"PID {args.old_pid} exited with only {count}/{target_count} validations persisted"
                )
            count, last_step, last_value = validation_status(args.source_tensorboard_dir)
            if count != previous_count:
                report(
                    f"Validation progress: {count}/{target_count}, step={last_step}, "
                    f"MR-mAP={last_value:.6f}"
                )
                previous_count = count
                write_state(
                    args.state_path,
                    status="waiting",
                    observed_validation_count=count,
                    last_validation_step=last_step,
                    last_validation_score=last_value,
                    **common,
                )

        write_state(
            args.state_path,
            status="stopping_old_run",
            observed_validation_count=count,
            last_validation_step=last_step,
            last_validation_score=last_value,
            **common,
        )
        report(f"Reached handoff boundary {count}/{target_count}")
        stop_old_process(args.old_pid, args.old_command_token)
        report(f"Verified old PID {args.old_pid} has exited")

        write_state(
            args.state_path,
            status="launching_new_run",
            observed_validation_count=count,
            last_validation_step=last_step,
            last_validation_score=last_value,
            **common,
        )
        time.sleep(5.0)
        launch_command = launch_new_run(args)
        new_pid = wait_for_new_training(args)
        write_state(
            args.state_path,
            status="launched",
            observed_validation_count=count,
            last_validation_step=last_step,
            last_validation_score=last_value,
            new_pid=new_pid,
            launch_command=launch_command,
            **common,
        )
        report(f"Verified coefficient={args.new_coefficient} training PID {new_pid}")
        return 0
    except Exception as error:  # pylint: disable=broad-except
        report(f"HANDOFF FAILED: {type(error).__name__}: {error}")
        write_state(args.state_path, status="failed", error=f"{type(error).__name__}: {error}", **common)
        return 1


if __name__ == "__main__":
    sys.exit(main())
