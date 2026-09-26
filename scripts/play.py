"""Watch a saved policy play one task, step by step, or play it yourself.

    python scripts/play.py summary_memory --benchmark dark_key_to_door \
        --condition raw_summary --seed 42 --split confirmation --task 0
    python scripts/play.py summary_memory --benchmark tmaze --condition full_gru \
        --seed 100 --auto --delay 0.05 --record tmaze.gif
    python scripts/play.py summary_memory --benchmark count_recall --no-policy

Every step prints the environment's observer frame (hidden layout and cue
included) and the policy's proposed action. Press Enter to let the policy act,
type an action (its number, or a movement key: a/w/d/s for left/up/right/down
on Key-to-Door, d/w/a/s for forward/up/back/down on the T-Maze, a count on
CountRecall) to override it, or ``q`` to stop. ``--auto`` runs the policy
without prompting; ``--record`` writes the ``rgb_array`` frames to a GIF.
``--no-policy`` skips the checkpoint and proposes random actions, which is the
quickest way to play an environment by hand.

Nothing is written under the run directory and no benchmark record is
produced; ``scripts/evaluate.py`` remains the evaluator.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from reasoned_icrl.experiments.contracts import ContractError
from reasoned_icrl.experiments.summary_memory import experiments as summary_memory
from reasoned_icrl.experiments.summary_memory.configs import (
    load_summary_memory_study,
)

STUDIES = {"summary_memory": summary_memory}
CLEAR = "\x1b[2J\x1b[H"
MOVEMENT_KEYS = {
    "left": "a",
    "up": "w",
    "right": "d",
    "down": "s",
    "stay": ".",
    "forward": "d",
    "back": "a",
}


def add_play_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--condition", default=None, help="required with a policy")
    parser.add_argument("--seed", type=int, default=None, help="required with a policy")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", default="development")
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="position in the split's ordered roster (default: the first task)",
    )
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument(
        "--checkpoint-rule",
        default="endpoint",
        choices=("selected", "final-epoch", "endpoint"),
        help="endpoint (default) plays the weights at the declared final epoch; "
        "selected plays --checkpoint (checkpoint.pt or policy_epoch_N)",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="load no checkpoint; proposals are uniform random actions",
    )
    parser.add_argument(
        "--sample", action="store_true", help="sample the policy instead of argmax"
    )
    parser.add_argument(
        "--auto", action="store_true", help="let the policy act without prompting"
    )
    parser.add_argument(
        "--delay", type=float, default=0.15, help="seconds between --auto frames"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="stop after this many decisions"
    )
    parser.add_argument(
        "--layout-period",
        type=int,
        default=None,
        help="Key-to-Door layout-change continuation: a new hidden layout after "
        "every P calls",
    )
    parser.add_argument(
        "--record", type=Path, default=None, help="write the frames to this GIF"
    )
    parser.add_argument(
        "--frame-ms", type=int, default=120, help="GIF frame duration in ms"
    )
    parser.add_argument(
        "--no-clear", action="store_true", help="do not clear the terminal per frame"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    studies = result.add_subparsers(dest="study_name", required=True)
    for name in STUDIES:
        add_play_arguments(studies.add_parser(name))
    return result


def key_bindings(names: tuple[str, ...]) -> dict[str, int]:
    """Typed tokens that select an action: its index, its name, its movement key."""
    bindings: dict[str, int] = {}
    for index, name in enumerate(names):
        bindings[str(index)] = index
        bindings[name] = index
        key = MOVEMENT_KEYS.get(name)
        if key is not None:
            bindings[key] = index
    return bindings


def describe(names: tuple[str, ...]) -> str:
    parts = []
    for index, name in enumerate(names):
        key = MOVEMENT_KEYS.get(name)
        parts.append(f"{index}={name}" + (f" [{key}]" if key else ""))
    return "  ".join(parts) if len(parts) <= 8 else f"0..{len(parts) - 1} (a count)"


def write_gif(frames: list[Any], path: Path, frame_ms: int) -> None:
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(int(frame_ms), 1),
        loop=0,
    )


def play(args: argparse.Namespace) -> int:
    from reasoned_icrl.experiments.benchmarks import saved_config
    from reasoned_icrl.runtime.play import close_player_experiment, load_player

    study_module = STUDIES[args.study_name]
    study = load_summary_memory_study(args.study)
    if args.no_policy:
        contract = study.contract(args.benchmark)
        config = None
    else:
        if args.condition is None or args.seed is None:
            raise ContractError("--condition and --seed name the run to play.")
        contract, resolved = study_module.resolve(
            study,
            benchmark=args.benchmark,
            condition=args.condition,
            seed=args.seed,
            device=args.device,
            output_root=args.output_root,
        )
        config = saved_config(resolved)
    player, roster, checkpoint = load_player(
        contract,
        config,
        split=args.split,
        checkpoint=args.checkpoint,
        checkpoint_rule=args.checkpoint_rule,
        sample_actions=args.sample,
        layout_period=args.layout_period,
    )
    if not 0 <= args.task < len(roster):
        raise ContractError(
            f"--task {args.task} is outside the {args.split!r} roster of {len(roster)}."
        )
    names = tuple(getattr(player.environment, "action_names", ()))
    if not names:
        names = tuple(f"action {i}" for i in range(player.environment.action_count))
    bindings = key_bindings(names)
    frames: list[Any] = []
    try:
        player.reset(roster[args.task])
        source = checkpoint or "random proposals"
        print(f"{contract.protocol}: {args.split} task {args.task} under {source}")
        while not player.done and (args.steps is None or player.steps < args.steps):
            frame = player.render("ansi")
            if args.record is not None:
                frames.append(player.render("rgb_array"))
            proposal = player.propose()
            if not args.no_clear:
                print(CLEAR, end="")
            print(frame)
            who = "policy" if player.has_policy else "random"
            print(f"{who} proposes {proposal} ({names[proposal]})")
            print(f"actions: {describe(names)}")
            action = proposal
            if args.auto:
                time.sleep(max(args.delay, 0.0))
            else:
                try:
                    line = input(
                        "[Enter] accept | action to override | q quit > "
                    ).strip()
                except EOFError:
                    line = "q"
                if line.lower() == "q":
                    break
                if line:
                    if line not in bindings:
                        print(f"unknown action {line!r}; accepted the proposal")
                    else:
                        action = bindings[line]
            reward, _, _ = player.step(action)
            print(f"executed {action} ({names[action]})  reward {reward:+.4f}")
        if args.record is not None:
            frames.append(player.render("rgb_array"))
        if not args.no_clear:
            print(CLEAR, end="")
        print(player.render("ansi"))
        outcome = "finished" if player.done else "stopped"
        print(
            f"{outcome} after {player.steps} decisions, "
            f"return {player.episode_return:+.4f}"
        )
        if args.record is not None and frames:
            write_gif(frames, args.record, args.frame_ms)
            print(f"wrote {len(frames)} frames to {args.record}")
    finally:
        player.close()
        if player.has_policy:
            close_player_experiment(player.experiment)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return play(args)
    except ContractError as error:
        print(f"play: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
