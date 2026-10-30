"""T4 pilot entrypoint.

Default mode prints the plan and, if ``--assume-tok-s`` is passed, a
what-if go/no-go stamped as assumed. It does not load Qwen.

``--run --i-have-a-t4`` is the GPU path. It was not executed in the CPU
package work. Treat that path as unverified until it has been run on a T4.
"""

from __future__ import annotations

import argparse
import sys

from loop_lingua.pilot import PilotConfig, describe_plan, plan_go_nogo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or run the 96-generation T4 pilot.")
    parser.add_argument(
        "--assume-tok-s",
        type=float,
        default=None,
        help="What-if tokens/second. Printed as assumed, not as a measurement.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Load Qwen3.5-4B and generate. Refused unless --i-have-a-t4 is also set.",
    )
    parser.add_argument(
        "--i-have-a-t4",
        action="store_true",
        help="Required together with --run. Confirms this process should use the GPU.",
    )
    args = parser.parse_args(argv)
    config = PilotConfig()
    print(describe_plan(config))
    print()
    if args.run:
        if not args.i_have_a_t4:
            print(
                "Refusing to start GPU generation. Pass --i-have-a-t4 on the T4 machine. "
                "This CPU package does not run it.",
                file=sys.stderr,
            )
            return 2
        print(
            "GPU path is implemented in transformers_greedy_generate and was not "
            "executed while building this package. Stopping before model load "
            "so a laptop or CPU worker cannot start it by accident. On the T4, "
            "call loop_lingua.pilot.transformers_greedy_generate from the notebook "
            "after you have confirmed the Qwen3.5 class exposes generate().",
            file=sys.stderr,
        )
        return 2
    if args.assume_tok_s is None:
        decision = plan_go_nogo(None, config, basis="none")
    else:
        decision = plan_go_nogo(args.assume_tok_s, config, basis="assumed")
    print(f"decision: {decision.decision} (basis: {decision.basis})")
    for reason in decision.reasons:
        print(f"reason: {reason}")
    if decision.estimated_full_hours is not None:
        print(f"estimated full-study hours (upper bound): {decision.estimated_full_hours:.2f}")
    if decision.max_new_tokens_that_fit is not None:
        print(f"max new tokens that fit in budget: {decision.max_new_tokens_that_fit}")
    print("caveats:")
    for caveat in decision.caveats:
        print(f"- {caveat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
