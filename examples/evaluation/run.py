"""Entry point for the Gleanr evaluation harness."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from examples.evaluation.evaluator import Evaluator, EvaluatorConfig
from examples.evaluation.reporter import save_reports
from examples.evaluation.scenarios import SCENARIOS
from examples.test_agent.config import ChatConfig


console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gleanr Evaluation Harness - Automated testing for Gleanr effectiveness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation (80 sessions: 10 iterations × 8 turn counts)
  python -m examples.evaluation.run

  # Run specific turn counts only
  python -m examples.evaluation.run --turns 10,20,30

  # Customize iterations per turn count
  python -m examples.evaluation.run --iterations 5

  # Limit concurrency (default: 5)
  python -m examples.evaluation.run --max-concurrent 3

  # Run specific scenario only
  python -m examples.evaluation.run --scenario decision_tracking

  # Run all 6 scenarios sequentially
  python -m examples.evaluation.run --scenario all --iterations 10

  # Quick sanity check (1 iteration, 10 turns only)
  python -m examples.evaluation.run --turns 10 --iterations 1

  # Specify output directory
  python -m examples.evaluation.run --output ./reports/

  # Verbose mode (print progress to console)
  python -m examples.evaluation.run --verbose

Available scenarios:
  - decision_tracking: Test if decisions are recalled over time
  - constraint_awareness: Test if constraints are recalled when relevant
  - failure_memory: Test if failures are remembered to avoid repetition
  - multi_fact_tracking: Test tracking multiple distinct facts
  - goal_tracking: Test if goals and objectives are tracked
  - progressive_requirements: Test fact updates via consolidation
""",
    )

    parser.add_argument(
        "--turns",
        type=str,
        default="10,20,30,40,50,60,70,80",
        help="Comma-separated list of turn counts to evaluate (default: 10,20,30,40,50,60,70,80)",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations per turn count (default: 10)",
    )

    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=5,
        help="Maximum concurrent sessions (default: 5)",
    )

    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        choices=["all", *SCENARIOS.keys()],
        default="progressive_requirements",
        help="Scenario to run, or 'all' to run every scenario (default: progressive_requirements)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./evaluation_output",
        help="Output directory for reports (default: ./evaluation_output)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run a quick sanity test (1 iteration, 10 turns)",
    )

    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Preserve session database files after evaluation (default: clean up)",
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )

    # OpenAI-compatible chat endpoint (overrides Ollama for chat)
    parser.add_argument(
        "--chat-base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible chat API (e.g. http://host:8040/v1)",
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default=None,
        help="Model name for OpenAI-compatible chat API",
    )
    parser.add_argument(
        "--chat-api-key",
        type=str,
        default="",
        help="API key for OpenAI-compatible chat API",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # List scenarios and exit
    if args.list_scenarios:
        console.print("\n[bold]Available Scenarios:[/bold]\n")
        for name, scenario in SCENARIOS.items():
            console.print(f"  [green]{name}[/green]")
            console.print(f"    {scenario.description}")
            probes = scenario.get_probes()
            console.print(f"    Probes at turns: {[t for t, _ in probes]}")
            console.print()
        return 0

    # Parse turn counts
    try:
        turn_counts = [int(t.strip()) for t in args.turns.split(",")]
    except ValueError:
        console.print("[red]Error: Invalid turn counts. Use comma-separated integers.[/red]")
        return 1

    # Quick mode overrides
    if args.quick:
        turn_counts = [10]
        args.iterations = 1
        console.print("[yellow]Quick mode: Running 1 iteration of 10 turns[/yellow]\n")

    # Build chat config if OpenAI-compatible endpoint specified
    chat_config: ChatConfig | None = None
    if args.chat_base_url and args.chat_model:
        chat_config = ChatConfig(
            base_url=args.chat_base_url,
            model=args.chat_model,
            api_key=args.chat_api_key,
        )
        console.print(
            f"[bold]Chat backend:[/bold] {args.chat_base_url} "
            f"(model: {args.chat_model})\n"
        )

    # Determine which scenarios to run
    if args.scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    failed_scenarios: list[str] = []

    try:
        for scenario_name in scenario_names:
            try:
                config = EvaluatorConfig(
                    turn_counts=turn_counts,
                    iterations_per_turn_count=args.iterations,
                    max_concurrent=args.max_concurrent,
                    scenario_name=scenario_name,
                    verbose=args.verbose,
                    output_dir=args.output,
                    keep_data=args.keep_data,
                    chat_config=chat_config,
                )

                evaluator = Evaluator(config)

                if args.quick:
                    result = await evaluator.run_quick_test(turn_count=turn_counts[0])
                    console.print(f"\n[green]Quick test completed for {scenario_name}![/green]")
                    if result.recall_hit_rate < 0.8:
                        failed_scenarios.append(scenario_name)
                else:
                    report = await evaluator.run_evaluation()

                    json_path, md_path = save_reports(report, config.output_dir)

                    console.print(f"\n[bold]Reports saved for {scenario_name}:[/bold]")
                    console.print(f"  JSON: {json_path}")
                    console.print(f"  Markdown: {md_path}")

            except Exception as e:
                console.print(
                    f"\n[red]Scenario '{scenario_name}' failed: {e}[/red]"
                )
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                failed_scenarios.append(scenario_name)

        if failed_scenarios:
            console.print(
                f"\n[yellow]Failed scenarios: {', '.join(failed_scenarios)}[/yellow]"
            )
            return 1
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        return 130


def run() -> None:
    """Entry point wrapper."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
