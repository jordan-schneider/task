""" Rich featured task tracker. """
import argparse
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dateparser
from dateutil import tz


@dataclass
class Span:
    """ A span of time over which a task was worked on."""

    start: datetime = field(default_factory=lambda: datetime.now(tz=tz.tzlocal()))
    stop: Optional[datetime] = None

    def duration(self) -> timedelta:
        """ Computes the duration of time this span of work took."""
        # inlining this function so mypy can infer that self.stop isn't None
        if self.stop is None or self.start is None:
            raise RuntimeError(
                "Both start and stop times must be defined for duration to be defined."
            )
        return self.stop - self.start

    def close(self, stop_time: Optional[datetime] = None) -> None:
        """ Closes the span by setting the stop time to the current time"""
        self.stop = datetime.now(tz=tz.tzlocal()) if stop_time is None else stop_time


@dataclass
class Task:
    """ A task item on the todo list. """

    name: str
    due: Optional[datetime] = None
    estimate: Optional[timedelta] = None
    tags: Optional[List[str]] = None
    spans: List[Span] = field(default_factory=list)
    open: bool = True


TaskDict = Dict[str, Task]

ADD = "add"
START = "start"
STOP = "stop"
CLOSE = "close"
QUEUE = "queue"
COMMANDS = [ADD, START, STOP, QUEUE]

BASE_TIME = datetime(2020, 1, 1)

TASKS_FILENAME = "tasks"
ACTIVE_TASK_FILENAME = "active_task"


def get_args():
    """ Parses and returns command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        type=str,
        choices=COMMANDS,
        required=True,
        help="Available commands.",
    )

    # TODO(joschnei): Print usage statement for different commands instead of all at once.
    # Add --help.
    add_group = parser.add_argument_group("Add task")
    add_group.add_argument(
        "name", type=str, required=True, help="Name of task to modify."
    )
    add_group.add_argument(
        "--due", type=str, default=None, help="Due date of the task."
    )
    add_group.add_argument_group(
        "-e",
        "--estimate",
        type=str,
        default=None,
        help="How long you expect the task to take.",
    )
    add_group.add_argument("tags", nargs="*", help="Additional tags to attach to task.")

    start_group = parser.add_argument_group("Start task timer")
    start_group.add_argument(
        "name", type=str, required=True, help="Name of task to start."
    )
    start_group.add_argument(
        "-t",
        "--start-time",
        type=str,
        default=None,
        help="Time to start at, instead of current time.",
    )

    stop_group.add_argument_group("Stop task")
    stop_group.add_argument(
        "-t",
        "--stop-time",
        type=str,
        default=None,
        help="Time to stop at, instead of current time.",
    )

    close_group = parser.add_argument_group("Close task")
    close_group.add_argument(
        "name", type=str, required=True, help="Name of task to close."
    )

    # TODO(joschnei): Check if $HOME is unix specific.
    parser.add_argument("--taskdir", type=str, default="$HOME/.task/")

    return parser.parse_args()


def read_state(taskdir: str) -> Tuple[Path, TaskDict, Optional[Task]]:
    """ Reads the state located in the task directory provided, if it exists."""
    dirpath = Path(taskdir)

    if not dirpath.exists():
        logging.info(f"Task directory not found. Creating directory at {taskdir}")
        dirpath.mkdir(parents=True, exist_ok=False)

    task_path = dirpath / TASKS_FILENAME
    tasks = pickle.load(open(task_path, "rb")) if task_path.exists() else dict()

    active_task_path = dirpath / ACTIVE_TASK_FILENAME
    name = open(active_task_path, "r").readline() if active_task_path.exists() else None
    active_task = tasks[name] if name is not None and name in tasks.keys() else None

    return dirpath, tasks, active_task


def parse_duration(duration: str) -> timedelta:
    """ Parses a nautral language duration (e.g. "three hours", "two days and seven mintues")."""

    # dateparser will parse durations as relative times. Dateparser will also allow us to specify
    # the time that we base our relative date on. The following command asks dateparser to parse
    # the duration only if it is a relative time, and then to subtract out the base time, leaving
    # only a timedelta. This is jank but I don't know a better way.
    return (
        dateparser.parse(
            duration,
            settings={
                "RELATIVE_BASE": BASE_TIME,
                "PARSERS": ["relative-time"],
                "PREFER_DATES_FROM": "future",
            },
        )
        - BASE_TIME
    )


def parse_local(raw: str) -> datetime:
    """ Parse raw datetime string and sets local timezone."""
    return dateparser.parse(raw).replace(tzinfo=tz.tzlocal())


def add(
    tasks: TaskDict,
    name: str,
    due_input: Optional[str] = None,
    estimate_input: Optional[str] = None,
    tags: List[str] = list(),
) -> TaskDict:
    """ Adds a new task to the todo list."""
    if name is None:
        # TODO(joschnei): Instead of printing specific error messages, print a usage statement
        # here.
        raise ValueError("You must specify a name of the task to add.")
    if name in tasks.keys():
        raise ValueError(f"Task with name {name} already exists.")

    due = parse_local(due_input) if due_input is not None else None
    estimate = parse_duration(estimate_input) if estimate_input is not None else None

    task = Task(name=name, due=due, estimate=estimate, tags=tags,)

    tasks[task.name] = task
    return tasks


def start(
    tasks: TaskDict,
    name: str,
    active_task: Optional[Task],
    start_time: Optional[str] = None,
) -> Tuple[TaskDict, Task]:
    """ Starts tracking the time for the named task."""
    if name is None:
        raise ValueError("You must specify a name of the task to add.")

    if active_task is not None:
        active_task.spans[-1].close()
        logging.info(f"Closed active task {active_task}")

    span = Span() if start_time is None else Span(start=parse_local(start_time))

    active_task = tasks[name]
    active_task.spans.append(span)

    return tasks, active_task


def stop(active_task: Optional[Task], stop_time: Optional[str] = None) -> None:
    """ Stops the timer on the active task."""
    if active_task is None:
        raise ValueError("No active task to stop.")

    active_span = active_task.spans[-1]

    if stop_time is None:
        active_span.close()
    else:
        active_span.close(parse_local(stop_time))

    # Intentionally returning None, as the active_task is None after stopping the active task.
    return None


def close(
    tasks: TaskDict, active_task: Optional[Task], name: str
) -> Tuple[TaskDict, Optional[Task]]:
    """ Closes a task, removing it from the list of open tasks."""
    if (
        active_task is not None
        and name == active_task.name
        and active_task.spans[-1].stop is None
    ):
        logging.info(f"Stopping active timer on task {name}")
        active_task = stop(active_task)

    task = tasks[name]
    task.open = False

    return tasks, active_task


def process_command(
    args, tasks: TaskDict, active_task: Optional[Task], taskdir: Path
) -> None:
    """ Identifies the requested command and executes it, saving the results to disk."""
    if args.command == ADD:
        tasks = add(
            tasks=tasks,
            name=args.name,
            due_input=args.due,
            estimate_input=args.estimate,
            tags=args.tags,
        )
    elif args.command == START:
        tasks, active_task = start(
            tasks=tasks,
            name=args.name,
            active_task=active_task,
            start_time=args.start_time,
        )
    elif args.command == STOP:
        active_task = stop(active_task=active_task, stop_time=args.stop_time)
    elif args.command == CLOSE:
        tasks, active_task = close(tasks=tasks, active_task=active_task, name=args.name)

    write(tasks, active_task, taskdir)


def write(tasks: TaskDict, active_task: Optional[Task], taskdir: Path) -> None:
    """ Write the todo task list and currently active task to the given task directory."""
    pickle.dump(tasks, open(taskdir / TASKS_FILENAME, "wb"))

    active_task_file = open(taskdir / ACTIVE_TASK_FILENAME, "w")
    active_task_file.writelines([active_task.name if active_task is not None else ""])
    active_task_file.close()


def main():
    args = get_args()

    taskdir, tasks, active_task = read_state(args.taskdir)

    process_command(args, tasks, active_task, taskdir)


if __name__ == "__main__":
    main()
