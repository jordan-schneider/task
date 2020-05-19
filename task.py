""" Rich featured task tracker. """
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dateparser  # type: ignore
import defopt  # type: ignore
from dateutil import tz
from tabulate import tabulate


@dataclass
class Span:
    """ A span of time over which a task was worked on."""

    start: datetime = field(default_factory=lambda: datetime.now(tz=tz.tzlocal()))
    stop: Optional[datetime] = None

    def duration(self) -> timedelta:
        """ Computes the duration of time this span of work took."""
        if self.stop is None:
            return datetime.now(tz=tz.tzlocal()) - self.start
        else:
            return self.stop - self.start

    def closed(self):
        """ Returns true if the span is closed."""
        return self.stop is not None

    def close(self, stop_time: Optional[datetime] = None) -> None:
        """ Closes the span by setting the stop time to the current time"""
        self.stop = datetime.now(tz=tz.tzlocal()) if stop_time is None else stop_time


def round_to_seconds(duration: timedelta) -> timedelta:
    """ Rounds a timedelta to the nearest second."""
    return duration - timedelta(microseconds=duration.microseconds)


@dataclass
class Task:
    """ A task item on the todo list. """

    name: str
    due: Optional[datetime] = None
    estimate: Optional[timedelta] = None
    tags: Tuple[str, ...] = ()
    spans: List[Span] = field(default_factory=list)
    open: bool = True

    def total_duration(self) -> timedelta:
        """ Returns the total duration of all spans in the task."""
        return sum([span.duration() for span in self.spans], timedelta())

    def __repr__(self):
        out = self.name + "\n"
        if self.due:
            out += f"Due: {self.due}\n"

        if self.estimate:
            out += (
                f"{round_to_seconds(self.total_duration())} / "
                f"{round_to_seconds(self.estimate)}\n"
            )
        elif len(self.spans) > 0:
            out += str(round_to_seconds(self.total_duration())) + "\n"

        if len(self.tags) > 0:
            out += "Tags: " + ", ".join(self.tags)

        return out


def format_tasks(tasks: Sequence[Task]):
    """ Formats a sequence of tasks into a table."""
    return tabulate(
        [
            [
                task.name,
                task.due,
                task.estimate,
                task.total_duration(),
                ", ".join(task.tags),
            ]
            for task in tasks
        ],
        headers=["Name", "Due Date", "Estimate", "Duration", "Tags"],
    )


TaskDict = Dict[str, Task]

BASE_TIME = datetime(2020, 1, 1)

TASKS_FILENAME = "tasks"
ACTIVE_TASK_FILENAME = "active_task"
DEFAULT_TASKDIR = Path.home() / ".tasks"


def read_state(taskdir: Path) -> Tuple[TaskDict, Optional[Task]]:
    """ Reads the state located in the task directory provided, if it exists."""
    dirpath = Path(taskdir)

    if not dirpath.exists():
        print(f"Task directory not found. Creating directory at {taskdir}")
        dirpath.mkdir(parents=True, exist_ok=False)

    task_path = dirpath / TASKS_FILENAME
    tasks = pickle.load(open(task_path, "rb")) if task_path.exists() else dict()

    active_task_path = dirpath / ACTIVE_TASK_FILENAME
    name = open(active_task_path, "r").readline() if active_task_path.exists() else None
    active_task = tasks[name] if name is not None and name in tasks.keys() else None

    return tasks, active_task


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
    *,
    name: str,
    due: Optional[str] = None,
    estimate: Optional[str] = None,
    tags: Tuple[str, ...] = (),
    taskdir: Path = DEFAULT_TASKDIR,
) -> None:
    """ Adds a new task to the todo list."""
    tasks, active_task = read_state(taskdir)

    if name in tasks.keys():
        raise ValueError(f"Task with name {name} already exists.")

    task = Task(
        name=name,
        due=parse_local(due) if due is not None else None,
        estimate=parse_duration(estimate) if estimate is not None else None,
        tags=tags,
    )

    tasks[task.name] = task

    write(tasks, active_task, taskdir)


def start(
    *, name: str, start_time: Optional[str] = None, taskdir: Path = DEFAULT_TASKDIR,
) -> None:
    """ Starts tracking the time for the named task."""
    tasks, active_task = read_state(taskdir)

    if active_task is not None:
        active_task.spans[-1].close()
        print(f"Closed active task {active_task}")

    span = Span() if start_time is None else Span(start=parse_local(start_time))

    active_task = tasks[name]
    active_task.spans.append(span)

    write(tasks, active_task, taskdir)


def stop(*, stop_time: Optional[str] = None, taskdir: Path = DEFAULT_TASKDIR) -> None:
    """ Stops the timer on the active task."""
    tasks, active_task = read_state(taskdir)

    if active_task is None:
        raise ValueError("No active task to stop.")

    active_span = active_task.spans[-1]

    if stop_time is None:
        active_span.close()
    else:
        active_span.close(parse_local(stop_time))
    print(active_task)

    write(tasks, active_task, taskdir)


def close(*, name: str, taskdir: Path = DEFAULT_TASKDIR) -> None:
    """ Closes a task, removing it from the list of open tasks."""
    tasks, active_task = read_state(taskdir)

    if (
        active_task is not None
        and name == active_task.name
        and active_task.spans[-1].stop is None
    ):
        print(f"Stopping active timer on task {name}")
        stop()
        active_task = None

    print(f"Closing task {name}")
    task = tasks[name]
    task.open = False
    print(task)

    write(tasks, active_task, taskdir)


def status(*, show_closed: bool = False, taskdir: Path = DEFAULT_TASKDIR):
    """ Shows the currently active task, and all tasks in the list."""
    tasks, active_task = read_state(taskdir)
    print(f"Active task: {active_task}")
    print()
    if show_closed:
        print("Tasks:")
        print(format_tasks(list(tasks.values())))
    else:
        print("Open tasks:")
        print(format_tasks([task for task in tasks.values() if task.open]))


def write(tasks: TaskDict, active_task: Optional[Task], taskdir: Path) -> None:
    """ Write the todo task list and currently active task to the given task directory."""
    pickle.dump(tasks, open(taskdir / TASKS_FILENAME, "wb"))

    active_task_file = open(taskdir / ACTIVE_TASK_FILENAME, "w")
    active_task_file.writelines([active_task.name if active_task is not None else ""])
    active_task_file.close()


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="")
    defopt.run([add, close, start, stop, status])
