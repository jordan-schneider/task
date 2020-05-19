""" Rich featured task tracker. """
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dateparser
import fire
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


def read_state(taskdir: Path) -> Tuple[TaskDict, Optional[Task]]:
    """ Reads the state located in the task directory provided, if it exists."""

    if not taskdir.exists():
        logging.info(f"Task directory not found. Creating directory at {taskdir}")
        taskdir.mkdir(parents=True, exist_ok=False)

    task_path = taskdir / TASKS_FILENAME
    tasks = pickle.load(open(task_path, "rb")) if task_path.exists() else dict()

    active_task_path = taskdir / ACTIVE_TASK_FILENAME
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


def write(tasks: TaskDict, active_task: Optional[Task], path: Path) -> None:
    """ Write the todo task list and currently active task to the given task directory."""
    logging.info(f"Writing output to {path}")
    pickle.dump(tasks, open(path / TASKS_FILENAME, "wb"))

    active_task_file = open(path / ACTIVE_TASK_FILENAME, "w")
    active_task_file.writelines([active_task.name if active_task is not None else ""])
    active_task_file.close()


class Command:
    def __init__(self, taskdir: Path = Path.home() / ".tasks"):
        self.path = taskdir
        self.tasks, self.active_task = read_state(taskdir)

    def add(
        self,
        name: str,
        due: Optional[str] = None,
        estimate: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """ Adds a new task to the todo list."""
        if name in self.tasks.keys():
            raise ValueError(f"Task with name {name} already exists.")

        task = Task(
            name=name,
            due=parse_local(due) if due is not None else None,
            estimate=parse_duration(estimate) if estimate is not None else None,
            tags=tags,
        )

        self.tasks[task.name] = task
        write(self.tasks, self.active_task, self.path)

    def start(self, name: str, start_time: Optional[str] = None,) -> None:
        """ Starts tracking the time for the named task."""
        if name is None:
            raise ValueError("You must specify a name of the task to add.")

        if self.active_task is not None:
            self.active_task.spans[-1].close()
            logging.info(f"Closed active task {self.active_task}")

        span = Span() if start_time is None else Span(start=parse_local(start_time))

        self.active_task = self.tasks[name]
        self.active_task.spans.append(span)
        write(self.tasks, self.active_task, self.path)

    def stop(self, stop_time: Optional[str] = None) -> None:
        """ Stops the timer on the active task."""
        if self.active_task is None:
            raise ValueError("No active task to stop.")

        active_span = self.active_task.spans[-1]

        if stop_time is None:
            active_span.close()
        else:
            active_span.close(parse_local(stop_time))
        write(self.tasks, self.active_task, self.path)

    def close(self, name: str) -> None:
        """ Closes a task, removing it from the list of open tasks."""
        if (
            self.active_task is not None
            and name == self.active_task.name
            and self.active_task.spans[-1].stop is None
        ):
            logging.info(f"Stopping active timer on task {name}")
            self.stop()

        task = self.tasks[name]
        task.open = False
        write(self.tasks, self.active_task, self.path)


if __name__ == "__main__":
    fire.Fire(Command)
