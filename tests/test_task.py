import datetime
import unittest
from typing import Optional

import dateparser
import dateutil.tz as tz
from testfixtures import Replace, test_datetime

import task

NOW = datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=tz.tzlocal())
ONE_SECOND = datetime.timedelta(seconds=1)
TEN_MINUTES = datetime.timedelta(minutes=10)
FIVE_PM = NOW.replace(hour=17)


class TestTask(unittest.TestCase):
    def test_add(self):
        tasks: task.TaskDict = dict()

        name = "Do the dishes"
        tags = ["chores"]

        with Replace("task.datetime", test_datetime(NOW)):
            dateparser.conf.settings.RELATIVE_BASE = NOW

            tasks = task.add(
                tasks=tasks,
                name=name,
                due_input="5pm today",
                estimate_input="10 minutes",
                tags=tags,
            )

        expected = task.Task(name=name, due=FIVE_PM, estimate=TEN_MINUTES, tags=tags,)

        self.assertEqual(len(tasks), 1)
        self.assertTrue(name in tasks.keys())
        self.assertEqual(tasks[name], expected)

    def test_start(self):

        name = "Do the dishes"
        name2 = "Take out the garbage"
        tags = ["chores"]

        tasks = {
            name: task.Task(
                name=name,
                due=FIVE_PM,
                estimate=TEN_MINUTES,
                tags=tags,
                spans=[task.Span(start=NOW + ONE_SECOND)],
            ),
            name2: task.Task(name=name2, due=FIVE_PM, estimate=TEN_MINUTES, tags=tags,),
        }

        active_task = tasks[name]

        with Replace(
            "task.datetime", test_datetime(NOW + ONE_SECOND + ONE_SECOND, delta=0)
        ):
            dateparser.conf.settings.RELATIVE_BASE = NOW + ONE_SECOND + ONE_SECOND
            tasks, active_task = task.start(
                tasks=tasks, name=name2, active_task=active_task
            )

        expected = task.Task(
            name=name,
            due=FIVE_PM,
            estimate=TEN_MINUTES,
            tags=tags,
            spans=[
                task.Span(start=NOW + ONE_SECOND, stop=NOW + ONE_SECOND + ONE_SECOND)
            ],
        )
        expected2 = task.Task(
            name=name2,
            due=FIVE_PM,
            estimate=TEN_MINUTES,
            tags=tags,
            spans=[task.Span(start=NOW + ONE_SECOND + ONE_SECOND)],
        )

        self.assertEqual(len(tasks), 2)

        self.assertEqual(tasks[name], expected)
        self.assertEqual(tasks[name].spans[0].duration(), ONE_SECOND)

        self.assertEqual(active_task, tasks[name2])
        self.assertEqual(active_task, expected2)

    def test_stop(self):
        name = "Do the dishes"
        tags = ["chores"]

        tasks = {
            name: task.Task(
                name=name,
                due=FIVE_PM,
                estimate=TEN_MINUTES,
                tags=tags,
                spans=[task.Span(start=NOW)],
            )
        }

        with Replace("task.datetime", test_datetime(NOW + ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + ONE_SECOND
            task.stop(active_task=tasks[name])

        self.assertEqual(len(tasks), 1)

        expected = task.Task(
            name=name,
            due=FIVE_PM,
            estimate=TEN_MINUTES,
            tags=tags,
            spans=[task.Span(start=NOW, stop=NOW + ONE_SECOND)],
        )
        self.assertEqual(tasks[name], expected)
        self.assertEqual(tasks[name].spans[0].duration(), ONE_SECOND)

    def test_close(self):
        name = "Do the dishes"
        tags = ["chores"]

        tasks = {
            name: task.Task(
                name=name,
                due=FIVE_PM,
                estimate=TEN_MINUTES,
                tags=tags,
                spans=[task.Span(start=NOW)],
            )
        }

        with Replace("task.datetime", test_datetime(NOW + ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + ONE_SECOND
            task.close(tasks=tasks, active_task=tasks[name], name=name)

        expected = task.Task(
            name=name,
            due=FIVE_PM,
            estimate=TEN_MINUTES,
            tags=tags,
            spans=[task.Span(start=NOW, stop=NOW + ONE_SECOND)],
            open=False,
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[name], expected)

    def test_integration(self):
        tasks: task.TaskDict = dict()
        active_task: Optional[task.Task] = None

        tags = ["chores"]
        names = ["Wash dishes", "Take out garbage", "Write task tracker"]

        with Replace("task.datetime", test_datetime(NOW, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW
            tasks = task.add(
                tasks,
                name=names[0],
                due_input="5pm",
                estimate_input="ten minutes",
                tags=tags,
            )

        expected = [
            task.Task(name=names[0], due=FIVE_PM, estimate=TEN_MINUTES, tags=tags)
        ]
        self.assertEqual(list(tasks.values()), expected)

        noon_tomorrow = NOW.replace(hour=12) + datetime.timedelta(days=1)

        with Replace("task.datetime", test_datetime(NOW + ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + ONE_SECOND
            tasks = task.add(
                tasks,
                name=names[1],
                due_input="noon tomorrow",
                estimate_input="90 seconds",
                tags=tags,
            )

        expected.append(
            task.Task(
                name=names[1], due=noon_tomorrow, estimate=90 * ONE_SECOND, tags=tags
            )
        )
        self.assertEqual(list(tasks.values()), expected)

        with Replace("task.datetime", test_datetime(NOW + 2 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 2 * ONE_SECOND
            tasks, active_task = task.start(
                tasks=tasks, active_task=active_task, name=names[0],
            )

        expected[0].spans = [task.Span(start=NOW + 2 * ONE_SECOND)]
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[0]])

        with Replace("task.datetime", test_datetime(NOW + 3 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 3 * ONE_SECOND
            tasks, active_task = task.start(
                tasks=tasks, active_task=active_task, name=names[1],
            )

        expected[0].spans[0].stop = NOW + 3 * ONE_SECOND
        expected[1].spans = [task.Span(start=NOW + 3 * ONE_SECOND)]
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[1]])

        with Replace("task.datetime", test_datetime(NOW + 4 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 4 * ONE_SECOND
            tasks = task.add(tasks=tasks, name=names[2],)

        expected.append(task.Task(name=names[2], tags=list()))
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[1]])

        with Replace("task.datetime", test_datetime(NOW + 5 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 5 * ONE_SECOND
            tasks, active_task = task.close(
                tasks=tasks, active_task=active_task, name=names[0],
            )

        expected[0].open = False
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[1]])

        with Replace("task.datetime", test_datetime(NOW + 6 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 6 * ONE_SECOND
            tasks, active_task = task.close(
                tasks=tasks, active_task=active_task, name=names[1],
            )

        expected[1].open = False
        expected[1].spans[0].stop = NOW + 6 * ONE_SECOND
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, None)

        with Replace("task.datetime", test_datetime(NOW + 7 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 7 * ONE_SECOND
            tasks, active_task = task.start(
                tasks=tasks, active_task=active_task, name=names[2]
            )

        expected[2].spans = [task.Span(start=NOW + 7 * ONE_SECOND)]
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[2]])

        with Replace("task.datetime", test_datetime(NOW + 8 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 8 * ONE_SECOND
            active_task = task.stop(active_task=active_task)

        expected[2].spans[0].stop = NOW + 8 * ONE_SECOND
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, None)

        with Replace("task.datetime", test_datetime(NOW + 9 * ONE_SECOND, delta=0)):
            dateparser.conf.settings.RELATIVE_BASE = NOW + 9 * ONE_SECOND
            tasks, active_task = task.start(
                tasks=tasks, active_task=active_task, name=names[2]
            )

        expected[2].spans.append(task.Span(start=NOW + 9 * ONE_SECOND))
        self.assertEqual(list(tasks.values()), expected)
        self.assertEqual(active_task, tasks[names[2]])


if __name__ == "__main__":
    unittest.main()
