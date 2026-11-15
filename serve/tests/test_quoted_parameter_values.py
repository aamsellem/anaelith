"""String arguments that quote a parameter: the quoted closing tag is the value's, with or without the real one."""
import json
import unittest

from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser,
                            parse_tool_call)

EDIT = {"name": "edit_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
    "required": ["path", "old_string", "new_string"]}}
OLD, NEW = f"{PARAM_START}limit>40{PARAM_END}", f"{PARAM_START}limit>25{PARAM_END}"
HEAD = f"{CALL_START}\n{FUNC_START}edit_file>\n{PARAM_START}path>\nchat.jinja\n{PARAM_END}\n"
TAIL = f"{FUNC_END}\n{CALL_END}"
FRAGMENT = f"{PARAM_START}path>README.md{PARAM_END}\n{PARAM_START}limit>40{PARAM_END}"


def param(name, value, close=True):
    """A parameter in the usual form, or with its own closing tag left out after the value."""
    return f"{PARAM_START}{name}>\n{value}\n" + (f"{PARAM_END}\n" if close else "")


# label -> (the call's text, the arguments it must give)
CASES = {
    "both tags written": (HEAD + param("old_string", OLD) + param("new_string", NEW) + TAIL,
                          {"path": "chat.jinja", "old_string": OLD, "new_string": NEW}),
    "the last value's own tag left out": (HEAD + param("old_string", OLD) + param("new_string", NEW, close=False) + TAIL,
                                          {"path": "chat.jinja", "old_string": OLD, "new_string": NEW}),
    "both values' own tags left out": (HEAD + param("old_string", OLD, close=False)
                                       + param("new_string", NEW, close=False) + TAIL,
                                       {"path": "chat.jinja", "old_string": OLD, "new_string": NEW}),
    "the quoted function end after a quoted parameter": (
        HEAD + param("old_string", f"{OLD}\n{FUNC_END}") + param("new_string", f"{NEW}\n{FUNC_END}") + TAIL,
        {"path": "chat.jinja", "old_string": f"{OLD}\n{FUNC_END}", "new_string": f"{NEW}\n{FUNC_END}"}),
    "two quoted parameters": (HEAD + param("old_string", FRAGMENT) + param("new_string", NEW) + TAIL,
                              {"path": "chat.jinja", "old_string": FRAGMENT, "new_string": NEW}),
    "two quoted parameters, own tag left out": (HEAD + param("old_string", FRAGMENT, close=False)
                                                + param("new_string", NEW) + TAIL,
                                                {"path": "chat.jinja", "old_string": FRAGMENT, "new_string": NEW}),
    "an opening the value only mentions": (
        HEAD + param("old_string", f"use {PARAM_START}x> to pass x") + param("new_string", "y") + TAIL,
        {"path": "chat.jinja", "old_string": f"use {PARAM_START}x> to pass x", "new_string": "y"}),
    "a quoted parameter over several lines": (
        HEAD + param("old_string", f"{PARAM_START}limit>\n40\n{PARAM_END}\n") + param("new_string", "y") + TAIL,
        {"path": "chat.jinja", "old_string": f"{PARAM_START}limit>\n40\n{PARAM_END}\n", "new_string": "y"}),
    # what the model gets wrong stays as it was: nothing is quoted, the structure decides
    "a parameter the tool does not declare": (
        HEAD + param("old_string", "a") + param("new_string", "b") + param("limit", "3") + TAIL,
        {"path": "chat.jinja", "old_string": "a", "new_string": "b", "limit": 3}),
    "a required parameter left out": (HEAD + param("old_string", "a") + TAIL,
                                      {"path": "chat.jinja", "old_string": "a"}),
    "a required parameter left out after a quoted one": (HEAD + param("old_string", OLD, close=False) + TAIL,
                                                         {"path": "chat.jinja", "old_string": OLD}),
}


class QuotedParameterValues(unittest.TestCase):
    def test_whole_streamed_and_chunked(self):
        for label, (text, expected) in CASES.items():
            self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)], EDIT).arguments, expected, label)
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[EDIT], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([c.arguments for c in calls], [expected])
                        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             expected)

    def test_a_second_call_follows(self):
        first = HEAD + param("old_string", OLD) + param("new_string", NEW, close=False) + TAIL
        second = (f"{CALL_START}\n{FUNC_START}bash>\n{PARAM_START}command>\npytest -q\n{PARAM_END}\n"
                  f"{PARAM_START}timeout>\n120\n{PARAM_END}\n{TAIL}")
        bash = {"name": "bash", "parameters": {"properties": {"command": {"type": "string"},
                                                               "timeout": {"type": "integer"}}, "required": ["command"]}}
        for size in (1, 7, 4096):
            with self.subTest(size=size):
                parser = OutputParser(thinking=False, tools=[EDIT, bash], stream_tools=True)
                text, events = first + "\n" + second, []
                for i in range(0, len(text), size):
                    events.extend(parser.feed(text[i:i + size]))
                events.extend(parser.finish())
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual([c.name for c in calls], ["edit_file", "bash"])
                self.assertEqual(calls[0].arguments["new_string"], NEW)
                self.assertEqual(calls[1].arguments, {"command": "pytest -q", "timeout": 120})

    def test_without_a_schema(self):
        text = HEAD + param("old_string", OLD, close=False) + param("new_string", NEW, close=False) + TAIL
        self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)]).arguments,
                         {"path": "chat.jinja", "old_string": OLD, "new_string": NEW})


if __name__ == "__main__":
    unittest.main()
