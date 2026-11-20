import ast
import string
from pathlib import Path

import pytest
from portrait_processor import i18n
from portrait_processor.i18n_es import ES

PACKAGE = Path(i18n.__file__).parent
# Calls whose string arguments people read on screen (or in error messages shown on screen).
VISIBLE = {
    "setText",
    "setToolTip",
    "setWindowTitle",
    "setPlaceholderText",
    "setSuffix",
    "setPrefix",
    "addItem",
    "addText",
    "setLabelText",
    "setStatusTip",
    "QLabel",
    "QPushButton",
    "QCheckBox",
    "QRadioButton",
    "QGroupBox",
    "QAction",
    "button",
    "hint",
    "card",
    "progress",
    "_status",
    "emit_progress",
    "question",
    "information",
    "warning",
    "critical",
    "getOpenFileNames",
    "getOpenFileName",
    "getExistingDirectory",
    "getSaveFileName",
    "RuntimeError",
    "ValueError",
    "TimeoutError",
    "Cancelled",
}


def is_logging(func):
    """LOG.warning(...) / logging.getLogger(...).warning(...) write logs, not screen text."""
    target = func.value if isinstance(func, ast.Attribute) else None
    while isinstance(target, (ast.Call, ast.Attribute)):
        if isinstance(target, ast.Call):
            target = target.func
        elif target.attr == "getLogger":
            return True
        else:
            target = target.value
    return isinstance(target, ast.Name) and target.id in {"LOG", "LOGGER", "logger", "logging"}


def name_of(func):
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def has_words(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return any(c.isalpha() for c in node.value)
    if isinstance(node, ast.JoinedStr):
        return any(has_words(v) for v in node.values)
    if isinstance(node, ast.BinOp):
        return has_words(node.left) or has_words(node.right)
    if isinstance(node, ast.IfExp):
        return has_words(node.body) or has_words(node.orelse)
    return False


def modules():
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name not in {"i18n.py", "i18n_es.py"}:
            yield path, ast.parse(path.read_text(encoding="utf-8"))


def translated_texts():
    for path, tree in modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and name_of(node.func) == "tr" and node.args:
                first = node.args[0]
                assert isinstance(first, ast.Constant), (
                    f"{path.name}:{node.lineno} tr() needs a literal"
                )
                yield path.name, node.lineno, first.value


def test_every_visible_text_goes_through_tr():
    missing = []
    for path, tree in modules():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and name_of(node.func) in VISIBLE
                and not is_logging(node.func)
            ):
                for arg in node.args:
                    if has_words(arg):
                        missing.append(f"{path.name}:{node.lineno} {ast.unparse(arg)[:70]}")
    assert not missing, "Wrap in tr():\n" + "\n".join(missing)


def fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_translated_text_has_spanish_with_the_same_placeholders():
    problems = []
    for module, line, text in translated_texts():
        if text not in ES:
            problems.append(f"{module}:{line} missing: {text!r}")
        elif fields(ES[text]) != fields(text):
            problems.append(f"{module}:{line} placeholders differ: {text!r}")
        elif not ES[text].strip():
            problems.append(f"{module}:{line} empty: {text!r}")
    assert not problems, "\n".join(problems)


@pytest.fixture
def spanish():
    previous = i18n._language
    i18n.set_language("es")
    yield
    i18n.set_language(previous or "en")


def test_tr_translates_and_formats(spanish):
    text, translated = next((k, v) for k, v in ES.items() if "{" not in k)
    assert i18n.tr(text) == translated
    assert i18n.tr("Not in the catalog {n}", n=3) == "Not in the catalog 3"


def test_language_choice_and_child_processes(monkeypatch):
    monkeypatch.setattr(i18n, "system_language", lambda: "es")
    assert i18n.resolve("auto") == "es"
    assert i18n.resolve(None) == "es"
    assert i18n.resolve("en") == "en"
    previous = i18n._language
    try:
        i18n.set_language("es")
        import os

        assert os.environ[i18n.ENVIRONMENT_KEY] == "es"
        i18n._language = None
        assert i18n.language() == "es"
    finally:
        i18n.set_language(previous or "en")
