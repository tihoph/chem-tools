"""Test the auto_parallel function."""

from __future__ import annotations

import multiprocessing as mp
import string
import time
from collections.abc import Generator
from typing import TypedDict

import pytest

from chem_tools._parallel import auto_parallel

# capfd can only capture current process output
# therefore, we need to write to a file to check the output


@pytest.fixture
def inputs() -> list[tuple[int, str]]:
    return list(enumerate(string.ascii_lowercase))


@pytest.fixture
def main_outputs(inputs: list[tuple[int, str]]) -> list[tuple[str, int, str]]:
    return [(mp.current_process().name, ix, value) for ix, value in inputs]


def args_func(ix: int, value: str) -> tuple[str, int, str]:
    time.sleep(0.005)
    return mp.current_process().name, ix, value


def timed_func(ix: int, value: str) -> tuple[float, str, int, str]:
    time.sleep(0.005)
    return time.time(), mp.current_process().name, ix, value


def tuple_func(args: tuple[int, str]) -> tuple[str, int, str]:
    ix, value = args
    return args_func(ix, value)


class _KwArgs(TypedDict):
    ix: int
    value: str


def dict_func(args: _KwArgs) -> tuple[str, int, str]:
    ix, value = args["ix"], args["value"]
    return args_func(ix, value)


def test_minus_one() -> None:
    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        auto_parallel(args_func, range(100), n_jobs=-1)


def test_tuple_encapsulation(
    inputs: list[tuple[int, str]], main_outputs: list[tuple[str, int, str]]
) -> None:
    # inputs is tuple - should work with the normal function
    # or the tuple function with keep_arg=True
    result = auto_parallel(args_func, inputs, n_jobs=1)
    assert result == main_outputs
    result = auto_parallel(tuple_func, inputs, n_jobs=1, keep_arg=True)
    assert result == main_outputs

    # should not work with the tuple function without keep_arg=True
    # or the dict function in any case
    with pytest.raises(
        TypeError, match=r"tuple_func\(\) takes 1 positional argument but 2 were given"
    ):
        auto_parallel(tuple_func, inputs, n_jobs=1)

    with pytest.raises(
        TypeError, match=r"tuple indices must be integers or slices, not str"
    ):
        auto_parallel(dict_func, inputs, n_jobs=1, keep_arg=True)

    with pytest.raises(
        TypeError, match=r"dict_func\(\) takes 1 positional argument but 2 were given"
    ):
        auto_parallel(dict_func, inputs, n_jobs=1)


def test_dict_encapsulation(
    inputs: list[tuple[int, str]], main_outputs: list[tuple[str, int, str]]
) -> None:
    # inputs is dict - should work with the normal function
    # or the dict function with keep_arg=True
    dict_inputs = [{"ix": ix, "value": value} for ix, value in inputs]
    result = auto_parallel(args_func, dict_inputs, n_jobs=1)
    assert result == main_outputs
    result = auto_parallel(dict_func, dict_inputs, n_jobs=1, keep_arg=True)
    assert result == main_outputs

    # should not work with the dict function without keep_arg=True
    # or the tuple function in any case
    with pytest.raises(
        TypeError, match=r"dict_func\(\) got an unexpected keyword argument"
    ):
        auto_parallel(dict_func, dict_inputs, n_jobs=1)

    # it unpacks over the dict keys instead of the values - hidden bug
    result = auto_parallel(tuple_func, dict_inputs, n_jobs=1, keep_arg=True)
    assert result == [("MainProcess", "ix", "value") for _ in inputs]

    with pytest.raises(
        TypeError, match=r"tuple_func\(\) got an unexpected keyword argument"
    ):
        auto_parallel(tuple_func, dict_inputs, n_jobs=1)


def test_non_string_keys(inputs: list[tuple[int, str]]) -> None:
    dict_inputs = [{ix: value} for ix, value in inputs]
    with pytest.raises(TypeError, match=r"Expanding dictionary requires string keys."):
        auto_parallel(dict_func, dict_inputs, n_jobs=1)


def assert_generator(
    gen: Generator[tuple[float, str, int, str]],
    main_outputs: list[tuple[str, int, str]],
) -> list[tuple[float, str, int, str]]:
    """Checks creation times and content, not worker names."""
    assert isinstance(gen, Generator)
    item = next(gen)
    assert isinstance(item, tuple)
    assert item[2:] == main_outputs[0][1:]
    remaining = list(gen)
    assert [x[2:] for x in remaining] == [x[1:] for x in main_outputs[1:]]
    return [item, *remaining]


def test_return_generator(
    inputs: list[tuple[int, str]], main_outputs: list[tuple[str, int, str]]
) -> None:
    gen = auto_parallel(timed_func, inputs, n_jobs=1, as_generator=True)
    result = assert_generator(gen, main_outputs)
    assert all(x[1] == "MainProcess" for x in result)


def test_parallel(
    inputs: list[tuple[int, str]], main_outputs: list[tuple[str, int, str]]
) -> None:
    result = auto_parallel(args_func, inputs, n_jobs=0)
    assert result == main_outputs
    result = auto_parallel(args_func, inputs, n_jobs=1)
    assert result == main_outputs

    n_jobs = 4
    result = auto_parallel(args_func, inputs, n_jobs=n_jobs)

    used_procs = {x[0] for x in result}
    assert len(used_procs) == n_jobs
    assert all(x.startswith("LokyProcess") for x in used_procs)
    assert [x[1:] for x in result] == [x[1:] for x in main_outputs]

    # test as generator
    gen = auto_parallel(timed_func, inputs, n_jobs=n_jobs, as_generator=True)
    timed_result = assert_generator(gen, main_outputs)
    used_procs = {x[1] for x in timed_result}
    assert len(used_procs) == n_jobs
    assert all(x.startswith("LokyProcess") for x in used_procs)
