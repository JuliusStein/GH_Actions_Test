"""
Utilities for accessing the CogWorks source material and making conversions using jupytext
"""
import shutil
from collections import defaultdict
from collections.abc import Sequence as Sequence_
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, Iterator, Sequence, Tuple, Union
from warnings import warn

import jupytext
from cogbooks._functions import strip_text
from jupytext.cli import jupytext as jupytext_cli
from typing_extensions import Final

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


__all__ = [
    "get_all_markdown_files",
    "get_all_notebook_files",
    "convert_all_ipynb_to_markdown",
    "convert_all_markdown_to_ipynb",
    "convert_src_to_html",
    "build_to_doc",
    "jupytext_stripped_reads",
]


def _strip_solns_wrapper(func):
    """func(x, **kwargs) -> func(cogbooks.strip_text(x), **kwargs)"""

    @wraps(func)
    def wrapped(text, **kwargs):
        func.__doc__ = (
            "(Modified to apply ``cogbooks.strip_text()`` to all text before conversion) \n\n"
            + func.__doc__
        )
        return func(strip_text(text), **kwargs)

    return wrapped


jupytext_stripped_reads = _strip_solns_wrapper(jupytext.reads)


SRC_DIR_NAME: Final = "website_src"

TEMP_SOLN_FILE_SUFFIX: Final = ".has_solutions_tmp"

assert isinstance(TEMP_SOLN_FILE_SUFFIX, str) and TEMP_SOLN_FILE_SUFFIX.startswith(
    "."
), TEMP_SOLN_FILE_SUFFIX


PathLike = Union[str, Path]

# Paths to all directories containing source notebooks (relative to `website_src/`
all_source_dirs: Tuple[Path, ...] = (
    Path("Math_Materials"),
    Path("Audio"),
    Path("Video"),
    Path("Language"),
)


# The following are names of notebooks (.ipynb) that exist natively as CogWeb
# source material. It should be excluded from markdown conversion
excluded_notebook_names: FrozenSet[str] = frozenset()


def get_source_dirs_from_root(root: PathLike, dirs=all_source_dirs) -> Tuple[Path]:
    if not isinstance(root, Path):
        root = Path(root)

    root /= SRC_DIR_NAME
    assert root.is_dir(), str(root.absolute())

    dirs = tuple([root] + [root / d for d in dirs])  # type: Tuple[Path]

    bad = [d for d in dirs if not d.is_dir()]
    if bad:
        raise AssertionError(
            "The following directories do not exist: "
            + ("\n".join((str(x) for x in bad)))
        )
    return dirs


def get_all_files(
    root: PathLike, suffixes: Union[str, Sequence[str]]
) -> Dict[str, Tuple[Path]]:
    """Given the root dir containing `website_src`, get all of the files of
    the specified suffixes among the listed source directories"""
    if isinstance(suffixes, str):
        suffixes = (suffixes,)
    else:
        if not (
            isinstance(suffixes, Sequence_)
            and all(isinstance(i, str) for i in suffixes)
        ):
            raise TypeError(
                f"`suffixes` must be a string or a sequence of strings, got: {suffixes}"
            )
    suffixes = tuple(set(s[1:] if s.startswith(".") else s for s in suffixes))

    dir_tree = defaultdict(list)
    for d in get_source_dirs_from_root(root):
        for suffix in suffixes:
            dir_tree[str(d.absolute())].extend(sorted(d.glob(f"*.{suffix}")))
    return {k: tuple(v) for k, v in dir_tree.items()}


def get_all_markdown_files(root) -> Dict[str, Tuple[Path]]:
    return get_all_files(root, suffixes="md")


def get_all_notebook_files(root: PathLike) -> Dict[str, Tuple[Path]]:
    return get_all_files(root, suffixes="ipynb")


def iterate_over_all_files(
    *,
    root: PathLike,
    suffixes: Union[str, Sequence[str]],
    verbose: bool = False,
    excluded_file_names: FrozenSet[str] = frozenset(),
) -> Iterator[Path]:
    """Yields each file (Path) matching the provided suffixes found among
    the source directories."""
    assert verbose in {True, False}

    for dir_, files in tqdm(
        get_all_files(root, suffixes).items(),
        desc="directory loop",
        disable=not verbose,
    ):
        if verbose:
            print(f"\n\tProcessing directory: {dir_}")
        for file in tqdm(files, desc="file loop", disable=not verbose):  # type: Path
            if file.name in excluded_file_names:
                continue
            yield file


def test_ipynb_roundtrip_on_all(*, root: PathLike, verbose=True):
    print(f"Using jupytext version: {jupytext.__version__}")

    for dir_, files in tqdm(
        get_all_notebook_files(root).items(), desc="directory loop"
    ):  # type: Tuple[str, Iterable[Path]]
        if verbose:
            print(f"Processing directory: {dir_}")
        for file in tqdm(files, desc="file loop"):  # type: Path
            if file.name in excluded_notebook_names:
                continue
            jupytext_cli(["--to", "md", "--test", str(file)])


def convert_all_markdown_to_ipynb(
    root: PathLike,
    verbose: bool = True,
    excluded_file_names: FrozenSet[str] = frozenset(),
):
    assert all(name.endswith(".md") for name in excluded_file_names)
    for file in iterate_over_all_files(root=root, suffixes="md", verbose=verbose):
        jupytext_cli(["--to", "notebook", str(file)])


def convert_all_ipynb_to_markdown(
    root: PathLike,
    verbose: bool = True,
    excluded_file_names=frozenset(excluded_notebook_names),
):
    assert all(name.endswith(".md") for name in excluded_file_names)
    for file in iterate_over_all_files(root=root, suffixes="md", verbose=verbose):
        jupytext_cli(["--to", "markdown", str(file)])


def _delete_all(
    root: Path,
    *,
    file_getter: Callable[[Path], Dict[str, Iterable[Path]]],
    excluded_file_names: FrozenSet[str],
    test: bool,
):
    import os

    assert test in {True, False}

    if test:
        print("Nothing will be deleted unless you pass `test=False`")
    for dir_, files in file_getter(root).items():
        for file in files:
            if file.name in excluded_file_names:
                continue
            if test:
                print(repr(file) + " will be deleted")
            else:
                os.remove(str(file.absolute()))


def delete_all_notebooks(
    root,
    *,
    excluded_file_names: FrozenSet[str] = frozenset(excluded_notebook_names),
    test=True,
):
    assert all(name.endswith(".ipynb") for name in excluded_file_names)
    return _delete_all(
        root,
        file_getter=get_all_notebook_files,
        excluded_file_names=excluded_file_names,
        test=test,
    )


def delete_all_markdown(
    root, *, excluded_file_names: FrozenSet[str] = frozenset(), test=True
):
    assert all(name.endswith(".md") for name in excluded_file_names)
    return _delete_all(
        root,
        file_getter=get_all_markdown_files,
        excluded_file_names=excluded_file_names,
        test=test,
    )


def build_to_doc(root: PathLike):
    """
    Copy all files from docs/ to docs_backup/
    Copy all files from Python/_build/ to docs/

    Checks for .nojekyll and CNAME files

    Parameters
    ----------
    root : pathlib.Path
        The path to the top-level directory containing the Python/ dir."""
    if not isinstance(root, Path):
        root = Path(root)

    assert (root / "docs").is_dir()
    assert (root / SRC_DIR_NAME / "_build").is_dir()
    shutil.copyfile(root / "docs" / "CNAME", root / SRC_DIR_NAME / "_build" / "CNAME")

    assert (root / SRC_DIR_NAME / "_build" / ".nojekyll").is_file()

    if (root / "docs_backup").is_dir():
        shutil.rmtree(root / "docs_backup")
    shutil.move(root / "docs", root / "docs_backup")
    shutil.copytree(root / SRC_DIR_NAME / "_build", root / "docs")

    assert (root / "docs" / ".nojekyll").is_file()
    assert (root / "docs" / "CNAME").is_file()
    print("Done. Make sure to commit the changes to `docs/` and `docs_backup/`")


def run_sphinx(sphinx_project_root):
    """Runs sphinx in the specified directory"""
    import subprocess
    import os

    wd = os.getcwd()
    os.chdir(sphinx_project_root)

    try:
        subprocess.run(["python", "-m", "sphinx", ".", "_build", "-j4"])
    finally:
        os.chdir(wd)


def convert_src_to_html(sphinx_project_root: PathLike, verbose: bool = False):
    """Migrates appropriate files to sphinx_project_root/_build and removes
    cogbook-tagged text from all .md and .ipynb in the specified source
    directories then runs::

        python -m sphinx . _build -j4

    in the specified directory

    Parameters
    ----------
    sphinx_project_root : PathLike
        The directory containing the sphinx conf.py file.
        (E.g. CogWeb/website_src/, if you cloned the
        CogWeb repo).
    """

    import shutil

    sphinx_project_root = Path(sphinx_project_root)

    # copy _images/ to _build/_images
    build_dir = sphinx_project_root / "_build"
    images_dir = sphinx_project_root / "_images"
    if not images_dir.is_dir():
        warn(
            f"Images directory:\n\t{images_dir.absolute()}\n was not found. "
            f"Some of your images may be missing!"
        )
    else:
        build_dir.mkdir(exist_ok=True)
        (build_dir / "_images").mkdir(exist_ok=True)
        for f in images_dir.glob("*"):
            if f.name.endswith("gitkeep"):
                continue
            shutil.copyfile(f, (build_dir / "_images" / f.name))

    run_sphinx(sphinx_project_root)

