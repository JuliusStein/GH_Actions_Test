from setuptools import find_packages, setup

DISTNAME = "cogweb"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski"
AUTHOR_EMAIL = "ryan.soklaski@gmail.com"
URL = "https://github.com/rsokl/CogWeb"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = ["typing-extensions", "cogbooks", "jupytext >= 1.2.0"]
DESCRIPTION = "Tooling for upgrading PLYMI source material for jupytext"


setup(
    name=DISTNAME,
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        exclude=[
            "tests",
            "tests.*",
            "Python",
            "Python.*",
            "docs",
            "docs.*",
            "docs_backup",
            "docs_backup.*",
        ],
    ),
    version="1.0",
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    url=URL,
    python_requires=">=3.7",
)
