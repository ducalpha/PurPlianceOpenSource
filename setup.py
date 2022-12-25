__author__ = 'Duc Bui'

import setuptools
from pathlib import Path

long_description = Path('README.rst').read_text()

setuptools.setup(
    name="oppnlp",
    version="0.0.2",
    author="Duc Bui",
    author_email="ducbui@umich.edu",
    description="Privacy policy analysis",
    license='Apache License 2.0',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ducalpha/privacy_policy_nlp",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
)
