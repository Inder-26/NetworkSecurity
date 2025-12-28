'''
The setup.py file is a build script for setuptools, used to package and distribute Python projects.
It typically contains metadata about the project, such as its name, version, author, and dependencies, as well as instructions on how to install the package.
'''
from setuptools import setup, find_packages
from typing import List

def get_requirements()-> List[str]:
    requirement_lst=[]
    """Reads the requirements.txt file and returns a list of dependencies."""
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirement.txt file not found")

    return requirement_lst

setup(
    name="Network Security",
    version="0.0.1",
    author="Inderjeet Singh",
    packages=find_packages(),
    install_requires=get_requirements()
)