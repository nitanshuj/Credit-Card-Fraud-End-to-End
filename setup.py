from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]: # Function will return a list of libraries
    """
        This function will return a list of requirements.
    """
    requirements = []
    with open("requirements.txt") as file_object:
        requirements = file_object.readlines()
        [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:     # If '-e .' in requirements --> remove it
            requirements.remove(HYPHEN_E_DOT)
    return requirements
    
setup(
    name="CreditCardFraudIdentifier",
    version='0.0.1',
    author='Nitanshu',
    author_email='nitanshuj138@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')#['pandas', 'numpy', 'seaborn', 'sklearn'],
)