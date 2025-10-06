from setuptools import find_packages, setup
#findpacakages - discover all the packages utilizied in our ML project 
from typing import List


## In real project we may use 100 packages
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)-> List[str]:
    """
    This function will return the list of requirements
    """

    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Mridul Katoch',
    author_email='mr.katoch03@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)