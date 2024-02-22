from setuptools import setup, find_packages

setup(
    name='sefaria_llm_interface',
    version='0.1',
    packages=find_packages(),
    author='Sefaria',
    author_email='hello@sefaria.org',
    description='Provides the interface for objects that the LLM repo receives as input and objects it outputs',
    url='https://github.com/Sefaria/LLM/tree/main/app/llm_interface',
    license='GPL 3.0',
    install_requires=[]
)