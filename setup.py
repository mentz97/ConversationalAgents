from setuptools import setup, find_packages

setup(
    name='simmc',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='SIMMC data extractor for ML project',
    long_description=open('README.md').read(),
    install_requires=[],
    url='REPOSITORY_URL',
    author='AUTHOR_NAME',
    author_email='AUTHOR_EMAIL'
)
