import setuptools
import versioneer

requirements = [
    # package requirements go here
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='openff-spellbook',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Handy functionality for working with OpenFF data",
    license="MIT",
    author="Trevor Gokey",
    author_email='tgokey@uci.edu',
    url='https://github.com/mobleylab/openff-spellbook',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    
    install_requires=requirements,
    keywords='openff-spellbook',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
    # packages=['offsb',
    #           'offsb/op',
    #           'offsb/tools',
    #           'offsb/search',
    #           'offsb/qcarchive',
    #           'offsb/rdutil',
    #           'treedi'],
