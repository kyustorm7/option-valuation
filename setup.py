from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Option Valuation Framework'
LONG_DESCRIPTION = 'Calculate and visualize option valuation process'

# Setting up
setup(
    name="optionval",
    version=VERSION,
    author="Kyusun Cho, Youngshin Lee, Youji Sung",
    author_email="kyustorm7@korea.ac.kr",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[ "scipy", "matplotlib", "numpy", "pandas"],  # add any additional packages that 
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'first package', 'finance', "option", "options", "valuation"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
