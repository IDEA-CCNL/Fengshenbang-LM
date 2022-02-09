from setuptools import setup, find_packages

setup(
    name="fengshen",
    version="0.0.1",
    description="fengshen",
    long_description="fengshen",
    license="MIT Licence",
    url="http://test.com",
    author="",
    author_email="test@gmail.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[],

    scripts=[],
    entry_points={
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
