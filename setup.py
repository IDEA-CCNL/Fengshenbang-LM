from setuptools import setup, find_packages

setup(
    name="fengshen",
    version="0.0.1",
    description="fengshen",
    long_description="fengshen",
    license="Apache Licence 2.0",
    url="https://github.com/IDEA-CCNL/Fengshenbang-LM",
    author="IDEA CCNL",
    author_email="test@gmail.com",
    packages=['fengshen'],
    include_package_data=True,
    platforms="any",
    install_requires=['transformers'],
)
