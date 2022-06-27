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
    install_requires=[
        'transformers == 4.20.0',
        'datasets == 2.0.0',
        'pytorch_lightning == 1.6.0',
        'deepspeed == 0.5.10',
        'protobuf == 3.20.1',
    ],

    scripts=[],
    entry_points={
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
