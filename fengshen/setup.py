from setuptools import setup, find_packages

setup(
    name="fengshen",
    version="0.0.1",
    description="fengshen",
    long_description="fengshen",
    license="MIT Licence",
    url="https://idea.edu.cn",
    author="gaoxinyu",
    author_email="gaoxinyu@idea.edu.cn",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        'transformers >= 4.17.0',
        'datasets >= 2.0.0',
        'pytorch_lightning >= 1.5.10',
        'deepspeed == 0.5.10',
        'jieba-fast >= 0.53',
        'jieba >= 0.40.0',
        'protobuf == 3.20.1',
    ],

    scripts=[],
    entry_points={
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
