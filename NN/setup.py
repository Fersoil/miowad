from distutils.core import setup

setup(
    name="networks",
    version="1.0",
    description="A simple neural network library",
    author="Tymoteusz Kwieciński",
    packages=[
        "networks",
    ],
    install_requires=["numpy", "matplotlib", "scipy"],
)
