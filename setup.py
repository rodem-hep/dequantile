from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#
#     long_description = fh.read()

setup(
    name="dequantile",
    version='0.01',
    description="Decorrelating discriminants with normalizing flows.",
    # long_description=long_description,
    long_description_content_type='text/markdown',
    # url="https://github.com/sambklein/implicitBIBae",
    author="Sam Klein",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    dependency_links=[],
)
