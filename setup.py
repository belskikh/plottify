import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="plottify",
    version="0.1",
    author="Aleksandr Belskikh",
    author_email="belskikh.aleksandr@gmail.com",
    description="Plotting tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/belskikh/plottify",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.6.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy>=1.16.4",
        "opencv-python",
    ]
)