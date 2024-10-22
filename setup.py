from setuptools import setup, find_packages

setup(
    name='scanpy',
    version='0.0.0',
    description="ASIO scan control",
    author='TEM Gemini Centre',
    author_email='emil.christiansen@ntnu.no',
    license='MIT',
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pip",
        "sounddevice",
    ],
    python_requires=">=3.7",
    package_data={
        "": ["LICENSE", "README.md"],
        "scanpy": ["*.py"],
    }
)