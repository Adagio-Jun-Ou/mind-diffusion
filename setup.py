from setuptools import setup, find_packages

setup(
    name="mind-diffusion",
    version="0.1.0",
    description="Diffusion-based simulation of neuron/mind latent dynamics",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)