from setuptools import setup, find_packages

setup(
    name='monocle',
    version='0.1',
    packages=find_packages(exclude=['coverage', 'build']),
    install_requires=[
        'argparse',
        'rich',
        'setuptools',
        'huggingface_hub',
        'numpy',
        'pyyaml',
        'torchvision',
        'torchaudio',
        'bitsandbytes',
        'accelerate',
        'transformers',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'monocle = Monocle.monocle:run',
            'monocle-ollama = Monocle.monocle_ollama:run'
        ]
    }
)
