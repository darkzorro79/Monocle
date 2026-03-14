from setuptools import setup, find_packages

setup(
    name='monocle',
    version='0.2',
    packages=find_packages(exclude=['coverage', 'build']),
    python_requires='>=3.10',
    install_requires=[
        'argparse',
        'rich',
        'setuptools',
        'huggingface_hub',
        'numpy',
        'pyyaml',
        'bitsandbytes>=0.43.0',
        'accelerate>=0.26.0',
        'transformers>=4.38.0',
        'torch>=2.7.0',
    ],
    extras_require={
        'ollama': ['ollama>=0.1.0', 'requests'],
    },
    entry_points={
        'console_scripts': [
            'monocle = Monocle.monocle:run',
            'monocle-ollama = Monocle.monocle_ollama:run'
        ]
    }
)
