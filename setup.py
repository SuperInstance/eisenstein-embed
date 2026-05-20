"""Setup script for eisenstein-embed."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eisenstein-embed",
    version="0.1.0",
    description="Enhanced static embeddings with a 5-layer matching cascade",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SuperInstance / Cocapn Fleet",
    author_email="casey@superinstance.com",
    url="https://github.com/SuperInstance/eisenstein-embed",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "model2vec": ["model2vec>=0.3.0"],
        "tensor-spline": ["tensor-spline>=0.1.0"],  # when published
        "plato": ["plato-training>=0.5.0"],  # when published
        "mesh": ["plato-core>=0.1.0"],
        "all": ["model2vec>=0.3.0", "faiss-cpu>=1.7.0", "plato-core>=0.1.0"],
    },
    entry_points={
        "superinstance.plugins": [
            "eisenstein-embed = eisenstein_embed.mesh:register_eisenstein",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
