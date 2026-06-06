"""
Setup configuration for Diabetes Prediction Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diabetes-prediction",
    version="1.0.0",
    author="Boujaada Mohammed",
    author_email="boujaadamohammed51@gmail.com",
    description="Machine Learning models for diabetes prediction using KNN, Random Forest, and Decision Tree",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boujaadamohammed/diabetes-prediction-project",
    project_urls={
        "Bug Tracker": "https://github.com/boujaadamohammed/diabetes-prediction-project/issues",
        "Documentation": "https://github.com/boujaadamohammed/diabetes-prediction-project#readme",
        "Source Code": "https://github.com/boujaadamohammed/diabetes-prediction-project",
    },
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.0",
            "jupyter>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    keywords=[
        "machine-learning",
        "diabetes-prediction",
        "classification",
        "knn",
        "random-forest",
        "decision-tree",
        "scikit-learn",
        "healthcare",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
