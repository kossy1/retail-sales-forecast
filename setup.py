"""
Retail Sales Forecasting Package
================================
A production-ready machine learning system for retail inventory forecasting.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="retail-sales-forecast",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="ML-based sales forecasting for retail inventory optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourcompany/retail-sales-forecast",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=requirements,
    
    # Additional metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    # Python version constraint
    python_requires=">=3.9, <3.12",
    
    # Entry points (CLI commands)
    entry_points={
        "console_scripts": [
            "retail-train=src.models.train:main",
            "retail-predict=src.models.predict:main",
            "retail-backtest=src.models.evaluate:run_backtest",
        ],
    },
    
    # Include non-Python files (e.g., configs, schemas)
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml"],
    },
    
    # Dependencies for specific extras
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "jupyter>=1.0",
            "pre-commit>=3.0",
        ],
        "gpu": ["torch==2.0.1+cu118"],
        "cloud": ["boto3>=1.28", "google-cloud-storage>=2.10"],
        "viz": ["plotly>=5.0", "seaborn>=0.12"],
    },
)