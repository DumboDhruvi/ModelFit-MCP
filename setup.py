from setuptools import setup, find_packages

setup(
    name="modelfit-mcp",
    version="0.1.1",
    description="Hardware-aware Hugging Face discovery, sizing, and swappable local model gateway for AI agents.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "modelfit = modelfit.cli:main",
            "modelfit-server = modelfit.server:run_stdio_server",
            "modelfit-api = modelfit.local_api:run_api_server",
        ],
    },
)
