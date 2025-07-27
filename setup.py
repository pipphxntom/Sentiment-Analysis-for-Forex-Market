from setuptools import setup, find_packages

setup(
    name="forex-sentiment-analyzer",
    version="0.1.0",
    description="A tool for analyzing forex market sentiment.",
    py_modules=["app", "backtester", "data_processor", "sentiment_analyzer", "utils"],
    install_requires=[
        "matplotlib>=3.10.1",
        "nltk>=3.9.1",
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "plotly>=6.0.1",
        "streamlit>=1.44.0",
        "textblob>=0.19.0",
    ],
)
