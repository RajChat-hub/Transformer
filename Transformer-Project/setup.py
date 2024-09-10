from setuptools import setup, find_packages

setup(
    name='transformer-project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch', 
        'numpy', 
        'pandas', 
        'scikit-learn', 
        'requests'
    ],
    include_package_data=True,
)
