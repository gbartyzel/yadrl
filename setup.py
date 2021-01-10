from setuptools import find_packages, setup

setup(name='yadrl',
      packages=[package for package in find_packages()
                if package.startswith('yadrl')],
      install_requires=[
            'torch',
            'numpy',
            'gym',
            'pyyaml'
      ],
      author="Grzegorz Bartyzel",
      url='https://github.com/Souphis/yadrl',
      author_email="gbartyzel@hotmail.com",
      license='MIT',
      version="0.0.0")
