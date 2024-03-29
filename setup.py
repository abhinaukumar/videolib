from setuptools import setup, find_packages

setup(
    name='videolib',
    author='Abhinau Kumar',
    author_email='ab.kumr98@gmail.com',
    version='0.2.0',
    url='https://github.com/abhinaukumar/videolib',
    description='Package for easy Video IO and color conversion in Python.',
    install_requires=['numpy', 'scikit-video', 'matplotlib'],
    python_requires='>=3.7.0',
    license='MIT License',
    packages=find_packages()
)
