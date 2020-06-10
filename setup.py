import setuptools


def get_version():
    """ Do this so we don't have to import lottery_ticket_pruner which requires keras which cannot be counted on
    to be installed when this package gets installed.
    """
    with open('lottery_ticket_pruner/__init__.py', 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                version = line.split('=')[1].strip().replace('"', '').replace('\'', '')
                return version
    return ''


def get_long_description():
    with open('README.md', 'r') as fh:
        return fh.read()


setuptools.setup(
    name='lottery-ticket-pruner',
    version=get_version(),
    author='Jim Meyer',
    author_email='jimm@racemed.com',
    description='Enables pruning of Keras DNNs using "lottery ticket" pruning',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/jim-meyer/lottery_ticket_pruner',
    packages=setuptools.find_packages(),
    install_requires=['keras>=2.1.0', 'numpy>=1.18.3'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=('>=3.6')
)
