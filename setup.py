import setuptools

import lottery_ticket_pruner


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='lottery-ticket-pruner',
    version=lottery_ticket_pruner.__version__,
    author='Jim Meyer',
    author_email='jimm@racemed.com',
    description='Enables pruning of Keras DNNs using "lottery ticket" pruning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jim-meyer/lottery_ticket_pruner',
    packages=setuptools.find_packages(),
    requires=['keras(>=2.1.0)', 'numpy(>=1.18.3)'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=('>=3.6')
)
