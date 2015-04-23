from setuptools import setup

setup(
    name='semeval',
    version='0.2',
    description='Align and Penalize architecture for Semantic Textual Similarity',  # nopep8
    url='https://github.com/juditacs/semeval',
    author='Judit Acs, Gabor Recski',
    author_email='judit@mokk.bme.hu',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='semantics nlp',

    package_dir={'': '.'},
    packages=['semeval'],

   install_requires=["gensim", "nltk"],
)
