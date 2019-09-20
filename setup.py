import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coocop",
    version="0.1.dev2",
    maintainer="Philip May",
    author="Philip May",
    author_email="pm@eniak.de",
    description="Copyout and CopyPaiting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/t-systems-on-site-services-gmbh/coocop",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
    keywords='augmentation CNN',
    classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
