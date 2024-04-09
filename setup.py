from setuptools import setup, find_packages
import afmfit


setup(
    name='afmfit',
    version=afmfit.__version__,
    description='Fitting package for Atomic Force Microscopy data',
    url='https://github.com/mms29/afm-fitting',
    author='Remi Vuillemot',
    author_email='remi.vuillemot@univ-grenoble-alpes.fr',
    license='TODO',
    include_package_data=True,
    packages=['afmfit'],
    install_requires=['tqdm',
    		          'numba',
                      'numpy',
                      'scikit-image',
                      'scipy',
                      'matplotlib',
                      'biopython',
                      'mrcfile',  
                      'scikit-learn',
                      'umap-learn',
                      'tiffile',                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)

