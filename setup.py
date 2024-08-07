#     AFMfit - Fitting package for Atomic Force Microscopy data
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from setuptools import setup, find_packages
import afmfit


setup(
    name='afmfit',
    version=afmfit.__version__,
    description='Fitting package for Atomic Force Microscopy data',
    url='https://github.com/mms29/afm-fitting',
    author='Remi Vuillemot',
    author_email='remi.vuillemot@univ-grenoble-alpes.fr',
    license='GPLv3',
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
                      "libasd",
                      "trackpy"
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)

