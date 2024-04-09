"""
afmfit.

A fitting package for Atomic Force Microscopy data.
"""
import afmfit
from os.path import join

__version__ = "0.4.0"
__author__ = 'Rémi Vuillemot'

NOLB_PATH = join(afmfit.__path__[0], join("..",join("nolb", "NOLB")))
CHIMERAX_PATH = "/usr/bin/chimerax"


VMD_PATH = "/usr/local/bin/vmd"
# AFMIZE_PATH = "/home/AD/vuillemr/afmize/bin/afmize"
