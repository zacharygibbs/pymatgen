#!/usr/bin/env python

from __future__ import division

'''
Created on Nov 8, 2011
'''

__author__ = "Shyue Ping Ong, alan Dozier"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__date__ = "May 11, 2012"

import argparse
from collections import OrderedDict

from pymatgen.io.feffio import FeffLdos
from pymatgen.electronic_structure.plotter import DosPlotter

parser = argparse.ArgumentParser(description='''Convenient DOS Plotter for Feff runs.
Author: Shyue Ping Ong, Alan Dozier
Version: 1.0
Last updated: May 10, 2012''')
parser.add_argument('filename', metavar='filename', type=str, nargs=1, help='ldos%% file set to plot')
parser.add_argument('filename1', metavar='filename1', type=str, nargs=1, help='feff.inp input file ')
parser.add_argument('-s', '--site', dest='site', action='store_const', const=True, help='plot site projected DOS')
parser.add_argument('-e', '--element', dest='element', action='store_const', const=True, help='plot element projected DOS')
parser.add_argument('-o', '--orbital', dest="orbital", action='store_const', const=True, help='plot orbital projected DOS')


args = parser.parse_args()
f = FeffLdos(args.filename1[0], args.filename[0])
dos = f.complete_dos

all_dos = OrderedDict()
all_dos['Total'] = dos

structure = f.structure

if args.site:
    for i in xrange(len(structure)):
        site = structure[i]
        all_dos['Site ' + str(i) + " " + site.specie.symbol] = dos.get_site_dos(site)
if args.element:
    all_dos.update(dos.get_element_dos())
if args.orbital:
    all_dos.update(dos.get_spd_dos())

plotter = DosPlotter()
plotter.add_dos_dict(all_dos)
plotter.show()