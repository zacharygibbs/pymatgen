#!/usr/bin/env python

"""
This module provides classes to define electronic structure, such as the density of states, etc.
"""

from __future__ import division

__author__ = "Shyue Ping Ong, Vincent L Chevrier, Rickard Armiento, Geoffroy Hautier"
__copyright__ = "Copyright 2011, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__status__ = "Production"
__date__ = "Sep 23, 2011"


class _SpinImpl(object):
    """
    Internal representation of a Spin. 
    Do not use directly.
    """

    def __init__(self, name):
        self._name = name

    def __int__(self):
        return 1 if self._name == "up" else -1

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        if other == None:
            return False
        return self._name == other._name

    def __hash__(self):
        return self.__int__()

    def __str__(self):
        return self._name


class Spin(object):
    """
    Enum type for Spin.  Only up and down.  Design follows somewhat the familiar Java syntax.
    """

    up = _SpinImpl("up")
    down = _SpinImpl("down")
    all_spins = (up, down)

    @staticmethod
    def from_int(i):
        if i == 1:
            return Spin.up
        elif i == -1:
            return Spin.down
        else:
            raise ValueError("Spin integers must be 1 or -1")


class _OrbitalImpl(object):
    """
    Internal representation of an orbital.  Do not use directly. 
    Use the Orbital class enum types.
    """

    def __init__(self, name, vasp_index):
        self._name = name
        self._vasp_index = vasp_index

    def __int__(self):
        return self._vasp_index

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        if other == None:
            return False
        return self._name == other._name

    def __hash__(self):
        return self.__int__()

    @property
    def orbital_type(self):
        return self._name[0].upper()

    def __str__(self):
        return self._name


class Orbital(object):
    """
    Enum type for OrbitalType. Indices are basically the azimutal quantum number, l.
    Design follows somewhat the familiar Java syntax.
    """

    s = _OrbitalImpl("s", 0)
    py = _OrbitalImpl("py", 1)
    pz = _OrbitalImpl("pz", 2)
    px = _OrbitalImpl("px", 3)
    dxy = _OrbitalImpl("dxy", 4)
    dyz = _OrbitalImpl("dyz", 5)
    dz2 = _OrbitalImpl("dz2", 6)
    dxz = _OrbitalImpl("dxz", 7)
    dx2 = _OrbitalImpl("dx2", 8)
    f_3 = _OrbitalImpl("f_3", 9)
    f_2 = _OrbitalImpl("f_2", 10)
    f_1 = _OrbitalImpl("f_1", 11)
    f0 = _OrbitalImpl("f0", 12)
    f1 = _OrbitalImpl("f1", 13)
    f2 = _OrbitalImpl("f2", 14)
    f3 = _OrbitalImpl("f3", 15)

    all_orbitals = (s, py, pz, px, dxy, dyz, dz2, dxz, dx2, f_3, f_2, f_1, f0, f1, f2, f3)

    @staticmethod
    def from_vasp_index(i):
        for orb in Orbital.all_orbitals:
            if int(orb) == i:
                return orb

    @staticmethod
    def from_string(orb_str):
        for orb in Orbital.all_orbitals:
            if str(orb) == orb_str:
                return orb
        raise ValueError("Illegal orbital definition!")
