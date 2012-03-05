import warnings
import sys
sys.path.append('/home/MDEJONG1/pythonplayground/pymatgen/pymatgen_repo/pymatgen_repo') # (If one does not want to change $PYTHONPATH)
import unittest
import pymatgen
from pymatgen.io.vaspio import Poscar
from pymatgen.io.vaspio import Poscar
from pymatgen.io.vaspio import Vasprun
from pymatgen.io.cifio import CifWriter
from pymatgen.io.cifio import CifParser
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import *
from pymatgen.core.structure_modifier import StructureEditor
import numpy as np
import os

class Clusters(object):
    # todo: write point group symmetry parser 

    def __init__(self, pymatgen_str, numclus=[4,4,4], clus_list=None):
        self._struct = pymatgen_str
        self._clus_list = clus_list 
        self._clus = numclus

    @property
    def make_clus_list(self):
        
        
        cluster_list = []
        
        
        print self._clus
        
        
        return cluster_list





#clus_list = []
#clus_list[0] = '123'
struct = CifParser('aluminum.cif').get_structures()[0]
#print struct.__dict__.keys()
#print struct._sites[2]

obj = Clusters(struct)
#print obj._cluster_list
print obj.__dict__.keys()
#obj.make_clus_list

clus1 = {'p':2, 'm':12, 'sites':np.zeros((8,3))}
clus2 = [6, 5.2437,  np.zeros((20,3))]
obj.cluster = [clus1, clus2]

obj.cluster.append(clus1)

print obj.cluster[2]['sites']

#print obj.cluster[0]['sites']


#print obj.__dict__.keys()
#print obj.pairs

#pair = {'multiplicity': 12, 'maxlength': '4'}
#dict4 = {'Alice': 234, 'maxlength': np.zeros((6,3))}

#print dict4['maxlength']
#cluster = []


#        self._strain = 0.5 * (np.matrix(self._dfm) * np.transpose(np.matrix(self._dfm)) - np.eye(3))

    # return a scaled version of this matrix
#    def get_scaled(self, scale_factor):
#        deformation_matrix = self._dfm * scale_factor
#        return Strain(deformation_matrix)

    # return Green-Lagrange strain matrix
#    @property
#    def strain(self):
#        return self._strain


"""
    @property
    def deformation_matrix(self):
        return self._dfm

#    # construct def. matrix from indices and amount
    @staticmethod
    def from_ind_amt_dfm(matrixpos, amt):
        F = np.identity(3)
        F[matrixpos] = F[matrixpos] + amt
        return Strain(F)

    def __eq__(self, other):
        df, df2 = self.deformation_matrix, other.deformation_matrix
i        for i, row in enumerate(df):
            for j, item in enumerate(row):
                if df[i][j] != df2[i][j]:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        #for now, just use a sum of elements * row * column^2
        df = self.deformation_matrix
        h_sum = 0.0
        for i, row in enumerate(df):
            for j, item in enumerate(row):
                h_sum += item * (i + 1) * (j + 1) * (j + 1)
        return h_sum


class IndependentStrain(Strain):
   #TODO: add polar decomposition method

    def __init__(self, deformation,tol=0.00000001):
       
"""
