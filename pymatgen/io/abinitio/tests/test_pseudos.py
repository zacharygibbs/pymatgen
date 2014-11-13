# coding: utf-8

from __future__ import unicode_literals, division, print_function

"""
Created on Fri Mar  8 23:14:02 CET 2013
"""

import os.path
import collections

from pymatgen.util.testing import PymatgenTest
from pymatgen.io.abinitio import *

_test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files')


def ref_file(filename):
    return os.path.join(_test_dir, filename)


def ref_files(*filenames):
    return map(ref_file, filenames)


class PseudoTestCase(PymatgenTest):

    def setUp(self):
        nc_pseudo_fnames = collections.defaultdict(list)
        nc_pseudo_fnames["Si"] = ref_files("14si.pspnc",  "14si.4.hgh", "14-Si.LDA.fhi")

        self.nc_pseudos = collections.defaultdict(list)

        for (symbol, fnames) in nc_pseudo_fnames.items():
            for fname in fnames:
                root, ext = os.path.splitext(fname)
                pseudo = Pseudo.from_file(ref_file(fname))
                self.nc_pseudos[symbol].append(pseudo)

                # Save the pseudo as instance attribute whose name 
                # is constructed with the rule: symbol_ppformat
                attr_name = symbol + "_" + ext[1:]
                if hasattr(self, attr_name):
                    raise RuntimError("self has already the attribute %s" % attr_name)

                setattr(self, attr_name, pseudo)

    def test_nc_pseudos(self):
        """Test norm-conserving pseudopotentials"""

        for (symbol, pseudos) in self.nc_pseudos.items():
            for pseudo in pseudos:
                print(repr(pseudo))
                print(pseudo)
                self.assertTrue(pseudo.isnc)
                self.assertFalse(pseudo.ispaw)
                self.assertEqual(pseudo.Z, 14)
                self.assertEqual(pseudo.symbol, symbol)
                self.assertEqual(pseudo.Z_val, 4)
                self.assertGreaterEqual(pseudo.nlcc_radius, 0.0)
                print(pseudo.as_dict())

                # Test pickle
                self.serialize_with_pickle(pseudo, test_eq=False)

        # HGH pseudos
        pseudo = self.Si_hgh
        self.assertFalse(pseudo.has_nlcc)
        self.assertEqual(pseudo.l_max, 1)
        self.assertEqual(pseudo.l_local, 0)

        # TM pseudos
        pseudo = self.Si_pspnc
        self.assertTrue(pseudo.has_nlcc)
        self.assertEqual(pseudo.l_max, 2)
        self.assertEqual(pseudo.l_local, 2)

        # FHI pseudos
        pseudo = self.Si_fhi
        self.assertFalse(pseudo.has_nlcc)
        self.assertEqual(pseudo.l_max, 3)
        self.assertEqual(pseudo.l_local, 2)
        
        # Test PseudoTable.
        table = PseudoTable(self.nc_pseudos["Si"])
        print(repr(table))
        print(table)
        self.assertTrue(table.allnc)
        self.assertTrue(not table.allpaw)
        self.assertFalse(not table.is_complete)
        assert len(table) == 3
        assert len(table[14]) == 3
        assert len(table.pseudos_with_symbol("Si")) == 3
        assert table.zlist == [14]

        # Test pickle
        self.serialize_with_pickle(table, test_eq=False)

    def test_pawxml_pseudos(self):
        """Test O.GGA_PBE-JTH-paw.xml."""
        oxygen = Pseudo.from_file(ref_file("O.GGA_PBE-JTH-paw.xml"))
        print(repr(oxygen))
        print(oxygen)
        print(oxygen.as_dict())

        self.assertTrue(oxygen.ispaw)
        self.assertTrue(oxygen.symbol == "O" and 
                       (oxygen.Z, oxygen.core, oxygen.valence) == (8, 2, 6),
                        oxygen.Z_val == 6,
                       )

        self.assert_almost_equal(oxygen.paw_radius, 1.4146523028)

        # Test pickle
        new_objs = self.serialize_with_pickle(oxygen, test_eq=False)

        for o in new_objs:
            print(repr(o))
            print(o)
                                                                                 
            self.assertTrue(o.ispaw)
            self.assertTrue(o.symbol == "O" and 
                           (o.Z, o.core, o.valence) == (8, 2, 6),
                            o.Z_val == 6,
                           )
                                                                                 
            self.assert_almost_equal(o.paw_radius, 1.4146523028)

    def test_ncvpsp_pseudo(self):
        """
        Test the NCVPSP Ge pseudo
        """
        ger = Pseudo.from_file(ref_file("ge.oncvpsp"))
        print(repr(ger))
        print(ger)
        print(ger.as_dict())

        self.assertTrue(ger.symbol == "Ge")
        self.assert_equal(ger.Z, 32.0)
        self.assert_equal(ger.Z_val, 4.0)
        self.assertTrue(ger.isnc)
        self.assertFalse(ger.ispaw)
        self.assert_equal(ger.l_max, 2)
        self.assert_equal(ger.l_local, 4)
        self.assert_equal(ger.rcore, None)


if __name__ == "__main__":
    import unittest
    unittest.main()
