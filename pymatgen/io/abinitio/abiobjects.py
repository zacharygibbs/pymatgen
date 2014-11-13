# coding: utf-8
"""
Low-level objects providing an abstraction for the objects involved in the
calculation.
"""
from __future__ import unicode_literals, division, print_function

import collections
import os
import abc
import six
import numpy as np
import pymatgen.core.units as units

from pprint import pformat
from monty.design_patterns import singleton
from monty.string import is_string
from monty.collections import AttrDict
from pymatgen.core.design_patterns import Enum
from pymatgen.core.units import ArrayWithUnit
from pymatgen.serializers.json_coders import PMGSONable
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure, Molecule

from .netcdf import structure_from_etsf_file


def contract(s):
    """
    >>> assert contract("1 1 1 2 2 3") == "3*1 2*2 1*3"
    >>> assert contract("1 1 3 2 3") == "2*1 1*3 1*2 1*3"
    """
    if not s: return s

    tokens = s.split()
    old = tokens[0]
    count = [[1, old]]

    for t in tokens[1:]:
        if t == old:
            count[-1][0] += 1
        else:
            old = t
            count.append([1, t])

    return " ".join("%d*%s" % (c, t) for c, t in count)


class AbivarAble(six.with_metaclass(abc.ABCMeta, object)):
    """
    An AbivarAble object provides a method to_abivars that returns a dictionary
    with the abinit variables.
    """

    @abc.abstractmethod
    def to_abivars(self):
        """
        Returns a dictionary with the abinit variables.
        """

    #@abc.abstractmethod
    #def from_abivars(cls, vars):
    #    """
    #    Build the object from a dictionary with Abinit variables.
    #    """

    def __str__(self):
        return pformat(self.to_abivars(), indent=1, width=80, depth=None)

    def __contains__(self, key):
        return key in self.to_abivars()


@singleton
class MandatoryVariable(object):
    """
    Singleton used to tag mandatory variables, just because I can use
    the cool syntax: variable is MANDATORY!
    """


@singleton
class DefaultVariable(object):
    """
    Singleton used to tag variables that will have the default value
    """

MANDATORY = MandatoryVariable()
DEFAULT = DefaultVariable()


class SpinMode(collections.namedtuple('SpinMode', "mode nsppol nspinor nspden"), AbivarAble):
    """
    Different configurations of the electron density as implemented in abinit:
    One can use as_spinmode to construct the object via SpinMode.as_spinmode
    (string) where string can assume the values:

        - polarized
        - unpolarized
        - afm (anti-ferromagnetic)
        - spinor (non-collinear magnetism)
        - spinor_nomag (non-collinear, no magnetism)
    """
    @classmethod
    def as_spinmode(cls, obj):
        """Converts obj into a SpinMode instance"""
        if isinstance(obj, cls):
            return obj
        else:
            # Assume a string with mode
            try:
                return _mode2spinvars[obj]
            except KeyError:
                raise KeyError("Wrong value for spin_mode: %s" % str(obj))

    def to_abivars(self):
        return {
            "nsppol": self.nsppol,
            "nspinor": self.nspinor,
            "nspden": self.nspden,
        }

# An handy Multiton
_mode2spinvars = {
    "unpolarized": SpinMode("unpolarized", 1, 1, 1),
    "polarized": SpinMode("polarized", 2, 1, 2),
    "afm": SpinMode("afm", 1, 1, 2),
    "spinor": SpinMode("spinor", 1, 2, 4),
    "spinor_nomag": SpinMode("spinor_nomag", 1, 2, 1),
}


class Smearing(AbivarAble, PMGSONable):
    """
    Variables defining the smearing technique. The preferred way to instanciate
    a Smearing object is via the class method Smearing.as_smearing(string)
    """
    #: Mapping string_mode --> occopt
    _mode2occopt = {
        'nosmearing': 1,
        'fermi_dirac': 3,
        'marzari4': 4,
        'marzari5': 5,
        'methfessel': 6,
        'gaussian': 7}

    def __init__(self, occopt, tsmear):
        self.occopt = occopt
        self.tsmear = tsmear

    def __str__(self):
        s = "occopt %d # %s Smearing\n" % (self.occopt, self.mode)
        if self.tsmear:
            s += 'tsmear %s' % self.tsmear
        return s

    def __eq__(self, other):
        return (self.occopt == other.occopt and
                np.allclose(self.tsmear, other.tsmear))

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        return self.mode != "nosmearing"

    # py2 old version
    __nonzero__ = __bool__

    @classmethod
    def as_smearing(cls, obj):
        """
        Constructs an instance of Smearing from obj. Accepts obj in the form:

            * Smearing instance
            * "name:tsmear"  e.g. "gaussian:0.004"  (Hartree units)
            * "name:tsmear units" e.g. "gaussian:0.1 eV"
            * None --> no smearing
        """
        if obj is None:
            return Smearing.nosmearing()

        if isinstance(obj, cls):
            return obj

        # obj is a string
        obj, tsmear = obj.split(":")
        obj.strip()

        if obj == "nosmearing":
            return cls.nosmearing()
        else:
            occopt = cls._mode2occopt[obj]
            try:
                tsmear = float(tsmear)
            except ValueError:
                tsmear, unit = tsmear.split()
                tsmear = units.Energy(float(tsmear), unit).to("Ha")

            return cls(occopt, tsmear)

    @property
    def mode(self):
        for (mode_str, occopt) in self._mode2occopt.items():
            if occopt == self.occopt:
                return mode_str
        raise AttributeError("Unknown occopt %s" % self.occopt)

    @staticmethod
    def nosmearing():
        return Smearing(1, None)

    def to_abivars(self):
        if self.mode == "nosmearing":
            return {}
        else:
            return {"occopt": self.occopt,
                    "tsmear": self.tsmear,}

    def as_dict(self):
        """json friendly dict representation of Smearing"""
        return {"occopt": self.occopt, "tsmear": self.tsmear,
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__}

    @staticmethod
    def from_dict(d):
        return Smearing(d["occopt"], d["tsmear"])


class ElectronsAlgorithm(dict, AbivarAble):
    """Variables controlling the SCF/NSCF algorithm."""
    # None indicates that we use abinit defaults.
    _DEFAULT = dict(
        iprcell=None, iscf=None, diemac=None, diemix=None, diemixmag=None,
        dielam=None, diegap=None, dielng=None, diecut=None, nstep=50)

    def __init__(self, *args, **kwargs):
        super(ElectronsAlgorithm, self).__init__(*args, **kwargs)

        for k in self:
            if k not in self._DEFAULT:
                raise ValueError("%s: No default value has been provided for "
                                 "key %s" % (self.__class__.__name__, k))

    def to_abivars(self):
        return self.copy()


class Electrons(AbivarAble):
    """
    The electronic degrees of freedom
    """
    def __init__(self, spin_mode="polarized", smearing="fermi_dirac:0.1 eV",
                 algorithm=None, nband=None, fband=None, charge=0.0, comment=None):  # occupancies=None,
        """
        Constructor for Electrons object.

        Args:
            comment:
                String comment for Electrons

            charge: float
                    Total charge of the system. Default is 0.
        """
        super(Electrons, self).__init__()

        self.comment = comment
        self.smearing = Smearing.as_smearing(smearing)
        self.spin_mode = SpinMode.as_spinmode(spin_mode)

        self.nband = nband
        self.fband = fband
        self.charge = charge
        self.algorithm = algorithm

    @property
    def nsppol(self):
        return self.spin_mode.nsppol

    @property
    def nspinor(self):
        return self.spin_mode.nspinor

    @property
    def nspden(self):
        return self.spin_mode.nspden

    #@property
    #def as_dict(self):
    #    "json friendly dict representation"
    #    d = {}
    #    d["@module"] = self.__class__.__module__
    #    d["@class"] = self.__class__.__name__
    #    raise NotImplementedError("")
    #    return d

    #@staticmethod
    #def from_dict(d):
    #    raise NotImplementedError("")

    def to_abivars(self):
        abivars = self.spin_mode.to_abivars()

        abivars.update({
            "nband"  : self.nband,
            "fband"  : self.fband,
            "charge" : self.charge,
        })

        if self.smearing:
            abivars.update(self.smearing.to_abivars())

        if self.algorithm:
            abivars.update(self.algorithm)

        abivars["#comment"] = self.comment
        return abivars


def asabistructure(obj):
    """
    Convert obj into an AbiStructure object. Accepts:

        - AbiStructure instance
        - Subinstances of pymatgen.
        - File paths
    """
    if isinstance(obj, AbiStructure):
        return obj

    if isinstance(obj, Structure):
        # Promote
        return AbiStructure(obj)

    if is_string(obj):
        # Handle file paths.
        if os.path.isfile(obj):

            if obj.endswith(".nc"):
                structure = structure_from_etsf_file(obj)
            else:
                structure = Structure.from_file(obj)

            # Promote
            return AbiStructure(structure)

    raise ValueError("Don't know how to convert object %s to an AbiStructure structure" % str(obj))


class AbiStructure(Structure, AbivarAble):
    """Patches the pymatgen structure adding the method to_abivars."""

    @staticmethod
    def asabistructure(obj):
        return asabistructure(obj)

    def __new__(cls, structure):
        new = structure
        new.__class__ = cls
        return new

    def __init__(self, structure):
        pass

    @classmethod
    def boxed_molecule(cls, pseudos, cart_coords, acell=3*(10,)):
        """
        Creates a molecule in a periodic box of lengths acell [Bohr]

        Args:
            pseudos:
                List of pseudopotentials
            cart_coords:
                Cartesian coordinates
            acell:
                Lengths of the box in *Bohr*
        """
        cart_coords = np.atleast_2d(cart_coords)

        molecule = Molecule([p.symbol for p in pseudos], cart_coords)

        l = ArrayWithUnit(acell, "bohr").to("ang")

        structure = molecule.get_boxed_structure(l[0], l[1], l[2])

        return cls(structure)

    @classmethod
    def boxed_atom(cls, pseudo, cart_coords=3*(0,), acell=3*(10,)):
        """
        Creates an atom in a periodic box of lengths acell [Bohr]

        Args:
            pseudo:
                Pseudopotential object.
            cart_coords:
                Cartesian coordinates
            acell:
                Lengths of the box in *Bohr*
        """
        return cls.boxed_molecule([pseudo], cart_coords, acell=acell)

    def get_sorted_structure(self):
        """
        orders the structure according to increasing Z of the elements
        """
        sites = sorted(self.sites, key=lambda site: site.specie.Z)
        structure = Structure.from_sites(sites)
        return AbiStructure(structure)

    def to_abivars(self):
        "Returns a dictionary with the abinit variables."
        types_of_specie = self.types_of_specie
        natom = self.num_sites

        znucl_type = [specie.number for specie in types_of_specie]

        znucl_atoms = self.atomic_numbers

        typat = np.zeros(natom, np.int)
        for (atm_idx, site) in enumerate(self):
            typat[atm_idx] = types_of_specie.index(site.specie) + 1

        rprim = ArrayWithUnit(self.lattice.matrix, "ang").to("bohr")
        xred = np.reshape([site.frac_coords for site in self], (-1,3))

        # Set small values to zero. This usually happens when the CIF file
        # does not give structure parameters with enough digits.
        #rprim = np.where(np.abs(rprim) > 1e-8, rprim, 0.0)
        #xred = np.where(np.abs(xred) > 1e-8, xred, 0.0)

        d = dict(
            natom=natom,
            ntypat=len(types_of_specie),
            typat=typat,
            xred=xred,
            znucl=znucl_type)

        d.update(dict(
            acell=3 * [1.0],
            rprim=rprim))

        #d.update(dict(
        #    acell=3 * [1.0],
        #    angdeg))

        return d


class KSampling(AbivarAble):
    """
    Input variables defining the K-point sampling.
    """
    # Modes supported by the constructor.
    modes = Enum(('monkhorst', 'path', 'automatic',))

    def __init__(self, mode="monkhorst", num_kpts= 0, kpts=((1, 1, 1),), kpt_shifts=(0.5, 0.5, 0.5),
                 kpts_weights=None, use_symmetries=True, use_time_reversal=True, chksymbreak=None,
                 comment=None):
        """
        Highly flexible constructor for KSampling objects.  The flexibility comes
        at the cost of usability and in general, it is recommended that you use
        the default constructor only if you know exactly what you are doing and
        requires the flexibility.  For most usage cases, the object be constructed
        far more easily using the convenience static constructors:

            #. gamma_only
            #. gamma_centered
            #. monkhorst
            #. monkhorst_automatic
            #. path

        and it is recommended that you use those.

        Args:
            mode:
                Mode for generating k-poits. Use one of the KSampling.modes
                enum types.
            num_kpts:
                Number of kpoints if mode is "automatic"
                Number of division for the sampling of the smallest segment if
                mode is "path".
                Not used for the other modes
            kpts:
                Number of divisions. Even when only a single specification is
                required, e.g. in the automatic scheme, the kpts should still
                be specified as a 2D array. e.g., [[20]] or [[2,2,2]].
            kpt_shifts:
                Shifts for Kpoints.
            use_symmetries:
                False if spatial symmetries should not be used to reduced the
                number of independent k-points.
            use_time_reversal:
                False if time-reversal symmetry should not be used to reduced
                the number of independent k-points.
            kpts_weights:
                Optional weights for kpoints. For explicit kpoints.
            chksymbreak:
                Abinit input variable: check whether the BZ sampling preserves
                the symmetry of the crystal.
            comment:
                String comment for Kpoints

        The default behavior of the constructor is monkhorst.
        """
        if mode not in KSampling.modes:
            raise ValueError("Unknown kpoint mode %s" % mode)

        super(KSampling, self).__init__()

        self.mode = mode
        self.comment = comment

        abivars = {}

        if mode in ("monkhorst",):
            assert num_kpts == 0
            ngkpt  = np.reshape(kpts, (-1,3))
            shiftk = np.reshape(kpt_shifts, (-1,3))

            if use_symmetries and use_time_reversal: kptopt = 1
            if not use_symmetries and use_time_reversal: kptopt = 2
            if not use_symmetries and not use_time_reversal: kptopt = 3
            if use_symmetries and not use_time_reversal: kptopt = 4

            abivars.update({
                "ngkpt"      : ngkpt,
                "shiftk"     : shiftk,
                "nshiftk"    : len(shiftk),
                "kptopt"     : kptopt,
                "chksymbreak": chksymbreak,
            })

        elif mode in ("path",):
            if num_kpts <= 0:
                raise ValueError("For Path mode, num_kpts must be specified and >0")

            kptbounds = np.reshape(kpts, (-1,3,))
            #print("in path with kptbound: %s " % kptbounds)

            abivars.update({
                "ndivsm"   : num_kpts,
                "kptbounds": kptbounds,
                "kptopt"   : -len(kptbounds)+1,
            })

        elif mode in ("automatic",):
            kpts = np.reshape(kpts, (-1,3))
            if len(kpts) != num_kpts:
                raise ValueError("For Automatic mode, num_kpts must be specified.")

            kptnrm = np.ones(num_kpts)

            abivars.update({
                "kptopt"     : 0,
                "kpt"        : kpts,
                "nkpt"       : num_kpts,
                "kptnrm"     : kptnrm,
                "wtk"        : kpts_weights,  # for iscf/=-2, wtk.
                "chksymbreak": chksymbreak,
            })

        else:
            raise ValueError("Unknown mode %s" % mode)

        self.abivars = abivars
        self.abivars["#comment"] = comment

    @property
    def is_homogeneous(self):
        return self.mode not in ["path"]

    @classmethod
    def gamma_only(cls):
        """Gamma-only sampling"""
        return cls(kpt_shifts=(0.0,0.0,0.0), comment="Gamma-only sampling")

    @classmethod
    def gamma_centered(cls, kpts=(1, 1, 1), use_symmetries=True, use_time_reversal=True):
        """
        Convenient static constructor for an automatic Gamma centered Kpoint grid.

        Args:
            kpts:
                Subdivisions N_1, N_2 and N_3 along reciprocal lattice vectors.
            use_symmetries:
                False if spatial symmetries should not be used to reduced the number of independent k-points.
            use_time_reversal:
                False if time-reversal symmetry should not be used to reduced the number of independent k-points.

        Returns:
            `KSampling` object.
        """
        return cls(kpts=[kpts], kpt_shifts=(0.0, 0.0, 0.0),
                   use_symmetries=use_symmetries, use_time_reversal=use_time_reversal,
                   comment="gamma-centered mode")

    @classmethod
    def monkhorst(cls, ngkpt, shiftk=(0.5, 0.5, 0.5), chksymbreak=None, use_symmetries=True,
                  use_time_reversal=True, comment=None):
        """
        Convenient static constructor for a Monkhorst-Pack mesh.

        Args:
            ngkpt:
                Subdivisions N_1, N_2 and N_3 along reciprocal lattice vectors.
            shiftk:
                Shift to be applied to the kpoints. 
            use_symmetries:
                Use spatial symmetries to reduce the number of k-points.
            use_time_reversal:
                Use time-reversal symmetry to reduce the number of k-points.

        Returns:
            `KSampling` object.
        """
        return cls(
            kpts=[ngkpt], kpt_shifts=shiftk,
            use_symmetries=use_symmetries, use_time_reversal=use_time_reversal, chksymbreak=chksymbreak,
            comment=comment if comment else "Monkhorst-Pack scheme with user-specified shiftk")

    @classmethod
    def monkhorst_automatic(cls, structure, ngkpt,
                            use_symmetries=True, use_time_reversal=True, chksymbreak=None, comment=None):
        """
        Convenient static constructor for an automatic Monkhorst-Pack mesh.

        Args:
            structure:
                paymatgen structure object.
            ngkpt:
                Subdivisions N_1, N_2 and N_3 along reciprocal lattice vectors.
            use_symmetries:
                Use spatial symmetries to reduce the number of k-points.
            use_time_reversal:
                Use time-reversal symmetry to reduce the number of k-points.

        Returns:
            KSampling object
        """
        sg = SpacegroupAnalyzer(structure)
        #sg.get_crystal_system()
        #sg.get_point_group()
        # TODO
        nshiftk = 1
        #shiftk = 3*(0.5,) # this is the default
        shiftk = 3*(0.5,)

        #if lattice.ishexagonal:
        #elif lattice.isbcc
        #elif lattice.isfcc

        return cls.monkhorst(
            ngkpt, shiftk=shiftk, use_symmetries=use_symmetries, use_time_reversal=use_time_reversal,
            chksymbreak=chksymbreak, comment=comment if comment else "Automatic Monkhorst-Pack scheme")

    @classmethod
    def _path(cls, ndivsm, structure=None, kpath_bounds=None, comment=None):
        """
        Static constructor for path in k-space.

        Args:
            structure:
                pymatgen structure.
            kpath_bounds:
                List with the reduced coordinates of the k-points defining the path.
            ndivsm:
                Number of division for the smallest segment.
            comment:
                Comment string.

        Returns:
            KSampling object
        """
        if kpath_bounds is None:
            # Compute the boundaries from the input structure.
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            sp = HighSymmKpath(structure)

            # Flat the array since "path" is a a list of lists!
            kpath_labels = []
            for labels in sp.kpath["path"]:
                kpath_labels.extend(labels)

            kpath_bounds = []
            for label in kpath_labels:
                red_coord = sp.kpath["kpoints"][label]
                #print("label %s, red_coord %s" % (label, red_coord))
                kpath_bounds.append(red_coord)

        return cls(mode=KSampling.modes.path, num_kpts=ndivsm, kpts=kpath_bounds,
                   comment=comment if comment else "K-Path scheme")

    @classmethod
    def path_from_structure(cls, ndivsm, structure):
        """See _path for the meaning of the variables"""
        return cls._path(ndivsm,  structure=structure, comment="K-path generated automatically from pymatgen structure")

    @classmethod
    def explicit_path(cls, ndivsm, kpath_bounds):
        """See _path for the meaning of the variables"""
        return cls._path(ndivsm, kpath_bounds=kpath_bounds, comment="Explicit K-path")

    @classmethod
    def automatic_density(cls, structure, kppa, chksymbreak=None, use_symmetries=True, use_time_reversal=True,
                          shifts=(0.5, 0.5, 0.5)):
        """
        Returns an automatic Kpoint object based on a structure and a kpoint
        density. Uses Gamma centered meshes for hexagonal cells and Monkhorst-Pack grids otherwise.

        Algorithm:
            Uses a simple approach scaling the number of divisions along each
            reciprocal lattice vector proportional to its length.

        Args:
            structure:
                Input structure
            kppa:
                Grid density
        """
        lattice = structure.lattice
        lengths = lattice.abc
        ngrid = kppa / structure.num_sites

        mult = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3.)

        num_div = [int(round(1.0 / lengths[i] * mult)) for i in range(3)]
        # ensure that num_div[i] > 0
        num_div = [i if i > 0 else 1 for i in num_div]

        angles = lattice.angles
        hex_angle_tol = 5      # in degrees
        hex_length_tol = 0.01  # in angstroms

        right_angles = [i for i in range(3) if abs(angles[i] - 90) < hex_angle_tol]

        hex_angles = [i for i in range(3)
                      if abs(angles[i] - 60) < hex_angle_tol or
                      abs(angles[i] - 120) < hex_angle_tol]

        is_hexagonal = (len(right_angles) == 2 and len(hex_angles) == 1
                        and abs(lengths[right_angles[0]] -
                                lengths[right_angles[1]]) < hex_length_tol)

        #style = Kpoints.modes.gamma
        #if not is_hexagonal:
        #    num_div = [i + i % 2 for i in num_div]
        #    style = Kpoints.modes.monkhorst

        comment = "pymatgen generated KPOINTS with grid density = " + "{} / atom".format(kppa)

        shifts = np.reshape(shifts, (-1, 3))

        return cls(
            mode="monkhorst", num_kpts=0, kpts=[num_div], kpt_shifts=shifts,
            use_symmetries=use_symmetries, use_time_reversal=use_time_reversal, chksymbreak=chksymbreak,
            comment=comment)

    def to_abivars(self):
        return self.abivars


class Constraints(AbivarAble):
    """This object defines the constraints for structural relaxation"""
    def to_abivars(self):
        raise NotImplementedError("")


class RelaxationMethod(AbivarAble):
    """
    This object stores the variables for the (constrained) structural optimization
    ionmov and optcell specify the type of relaxation.
    The other variables are optional and their use depend on ionmov and optcell.
    A None value indicates that we use abinit default. Default values can
    be modified by passing them to the constructor.
    The set of variables are constructed in to_abivars depending on ionmov and optcell.
    """
    _default_vars =  {
        "ionmov"           : MANDATORY,
        "optcell"          : MANDATORY,
        "ntime"            : 80,
        "dilatmx"          : 1.05,
        "ecutsm"           : 0.5,
        "strfact"          : None,
        "tolmxf"           : None,
        "strtarget"        : None,
        "atoms_constraints": {}, # Constraints are stored in a dictionary. {} means if no constraint is enforced.
    }

    IONMOV_DEFAULT = 3
    OPTCELL_DEFAULT = 2

    def __init__(self, *args, **kwargs):

        # Initialize abivars with the default values.
        self.abivars = self._default_vars

        # Overwrite the keys with the args and kwargs passed to constructor.
        self.abivars.update(*args, **kwargs)

        self.abivars = AttrDict(self.abivars)

        for k in self.abivars:
            if k not in self._default_vars:
                raise ValueError("%s: No default value has been provided for key %s" % (self.__class__.__name__, k))

        for k in self.abivars:
            if k is MANDATORY:
                raise ValueError("%s: No default value has been provided for the mandatory key %s" %
                                 (self.__class__.__name__, k))

    @classmethod
    def atoms_only(cls, atoms_constraints=None):
        if atoms_constraints is None:
            new = cls(ionmov=cls.IONMOV_DEFAULT, optcell=0)
        else:
            new = cls(ionmov=cls.IONMOV_DEFAULT, optcell=0, atoms_constraints=atoms_constraints)
        return new

    @classmethod
    def atoms_and_cell(cls, atoms_constraints=None):
        if atoms_constraints is None:
            new = cls(ionmov=cls.IONMOV_DEFAULT, optcell=cls.OPTCELL_DEFAULT)
        else:
            new = cls(ionmov=cls.IOMOV_DEFAULT, optcell=cls.OPTCELL_DEFAULT, atoms_constraints=atoms_constraints)
        return new

    @property
    def move_atoms(self):
        """True if atoms must be moved."""
        return self.abivars.ionmov != 0

    @property
    def move_cell(self):
        """True if lattice parameters must be optimized."""
        return self.abivars.optcell != 0

    def to_abivars(self):
        """Returns a dictionary with the abinit variables"""
        # These variables are always present.
        out_vars = {
            "ionmov" : self.abivars.ionmov,
            "optcell": self.abivars.optcell,
            "ntime"  : self.abivars.ntime,
        }

        # Atom relaxation.
        if self.move_atoms:
            out_vars.update({
                "tolmxf": self.abivars.tolmxf,
            })

        if self.abivars.atoms_constraints:
            # Add input variables for constrained relaxation.
            raise NotImplementedError("")
            out_vars.update(self.abivars.atoms_constraints.to_abivars())

        # Cell relaxation.
        if self.move_cell:
            out_vars.update({
                "dilatmx"  : self.abivars.dilatmx,
                "ecutsm"   : self.abivars.ecutsm,
                "strfact"  : self.abivars.strfact,
                "strtarget": self.abivars.strtarget,
            })

        return out_vars


class PPModel(AbivarAble, PMGSONable):
    """
    Parameters defining the plasmon-pole technique.
    The common way to instanciate a PPModel object is via the class method
    PPModel.as_ppmodel(string)
    """
    _mode2ppmodel = {
        "noppmodel": 0,
        "godby"    : 1,
        "hybersten": 2,
        "linden"   : 3,
        "farid"    : 4,
    }

    modes = Enum(k for k in _mode2ppmodel)

    @classmethod
    def as_ppmodel(cls, obj):
        """
        Constructs an instance of PPModel from obj.

        Accepts obj in the form:
            * PPmodel instance
            * string. e.g "godby:12.3 eV", "linden".
        """
        if isinstance(obj, cls):
            return obj

        # obj is a string
        if ":" not in obj:
            mode, plasmon_freq = obj, None
        else:
            # Extract mode and plasmon_freq
            mode, plasmon_freq = obj.split(":")
            try:
                plasmon_freq = float(plasmon_freq)
            except ValueError:
                plasmon_freq, unit = plasmon_freq.split()
                plasmon_freq = units.Energy(float(plasmon_freq), unit).to("Ha")

        return cls(mode=mode, plasmon_freq=plasmon_freq)

    def __init__(self, mode="godby", plasmon_freq=None):
        assert mode in PPModel.modes
        self.mode = mode
        self.plasmon_freq = plasmon_freq

    def __eq__(self, other):
        if other is None:
            return False
        else:
            if self.mode != other.mode:
                return False

            if self.plasmon_freq is None:
                return other.plasmon_freq is None
            else:
                return np.allclose(self.plasmon_freq, other.plasmon_freq)

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        return self.mode != "noppmodel"

    # py2 old version
    __nonzero__ = __bool__

    def __repr__(self):
        return "<%s at %s, mode = %s>" % (self.__class__.__name__, id(self),
                                          str(self.mode))

    def to_abivars(self):
        if self:
            return {"ppmodel": self._mode2ppmodel[self.mode], "ppmfrq": self.plasmon_freq}
        else:
            return {}

    @classmethod
    def noppmodel(cls):
        return cls(mode="noppmodel", plasmon_freq=None)

    def as_dict(self):
        return {"mode": self.mode, "plasmon_freq": self.plasmon_freq,
                "@module": self.__class__.__module__,
                "@class": self.__class__.__name__}

    @staticmethod
    def from_dict(d):
        return PPModel(mode=d["mode"], plasmon_freq=d["plasmon_freq"])


class HilbertTransform(AbivarAble):
    """
    Parameters for the Hilbert-transform method (Screening code)
    i.e. the parameters defining the frequency mesh used for the spectral function
    and the frequency mesh used for the polarizability
    """
    def __init__(self, nomegasf, domegasf=None, spmeth=1, nfreqre=None, freqremax=None, nfreqim=None, freqremin=None):
        """
        Args:
            nomegasf:
                Number of points for sampling the spectral function along the real axis.
            domegasf:
                Step in Ha for the linear mesh used for the spectral function.
            spmeth:
                Algorith for the representation of the delta function.
            nfreqre:
                Number of points along the real axis (linear mesh).
            freqremax:
                Maximum frequency for W along the real axis (in hartree).
            nfreqim:
                Number of point along the imaginary axis (Gauss-Legendre mesh).
            freqremin:
                Minimum frequency for W along the real axis (in hartree).
        """
        # Spectral function
        self.nomegasf = nomegasf
        self.domegasf = domegasf
        self.spmeth = spmeth

        # Mesh for the contour-deformation method used for the integration of the self-energy
        self.nfreqre = nfreqre
        self.freqremax = freqremax
        self.freqremin = freqremin
        self.nfreqim = nfreqim

    def to_abivars(self):
        """Returns a dictionary with the abinit variables"""
        return {
                # Spectral function
                "nomegasf": self.nomegasf,
                "domegasf": self.domegasf,
                "spmeth"  : self.spmeth,
                 # Frequency mesh for the polarizability
                "nfreqre"  : self.nfreqre,
                "freqremax": self.freqremax,
                "nfreqim"  : self.nfreqim,
                "freqremin": self.freqremin,
                }


class ModelDielectricFunction(AbivarAble):
    """Model dielectric function used for BSE calculation"""
    def __init__(self, mdf_epsinf):
        self.mdf_epsinf = mdf_epsinf

    def to_abivars(self):
        return {"mdf_epsinf": self.mdf_epsinf}

##########################################################################################
#################################  WORK IN PROGRESS ######################################
##########################################################################################


class Screening(AbivarAble):
    """
    This object defines the parameters used for the
    computation of the screening function.
    """
    # Approximations used for W
    _WTYPES = {
        "RPA": 0,
    }

    # Self-consistecy modes
    _SC_MODES = {
        "one_shot"     : 0,
        "energy_only"  : 1,
        "wavefunctions": 2,
    }

    def __init__(self, ecuteps, nband, w_type="RPA", sc_mode="one_shot",
                 hilbert=None, ecutwfn=None, inclvkb=2):
        """
        Args:
            ecuteps:
                Cutoff energy for the screening (Ha units).
            nband
                Number of bands for the Green's function
            w_type:
                Screening type
            sc_mode:
                Self-consistency mode.
            hilbert:
                Instance of `HilbertTransform` defining the paramentes for the Hilber transform method.
            ecutwfn:
                Cutoff energy for the wavefunctions (Default: ecutwfn == ecut).
            inclvkb=2
                Option for the treatment of the dipole matrix elements (NC pseudos).
        """
        if w_type not in self._WTYPES:
            raise ValueError("W_TYPE: %s is not supported" % w_type)

        if sc_mode not in self._SC_MODES:
            raise ValueError("Self-consistecy mode %s is not supported" % sc_mode)

        self.ecuteps = ecuteps
        self.nband   = nband
        self.w_type  = w_type
        self.sc_mode = sc_mode

        self.ecutwfn = ecutwfn
        self.inclvkb = inclvkb

        if hilbert is not None:
            raise NotImplementedError("Hilber transform not coded yet")
            self.hilbert = hilbert

        # Default values
        # TODO Change abinit defaults
        self.gwpara = 2
        self.awtr   = 1
        self.symchi = 1

    @property
    def use_hilbert(self):
        return hasattr(self, "hilbert")

    #@property
    #def gwcalctyp(self):
    #    "Return the value of the gwcalctyp input variable"
    #    dig0 = str(self._SIGMA_TYPES[self.type])
    #    dig1 = str(self._SC_MODES[self.sc_mode])
    #    return dig1.strip() + dig0.strip()

    def to_abivars(self):
        "Returns a dictionary with the abinit variables"
        abivars = {
            "ecuteps"   : self.ecuteps,
            "ecutwfn"   : self.ecutwfn,
            "inclvkb"   : self.inclvkb,
            "gwpara"    : self.gwpara,
            "awtr"      : self.awtr,
            "symchi"    : self.symchi,
            #"gwcalctyp": self.gwcalctyp,
            #"fftgw"    : self.fftgw,
        }

        # Variables for the Hilber transform.
        if self.use_hilbert:
            abivars.update(self.hilbert.to_abivars())

        return abivars


class SelfEnergy(AbivarAble):
    """
    This object defines the parameters used for the
    computation of the self-energy.
    """
    _SIGMA_TYPES = {
        "gw"          : 0,
        "hartree_fock": 5,
        "sex"         : 6,
        "cohsex"      : 7,
        "model_gw_ppm": 8,
        "model_gw_cd" : 9,
    }

    _SC_MODES = {
        "one_shot"     : 0,
        "energy_only"  : 1,
        "wavefunctions": 2,
    }

    def __init__(self, se_type, sc_mode, nband, ecutsigx, screening,
                 gw_qprange=1, ppmodel=None, ecuteps=None, ecutwfn=None, gwpara=2):
        """
        Args:
            se_type:
                Type of self-energy (str)
            sc_mode:
                Self-consistency mode.
            nband
                Number of bands for the Green's function
            ecutsigx:
                Cutoff energy for the exchange part of the self-energy (Ha units).
            screening:
                `Screening` instance.
            gw_qprange:
                Option for the automatic selection of k-points and bands for GW corrections.
                See Abinit docs for more detail. The default value makes the code computie the
                QP energies for all the point in the IBZ and one band above and one band below the Fermi level.
            ppmodel:
                `PPModel` instance with the parameters used for the plasmon-pole technique.
            ecuteps:
                Cutoff energy for the screening (Ha units).
            ecutwfn:
                Cutoff energy for the wavefunctions (Default: ecutwfn == ecut).
        """
        if se_type not in self._SIGMA_TYPES:
            raise ValueError("SIGMA_TYPE: %s is not supported" % se_type)

        if sc_mode not in self._SC_MODES:
            raise ValueError("Self-consistecy mode %s is not supported" % sc_mode)

        self.type = se_type
        self.sc_mode = sc_mode
        self.nband = nband
        self.ecutsigx = ecutsigx
        self.screening = screening
        self.gw_qprange = gw_qprange
        self.gwpara = gwpara

        if ppmodel is not None:
            assert not screening.use_hilbert
            self.ppmodel = PPModel.as_ppmodel(ppmodel)

        self.ecuteps = ecuteps if ecuteps is not None else screening.ecuteps
        self.ecutwfn = ecutwfn

        #band_mode in ["gap", "full"]

        #if isinstance(kptgw, str) and kptgw == "all":
        #    self.kptgw = None
        #    self.nkptgw = None
        #else:
        #    self.kptgw = np.reshape(kptgw, (-1,3))
        #    self.nkptgw =  len(self.kptgw)

        #if bdgw is None:
        #    raise ValueError("bdgw must be specified")

        #if isinstance(bdgw, str):
        #    # TODO add new variable in Abinit so that we can specify
        #    # an energy interval around the KS gap.
        #    homo = float(nele) / 2.0
        #    #self.bdgw =

        #else:
        #    self.bdgw = np.reshape(bdgw, (-1,2))

        #self.freq_int = freq_int

    @property
    def use_ppmodel(self):
        """True if we are using the plasmon-pole approximation."""
        return hasattr(self, "ppmodel")

    @property
    def gwcalctyp(self):
        """Returns the value of the gwcalctyp input variable."""
        dig0 = str(self._SIGMA_TYPES[self.type])
        dig1 = str(self._SC_MODES[self.sc_mode])
        return dig1.strip() + dig0.strip()

    @property
    def symsigma(self):
        """1 if symmetries can be used to reduce the number of q-points."""
        return 1 if self.sc_mode == "one_shot" else 0

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        abivars = dict(
            gwcalctyp=self.gwcalctyp,
            ecuteps=self.ecuteps,
            ecutsigx=self.ecutsigx,
            symsigma=self.symsigma,
            gw_qprange=self.gw_qprange,
            gwpara=self.gwpara
            #"ecutwfn"  : self.ecutwfn,
            #"kptgw"    : self.kptgw,
            #"nkptgw"   : self.nkptgw,
            #"bdgw"     : self.bdgw,
        )

        # FIXME: problem with the spin
        #assert len(self.bdgw) == self.nkptgw

        # ppmodel variables
        if self.use_ppmodel:
            abivars.update(self.ppmodel.to_abivars())

        return abivars


class ExcHamiltonian(AbivarAble):
    """This object contains the parameters for the solution of the Bethe-Salpeter equation."""
    # Types of excitonic Hamiltonian.
    _EXC_TYPES = {
        "TDA": 0,          # Tamm-Dancoff approximation.
        "coupling": 1,     # Calculation with coupling.
    }

    # Algorithms used to compute the macroscopic dielectric function
    # and/or the exciton wavefunctions.
    _ALGO2VAR = {
        "direct_diago": 1,
        "haydock"     : 2,
        "cg"          : 3,
    }

    # Options specifying the treatment of the Coulomb term.
    _COULOMB_MODES = [
        "diago",
        "full",
        "model_df"
        ]

    def __init__(self, bs_loband, nband, soenergy, coulomb_mode, ecuteps, spin_mode="polarized", mdf_epsinf=None,
                exc_type="TDA", algo="haydock", with_lf=True, bs_freq_mesh=None, zcut=None, **kwargs):
        """
        Args:
            bs_loband:
                Lowest band index used in the e-h  basis set. Can be scalar or array of shape (nsppol,)
            nband:
                Max band index used in the e-h  basis set.
            soenergy:
                Scissors energy in Hartree.
            coulomb_mode:
                Treatment of the Coulomb term.
            ecuteps:
                Cutoff energy for W in Hartree.
            mdf_epsinf:
                Macroscopic dielectric function :math:`\epsilon_\inf` used in
                the model dielectric function.
            exc_type:
                Approximation used for the BSE Hamiltonian
            with_lf:
                True if local field effects are included <==> exchange term is included
            bs_freq_mesh:
                Frequency mesh for the macroscopic dielectric function (start, stop, step) in Ha.
            zcut:
                Broadening parameter in Ha.
            **kwargs:
                Extra keywords
        """
        spin_mode = SpinMode.as_spinmode(spin_mode)

        # We want an array bs_loband(nsppol).
        try:
            bs_loband = np.reshape(bs_loband, (spin_mode.nsppol))
        except ValueError:
            bs_loband = np.array(spin_mode.nsppol * [int(bs_loband)])

        self.bs_loband = bs_loband
        self.nband  = nband
        self.soenergy = soenergy
        self.coulomb_mode = coulomb_mode
        assert coulomb_mode in self._COULOMB_MODES
        self.ecuteps = ecuteps

        self.mdf_epsinf = mdf_epsinf
        self.exc_type = exc_type
        assert exc_type in self._EXC_TYPES
        self.algo = algo
        assert algo in self._ALGO2VAR
        self.with_lf = with_lf

        # if bs_freq_mesh is not given, abinit will select its own mesh.
        self.bs_freq_mesh = np.array(bs_freq_mesh) if bs_freq_mesh is not None else bs_freq_mesh
        self.zcut = zcut

        # Extra options.
        self.kwargs = kwargs
        if "chksymbreak" not in self.kwargs:
            self.kwargs["chksymbreak"] = 0

    @property
    def inclvkb(self):
        """Treatment of the dipole matrix element (NC pseudos, default is 2)"""
        return self.kwargs.get("inclvkb", 2)

    @property
    def use_haydock(self):
        """True if we are using the Haydock iterative technique."""
        return self.algo == "haydock"

    @property
    def use_cg(self):
        """True if we are using the conjugate gradient method."""
        return self.algo == "cg"

    @property
    def use_direct_diago(self):
        """True if we are performing the direct diagonalization of the BSE Hamiltonian."""
        return self.algo == "direct_diago"

    def to_abivars(self):
        """Returns a dictionary with the abinit variables."""
        abivars = dict(
            bs_calctype=1,
            bs_loband=self.bs_loband,
            nband=self.nband,
            soenergy=self.soenergy,
            ecuteps=self.ecuteps,
            #bs_algorithm = self._ALGO2VAR[self.algo],
            bs_coulomb_term=21,
            mdf_epsinf=self.mdf_epsinf,
            bs_exchange_term=1 if self.with_lf else 0,
            inclvkb=self.inclvkb,
            zcut=self.zcut,
            bs_freq_mesh=self.bs_freq_mesh,
            )

        if self.use_haydock:
            # FIXME
            abivars.update(
                bs_haydock_niter=100,      # No. of iterations for Haydock
                bs_hayd_term=0,            # No terminator
                bs_haydock_tol=[0.05, 0],  # Stopping criteria
            )

        elif self.use_direct_diago:
            raise NotImplementedError("")

        elif self.use_cg:
            raise NotImplementedError("")

        else:
            raise ValueError("Unknown algorithm for EXC: %s" % self.algo)

        # Add extra kwargs
        abivars.update(self.kwargs)

        return abivars
