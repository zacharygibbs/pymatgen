# coding: utf-8
"""
Classes defining Abinit calculations and workflows
"""
from __future__ import division, print_function, unicode_literals

import os
import time
import datetime
import shutil
import collections
import abc
import copy
import yaml
import six

from pprint import pprint
from monty.termcolor import colored
from six.moves import map, zip, StringIO
from monty.serialization import loadfn
from monty.string import is_string, list_strings
from monty.io import FileLock
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from pymatgen.util.string_utils import WildCard
from pymatgen.util.num_utils import maxloc
from pymatgen.serializers.json_coders import PMGSONable, json_pretty_dump, pmg_serialize
from .utils import File, Directory, irdvars_for_ext, abi_splitext, abi_extensions, FilepathFixer, Condition
from .netcdf import ETSF_Reader
from .strategies import StrategyWithInput, OpticInput
from . import abiinspect
from . import events 


try:
    from pydispatch import dispatcher
except ImportError:
    pass


__author__ = "Matteo Giantomassi"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Matteo Giantomassi"

__all__ = [
    "TaskManager",
    "ParalHintsParser",
    "ScfTask",
    "NscfTask",
    "RelaxTask",
    "DdkTask",
    "PhononTask",
    "SigmaTask",
    "OpticTask",
    "AnaddbTask",
]

import logging
logger = logging.getLogger(__name__)


# Tools and helper functions.

def straceback():
    """Returns a string with the traceback."""
    import traceback
    return traceback.format_exc()


class GridFsFile(AttrDict):
    def __init__(self, path, fs_id=None, mode="b"):
        super(GridFsFile, self).__init__(path=path, fs_id=fs_id, mode=mode)


class NodeResults(dict, PMGSONable):
    """
    Dictionary used to store the most important results produced by a Node.
    """
    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "node_id": {"type": "integer", "required": True},
            "node_finalized": {"type": "boolean", "required": True},
            "node_history": {"type": "array", "required": True},
            "node_class": {"type": "string", "required": True},
            "node_name": {"type": "string", "required": True},
            "node_status": {"type": "string", "required": True},
            "in": {"type": "object", "required": True, "description": "dictionary with input parameters"},
            "out": {"type": "object", "required": True, "description": "dictionary with the output results"},
            "exceptions": {"type": "array", "required": True},
            "files": {"type": "object", "required": True},
        },
    }

    @classmethod
    def from_node(cls, node):
        """Initialize an instance of `NodeResults` from a `Node` subclass."""
        kwargs = dict(
            node_id=node.node_id,
            node_finalized=node.finalized,
            node_history=list(node.history),
            node_name=node.name, 
            node_class=node.__class__.__name__,
            node_status=str(node.status),
        )

        return node.Results(node, **kwargs)

    def __init__(self, node, **kwargs):
        super(NodeResults, self).__init__(**kwargs)
        self.node = node

        if "in" not in self: self["in"] = Namespace()
        if "out" not in self: self["out"] = Namespace()
        if "exceptions" not in self: self["exceptions"] = []
        if "files" not in self: self["files"] = Namespace()

    @property
    def exceptions(self):
        return self["exceptions"]

    @property
    def gridfs_files(self):
        """List with the absolute paths of the files to be put in GridFs."""
        return self["files"]

    def add_gridfs_files(self, **kwargs):
        """
        This function registers the files that will be saved in GridFS.
        kwargs is a dictionary mapping the key associated to the file (usually the extension)
        to the absolute path. By default, files are assumed to be in binary form, for formatted files
        one should pass a tuple ("filepath", "t").

        Example::

            add_gridfs(GSR="path/to/GSR.nc", text_file=("/path/to/txt_file", "t"))

        The GSR file is a binary file, whereas text_file is a text file.
        """
        d = {}
        for k, v in kwargs.items():
            mode = "b" 
            if isinstance(v, (list, tuple)): v, mode = v
            d[k] = GridFsFile(path=v, mode=mode)
            
        self["files"].update(d)
        return self

    def push_exceptions(self, *exceptions):
        for exc in exceptions:
            newstr = str(exc)
            if newstr not in self.exceptions:
                self["exceptions"] += [newstr,]

    #def assert_valid(self):
    #    """
    #    Returns an empty list if results seem valid. 

    #    The try assert except trick allows one to get a string with info on the exception.
    #    We use the += operator so that sub-classes can add their own message.
    #    """
    #    # TODO Better treatment of events.
    #    try:
    #        assert (self["task_returncode"] == 0 and self["task_status"] == self.S_OK)
    #    except AssertionError as exc:
    #        self.push_exceptions(str(exc))
    #    return self.exceptions

    @pmg_serialize
    def as_dict(self):
        return self.copy()
                                                                                
    @classmethod
    def from_dict(cls, d):
        return cls({k: v for k, v in d.items() if k not in ("@module", "@class")})

    def json_dump(self, filename):
        json_pretty_dump(self.as_dict(), filename)

    @classmethod
    def json_load(cls, filename):
        return cls.from_dict(loadfn(filename))

    def validate_json_schema(self):
        import validictory
        d = self.as_dict()
        try:
            validictory.validate(d, self.JSON_SCHEMA)
            return True
        except ValueError as exc:
            pprint(d)
            print(exc)
            return False

    def update_collection(self, collection):
        """
        Update a mongodb collection.
        """
        node = self.node 
        flow = node if node.is_flow else node.flow

        # Build the key used to store the entry in the document.
        key = node.name
        if node.is_task:
            key = "w" + str(node.pos[0]) + "_t" + str(node.pos[1])
        elif node.is_workflow:
            key = "w" + str(node.pos)

        db = collection.database

        # Save files with GridFs first in order to get the ID.
        if self.gridfs_files:
            import gridfs
            fs = gridfs.GridFS(db)
            for ext, gridfile in self.gridfs_files.items():
                logger.info("gridfs: about to put file:", str(gridfile))
                # Here we set gridfile.fs_id that will be stored in the mondodb document
                try:
                    with open(gridfile.path, "r" + gridfile.mode) as f:
                        gridfile.fs_id = fs.put(f, filename=gridfile.path)
                except IOError as exc:
                    logger.critical(str(exc))

        if flow.mongo_id is None:
            # Flow does not have a mongo_id, allocate doc for the flow and save its id.
            flow.mongo_id = collection.insert({})
            print("Creating flow.mongo_id", flow.mongo_id, type(flow.mongo_id))

        # Get the document from flow.mongo_id and update it.
        doc = collection.find_one({"_id": flow.mongo_id})
        if key in doc:
            raise ValueError("%s is already in doc!" % key)
        doc[key] = self.as_dict()

        collection.save(doc)
        #collection.update({'_id':mongo_id}, {"$set": doc}, upsert=False)


class AbinitTaskResults(NodeResults):

    JSON_SCHEMA = NodeResults.JSON_SCHEMA.copy() 
    JSON_SCHEMA["properties"] = {
        "executable": {"type": "string", "required": True},
    }

    @classmethod
    def from_node(cls, task):
        """Initialize an instance from an AbinitTask instance."""
        new = super(AbinitTaskResults, cls).from_node(task)

        new.update(
            executable=task.executable,
            #executable_version:
            #task_events=
            pseudos=task.strategy.pseudos.as_dict()
            #input=task.strategy
        )

        new.add_gridfs_files(
            run_abi=(task.input_file.path, "t"),
            run_abo=(task.output_file.path, "t"),
        )

        return new


class ParalConf(AttrDict):
    """
    This object store the parameters associated to one 
    of the possible parallel configurations reported by ABINIT.
    Essentially it is a dictionary whose values can also be accessed 
    as attributes. It also provides default values for selected keys
    that might not be present in the ABINIT dictionary.

    Example:

        --- !Autoparal
        info: 
            version: 1
            autoparal: 1
            max_ncpus: 108
        configurations:
            -   tot_ncpus: 2         # Total number of CPUs
                mpi_ncpus: 2         # Number of MPI processes.
                omp_ncpus: 1         # Number of OMP threads (1 if not present)
                mem_per_cpu: 10      # Estimated memory requirement per MPI processor in Megabytes.
                efficiency: 0.4      # 1.0 corresponds to an "expected" optimal efficiency (strong scaling).
                vars: {              # Dictionary with the variables that should be added to the input.
                      varname1: varvalue1
                      varname2: varvalue2
                      }
            -
        ...

    For paral_kgb we have:
    nproc     npkpt  npspinor    npband     npfft    bandpp    weight   
       108       1         1        12         9         2        0.25
       108       1         1       108         1         2       27.00
        96       1         1        24         4         1        1.50
        84       1         1        12         7         2        0.25
    """
    _DEFAULTS = {
        "omp_ncpus": 1,     
        "mem_per_cpu": 0.0, 
        "vars": {}       
    }

    def __init__(self, *args, **kwargs):
        super(ParalConf, self).__init__(*args, **kwargs)
        
        # Add default values if not already in self.
        for k, v in self._DEFAULTS.items():
            if k not in self:
                self[k] = v

    def __str__(self):
        stream = StringIO()
        pprint(self, stream=stream)
        return stream.getvalue()

    # TODO: Change name in abinit
    @property
    def tot_cores(self):
        return self.tot_ncpus

    @property
    def mem_per_proc(self):
        return self.mem_per_cpu

    @property
    def mpi_procs(self):
        return self.mpi_ncpus

    @property
    def omp_threads(self):
        return self.omp_ncpus

    @property
    def speedup(self):
        """Estimated speedup reported by ABINIT."""
        return self.efficiency * self.tot_cores

    @property
    def tot_mem(self):
        """Estimated total memory in Mbs (computed from mem_per_cpu)"""
        return self.mem_per_proc * self.mpi_procs


class ParalHintsError(Exception):
    """Base error class for `ParalHints`."""


class ParalHintsParser(object):

    Error = ParalHintsError

    def __init__(self):
        # Used to push error strings.
        self._errors = collections.deque(maxlen=100)

    def parse(self, filename):
        """
        Read the AutoParal section (YAML format) from filename.
        Assumes the file contains only one section.
        """
        with abiinspect.YamlTokenizer(filename) as r:
            doc = r.next_doc_with_tag("!Autoparal")
            try:
                d = yaml.load(doc.text_notag)
                return ParalHints(info=d["info"], confs=d["configurations"])
            except:
                import traceback
                sexc = traceback.format_exc()
                err_msg = "Wrong YAML doc:\n%s\n\nException" % (doc.text, sexc)
                self._errors.append(err_msg)
                logger.critical(err_msg)
                raise self.Error(err_msg)


class ParalHints(collections.Iterable):
    """
    Iterable with the hints for the parallel execution reported by ABINIT.
    """
    Error = ParalHintsError

    def __init__(self, info, confs):
        self.info = info
        self._confs = [ParalConf(**d) for d in confs]

    def __getitem__(self, key):
        return self._confs[key]

    def __iter__(self):
        return self._confs.__iter__()

    def __len__(self):
        return self._confs.__len__()

    def __str__(self):
        return "\n".join(str(conf) for conf in self)

    @pmg_serialize
    def as_dict(self):
        return {"info": self.info, "confs": self._confs}

    @classmethod
    def from_dict(cls, d):
        return cls(info=d["info"], confs=d["confs"])

    def copy(self):
        """Shallow copy of self."""
        return copy.copy(self)

    def select_with_condition(self, condition, key=None):
        """
        Remove all the configurations that do not satisfy the given condition.

            Args:
                `Condition` object with operators expressed with a Mongodb-like syntax
            key:
                Selects the sub-dictionary on which condition is applied, e.g. key="vars"
                if we have to filter the configurations depending on the values in vars
        """
        new_confs = []

        for conf in self:
            # Select the object on which condition is applied
            obj = conf if key is None else AttrDict(conf[key])
            add_it = condition(obj=obj)
            #if key is "vars": print("conf", conf, "added:", add_it)
            if add_it:
                new_confs.append(conf)

        self._confs = new_confs

    def sort_by_efficiency(self, reverse=False):
        """
        Sort the configurations in place so that conf with lowest efficieny 
        appears in the first positions.
        """
        self._confs.sort(key=lambda c: c.efficiency, reverse=reverse)

    def sort_by_speedup(self, reverse=False):
        """
        Sort the configurations in place so that conf with lowest speedup 
        appears in the first positions.
        """
        self._confs.sort(key=lambda c: c.speedup, reverse=reverse)

    def sort_by_mem_per_core(self, reverse=False):
        """
        Sort the configurations in place so that conf with lowest memory per core
        appears in the first positions.
        """
        # Avoid sorting if mem_per_cpu is not available.
        if any(c.mem_per_cpu > 0.0 for c in self):
            self._confs.sort(key=lambda c: c.mem_per_cpu, reverse=reverse)

    def select_optimal_conf(self, policy):
        """
        Find the optimal configuration according to the `TaskPolicy` policy.
        """
        # Make a copy since we are gonna change the object in place.
        #hints = self.copy()

        hints = ParalHints(self.info, confs=[c for c in self if c.tot_cores <= policy.max_ncpus])
        #logger.info('hints: \n' + str(hints) + '\n')

        # First select the configurations satisfying the condition specified by the user (if any)
        if policy.condition:
            #logger.info("condition %s" % str(policy.condition))
            hints.select_with_condition(policy.condition)
            #logger.info("after condition %s" % str(hints))

            # If no configuration fullfills the requirements, 
            # we return the one with the highest speedup.
            if not hints:
                logger.warning("empty list of configurations after policy.condition")
                hints = self.copy()
                hints.sort_by_speedup()
                return hints[-1].copy()

        # Now filter the configurations depending on the values in vars
        if policy.vars_condition:
            logger.info("vars_condition %s" % str(policy.vars_condition))
            hints.select_with_condition(policy.vars_condition, key="vars")
            logger.info("After vars_condition %s" % str(hints))

            # If no configuration fullfills the requirements,
            # we return the one with the highest speedup.
            if not hints:
                logger.warning("empty list of configurations after policy.vars_condition")
                hints = self.copy()
                hints.sort_by_speedup()
                return hints[-1].copy()

        hints.sort_by_speedup()

        logger.info('speedup hints: \n' + str(hints) + '\n')

        #hints.sort_by_efficiency()

        #logger.info('efficiency hints: \n' + str(hints) + '\n')

        # Find the optimal configuration according to policy.mode.
        #if policy.mode in ["default", "aggressive"]:
        #    hints.sort_by_spedup()
        #elif policy.mode == "conservative":
        #    hints.sort_by_efficiency()
        #    # Remove tot_cores == 1
        #    hints.pop(tot_cores==1)
        #else:
        #    raise ValueError("Wrong value for policy.mode: %s" % str(policy.mode))
        #if not hints:

        # Return a copy of the configuration.
        optimal = hints[-1].copy()
        logger.info("Will relaunch the job with optimized parameters:\n %s" % optimal)

        return optimal


class TaskPolicy(object):
    """
    This object stores the parameters used by the `TaskManager` to 
    create the submission script and/or to modify the ABINIT variables 
    governing the parallel execution. A `TaskPolicy` object contains 
    a set of variables that specify the launcher, as well as the options
    and the condition used to select the optimal configuration for the parallel run 
    """
    @classmethod
    def as_policy(cls, obj):
        """
        Converts an object obj into a TaskPolicy. Accepts:

            * None
            * TaskPolicy
            * dict-like object
        """
        if obj is None:
            # Use default policy.
            return TaskPolicy()
        else:
            if isinstance(obj, cls):
                return obj
            elif isinstance(obj, collections.Mapping):
                return cls(**obj) 
            else:
                raise TypeError("Don't know how to convert type %s to %s" % (type(obj), cls))

    def __init__(self, autoparal=0, automemory=0, mode="default", max_ncpus=None,
                 condition=None, vars_condition=None):
        """
        Args:
            autoparal: 
                Value of ABINIT autoparal input variable. None to disable the autoparal feature.
            automemory:
                int defining the memory policy. 
                If > 0 the memory requirements will be computed at run-time from the autoparal section
                produced by ABINIT. In this case, the job script will report the autoparal memory
                instead of the one specified by the user.
            mode:
                Select the algorith to select the optimal configuration for the parallel execution.
                Possible values: ["default", "aggressive", "conservative"]
            max_ncpus:
                Maximal number of phyiscal CPUs that can be used (must be specified if autoparal > 0).
            condition:
                condition used to filter the autoparal configuration (Mongodb-like syntax)
            vars_condition:
                condition used to filter the list of Abinit variables suggested by autoparal (Mongodb-like syntax)
        """
        self.autoparal = autoparal
        self.automemory = automemory
        self.mode = mode 
        self.max_ncpus = max_ncpus
        self.condition = Condition(condition) if condition is not None else condition
        self.vars_condition = Condition(vars_condition) if vars_condition is not None else vars_condition
        self._LIMITS = {'max_ncpus': 240}

        if self.autoparal and self.max_ncpus is None:
            raise ValueError("When autoparal is not zero, max_ncpus must be specified.")

    def __str__(self):
        lines = []
        app = lines.append
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            app("%s: %s" % (k, v))
        return "\n".join(lines)

    def increase_max_ncpus(self):
        base_increase = 12
        new = self.max_ncpus + base_increase
        if new <= 360:
            logger.info('set max_ncps to '+str(new))
            self.max_ncpus = new
            return True
        else:
            return False


class TaskManager(object):
    """
    A `TaskManager` is responsible for the generation of the job script and the submission 
    of the task, as well as for the specification of the parameters passed to the resource manager
    (e.g. Slurm, PBS ...) and/or the run-time specification of the ABINIT variables governing the 
    parallel execution. A `TaskManager` delegates the generation of the submission
    script and the submission of the task to the `QueueAdapter`. 
    A `TaskManager` has a `TaskPolicy` that governs the specification of the parameters for the parallel executions.
    Ideally, the TaskManager should be the **main entry point** used by the task to deal with job submission/optimization
    """
    YAML_FILE = "taskmanager.yml"
    USER_CONFIG_DIR = os.path.join(os.getenv("HOME"), ".abinit", "abipy")

    @classmethod
    def from_dict(cls, d):
        """Create an instance from dictionary d."""
        return cls(**d)
                                                                              
    @classmethod
    def from_string(cls, s):
        """Create an instance from string s containing a YAML dictionary."""
        stream = StringIO(s)
        stream.seek(0)
        return cls.from_dict(yaml.load(stream))

    @classmethod
    def from_file(cls, filename):
        """Read the configuration parameters from the Yaml file filename."""
        with open(filename, "r") as fh:
            return cls.from_dict(yaml.load(fh))

    @classmethod
    def from_user_config(cls):
        """
        Initialize the `TaskManager` from the YAML file 'taskmanager.yaml'.
        Search first in the working directory and then in the configuration directory of abipy.

        Raises:
            RuntimeError if file is not found.
        """
        # Try in the current directory.
        path = os.path.join(os.getcwd(), cls.YAML_FILE)
        if os.path.exists(path):
            return cls.from_file(path)

        # Try in the configuration directory.
        path = os.path.join(cls.USER_CONFIG_DIR, cls.YAML_FILE)
        if os.path.exists(path):
            return cls.from_file(path)
    
        raise RuntimeError("Cannot locate %s neither in current directory nor in %s" % (cls.YAML_FILE, path))

    @classmethod 
    def sequential(cls):
        """
        Build a simple `TaskManager` that submits jobs via a simple shell script.
        Assume the shell environment has been already initialized.
        """
        return cls(qtype="shell")

    @classmethod 
    def simple_mpi(cls, mpi_runner="mpirun", mpi_procs=1, policy=None):
        """
        Build a `TaskManager` that submits jobs with a simple shell script and mpirun.
        Assume the shell environment is already properly initialized.
        """
        return cls(qtype="shell", qparams=dict(MPI_PROCS=mpi_procs), mpi_runner=mpi_runner, policy=policy)

    def __init__(self, qtype, qparams=None, setup=None, modules=None, shell_env=None, omp_env=None, 
                 pre_run=None, post_run=None, mpi_runner=None, policy=None, partitions=None, db_connector=None):

        from .qadapters import qadapter_class, Partition
        qad_class = qadapter_class(qtype)
        self.qadapter = qad_class(qparams=qparams, setup=setup, modules=modules, shell_env=shell_env, omp_env=omp_env, 
                                  pre_run=pre_run, post_run=post_run, mpi_runner=mpi_runner)

        self.policy = TaskPolicy.as_policy(policy)

        # Initialize the partitions:
        # order them according to priority and make sure that each partition has different priority
        self.parts = []
        if partitions is not None:
            if not isinstance(partitions, (list, tuple)): partitions = [partitions]
            self.parts = sorted([Partition(**part) for part in partitions], key=lambda p: p.priority)

        priorities = [p.priority for p in self.parts]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Two or more partitions have same priority. This is not allowed. Check taskmanager.yml")

        # Initialize database connector (if specified)
        from .db import DBConnector
        self.db_connector = DBConnector(config_dict=db_connector)

    def __str__(self):
        """String representation."""
        lines = []
        app = lines.append
        #app("tot_cores %d, mpi_procs %d, omp_threads %s" % (self.tot_cores, self.mpi_procs, self.omp_threads))
        app("[Partitions #%d]\n" % len(self.parts))
        lines.extend(p for p in self.parts)
        app("[Qadapter]\n%s" % str(self.qadapter))
        app("[Task policy]\n%s" % str(self.policy))

        if self.has_db:
            app("[MongoDB database]:")
            app(str(self.db_connector))

        return "\n".join(lines)

    @property
    def has_db(self):
        """True if we are using MongoDB database"""
        return bool(self.db_connector)

    @property
    def has_omp(self):
        """True if we are using OpenMP parallelization."""
        return self.qadapter.has_omp

    @property
    def tot_cores(self):
        """Total number of CPUs used to run the task."""
        return self.qadapter.tot_cores

    @property
    def mpi_procs(self):
        """Number of MPI processes."""
        return self.qadapter.mpi_procs

    @property
    def omp_threads(self):
        """Number of OpenMP threads"""
        return self.qadapter.omp_threads

    def get_collection(self, **kwargs):
        """Return the MongoDB collection used to store the results."""
        return self.db_connector.get_collection(**kwargs)

    def to_shell_manager(self, mpi_procs=1, policy=None):
        """
        Returns a new `TaskManager` with the same parameters as self but replace the `QueueAdapter` 
        with a `ShellAdapter` with mpi_procs so that we can submit the job without passing through the queue.
        Replace self.policy with a `TaskPolicy` with autoparal==0.
        """
        qad = self.qadapter.deepcopy()

        policy = TaskPolicy(autoparal=0) if policy is None else policy

        cls = self.__class__
        new = cls("shell", qparams={"MPI_PROCS": mpi_procs}, setup=qad.setup, modules=qad.modules, 
                  shell_env=qad.shell_env, omp_env=None, pre_run=qad.pre_run, 
                  post_run=qad.post_run, mpi_runner=qad.mpi_runner, policy=policy)

        return new

    def new_with_policy(self, policy):
        """
        Returns a new `TaskManager` with same parameters as self except for policy.
        """
        new = self.deepcopy()
        new.policy = policy
        return new

    #def copy(self):
    #    """Shallow copy of self."""
    #    return copy.copy(self)

    def deepcopy(self):
        """Deep copy of self."""
        return copy.deepcopy(self)

    def set_mpi_procs(self, mpi_procs):
        """Set the number of MPI nodes to use."""
        self.qadapter.set_mpi_procs(mpi_procs)

    def set_omp_threads(self, omp_threads):
        """Set the number of OpenMp threads to use."""
        self.qadapter.set_omp_threads(omp_threads)

    def set_mem_per_cpu(self, mem_mb):
        """Set the memory (in Megabytes) per CPU."""
        self.qadapter.set_mem_per_cpu(mem_mb)

    def set_autoparal(self, value):
        """Set the value of autoparal."""
        assert value in [0, 1]
        self.policy.autoparal = value

    def set_max_ncpus(self, value):
        """Set the value of max_ncpus."""
        self.policy.max_ncpus = value

    #@property
    #def max_ncpus(self):
    #    return max(p.max_ncores for p in self.partitions)

    def get_njobs_in_queue(self, username=None):
        """
        returns the number of jobs in the queue,
        returns None when the number of jobs cannot be determined.

        Args:
            username: (str) the username of the jobs to count (default is to autodetect)
        """
        return self.qadapter.get_njobs_in_queue(username=username)

    @property
    def active_partition(self):
        return None
        try:
            return self._active_partition
        except AttributeError:
            return self.parts[0]

    def select_partition(self, pconf):
        """
        Select a partition to run the parallel configuration pconf
        Set self.active_partition. Return None if no partition could be found.
        """
        #self._selected_partition = None
        #return None
        # TODO
        scores = [part.get_score(pconf) for part in self.parts]
        if all(sc < 0 for sc in scores): return None
        self._active_partition = self.parts[maxloc(scores)]
        return self._active_partition

    def cancel(self, job_id):
        """Cancel the job. Returns exit status."""
        return self.qadapter.cancel(job_id)

    def write_jobfile(self, task):
        """Write the submission script. return the path of the script"""
        script = self.qadapter.get_script_str(
            job_name=task.name, 
            launch_dir=task.workdir,
            partition=self.active_partition,
            executable=task.executable,
            qout_path=task.qout_file.path,
            qerr_path=task.qerr_file.path,
            stdin=task.files_file.path, 
            stdout=task.log_file.path,
            stderr=task.stderr_file.path,
        )

        # Write the script.
        with open(task.job_file.path, "w") as fh:
            fh.write(script)
            return task.job_file.path

    def launch(self, task):
        """
        Build the input files and submit the task via the `Qadapter` 

        Args:
            task:
                `TaskObject`
        
        Returns:
            Process object.
        """
        # Build the task 
        task.build()

        # Submit the task and save the queue id.
        # FIXME: CD to script file dir?
        task.set_status(task.S_SUB)
        script_file = self.write_jobfile(task)
        process, queue_id = self.qadapter.submit_to_queue(script_file)
        task.set_queue_id(queue_id)

        return process

    def increase_resources(self):
            # with GW calculations in mind with GW mem = 10, the response fuction is in memory and not distributed
            # we need to increas memory if jobs fail ...
        return self.qadapter.increase_mem()

#        if self.policy.autoparal == 1:
#            #if self.policy.increase_max_ncpus():
#                return True
#            else:
#                return False
#        elif self.qadapter is not None:
#            if self.qadapter.increase_cpus():
#                return True
#            else:
#                return False
#        else:
#            return False


# The code below initializes a counter from a file when the module is imported 
# and save the counter's updated value automatically when the program terminates 
# without relying on the application making an explicit call into this module at termination.
conf_dir = os.path.join(os.getenv("HOME"), ".abinit", "abipy")

if not os.path.exists(conf_dir):
    os.makedirs(conf_dir)

_COUNTER_FILE = os.path.join(conf_dir, "nodecounter")
del conf_dir

try:
    with open(_COUNTER_FILE, "r") as _fh:
        _COUNTER = int(_fh.read())

except IOError:
    _COUNTER = -1


def get_newnode_id():
    """
    Returns a new node identifier used both for `Task` and `Workflow` objects.

    .. warnings:
        The id is unique inside the same python process so be careful when 
        Workflows and Task are constructed at run-time or when threads are used.
    """
    global _COUNTER
    _COUNTER += 1
    return _COUNTER


def save_lastnode_id():
    """Save the id of the last node created."""
    with FileLock(_COUNTER_FILE) as lock:
        with open(_COUNTER_FILE, "w") as fh:
            fh.write("%d" % _COUNTER)

import atexit
atexit.register(save_lastnode_id)


class FakeProcess(object):
    """
    This object is attached to a Task instance if the task has not been submitted
    This trick allows us to simulate a process that is still running so that 
    we can safely poll task.process.
    """
    def poll(self):
        return None

    def wait(self):
        raise RuntimeError("Cannot wait a FakeProcess")

    def communicate(self, input=None):
        raise RuntimeError("Cannot communicate with a FakeProcess")

    def kill(self):
        raise RuntimeError("Cannot kill a FakeProcess")

    @property
    def returncode(self):
        return None


class Product(object):
    """
    A product represents an output file produced by ABINIT instance.
    This file is needed to start another `Task` or another `Workflow`.
    """
    def __init__(self, ext, path):
        """
        Args:
            ext:
                ABINIT file extension
            path:
                (asbolute) filepath
        """
        if ext not in abi_extensions():
            raise ValueError("Extension %s has not been registered in the internal database" % str(ext))

        self.ext = ext
        self.file = File(path)

    @classmethod
    def from_file(cls, filepath):
        """Build a `Product` instance from a filepath."""
        # Find the abinit extension.
        for i in range(len(filepath)):
            if filepath[i:] in abi_extensions():
                ext = filepath[i:]
                break
        else:
            raise ValueError("Cannot detect abinit extension in %s" % filepath)
        
        return cls(ext, filepath)

    def __str__(self):
        return "File=%s, Extension=%s, " % (self.file.path, self.ext)

    @property
    def filepath(self):
        """Absolute path of the file."""
        return self.file.path

    def connecting_vars(self):
        """
        Returns a dictionary with the ABINIT variables that 
        must be used to make the code use this file.
        """
        return irdvars_for_ext(self.ext)


class Dependency(object):
    """
    This object describes the dependencies among the nodes of a calculation.

    A `Dependency` consists of a `Node` that produces a list of products (files) 
    that are used by the other nodes (`Task` or `Workflow`) to start the calculation.
    One usually creates the object by calling work.register 

    Example:

        # Register the SCF task in work.
        scf_task = work.register(scf_strategy)

        # Register the NSCF calculation and its dependency on the SCF run via deps.
        nscf_task = work.register(nscf_strategy, deps={scf_task: "DEN"})
    """
    def __init__(self, node, exts=None):
        """
        Args:
            node:
                The task or the worfklow associated to the dependency.
            exts:
                Extensions of the output files that are needed for running the other tasks.
        """
        self._node = node

        if exts and is_string(exts):
            exts = exts.split()

        self.exts = exts or []

    def __hash__(self):
        return hash(self._node)

    def __repr__(self):
        return "Node %s will produce: %s " % (repr(self.node), repr(self.exts))

    def __str__(self):
        return "Node %s will produce: %s " % (str(self.node), str(self.exts))

    @property
    def info(self):
        return str(self.node)

    @property
    def node(self):
        """The node associated to the dependency."""
        return self._node

    @property
    def status(self):
        """The status of the dependency, i.e. the status of the node."""
        return self.node.status

    @lazy_property
    def products(self):
        """List of output files produces by self."""
        _products = []
        for ext in self.exts:
            prod = Product(ext, self.node.opath_from_ext(ext))
            _products.append(prod)

        return _products

    def connecting_vars(self):
        """
        Returns a dictionary with the variables that must be added to the 
        input file in order to connect this `Node` to its dependencies.
        """
        vars = {}
        for prod in self.products:
            vars.update(prod.connecting_vars())

        return vars

    def get_filepaths_and_exts(self):
        """Returns the paths of the output files produced by self and its extensions"""
        filepaths = [prod.filepath for prod in self.products]
        exts = [prod.ext for prod in self.products]

        return filepaths, exts

def _2attrs(item):
        return item if item is None or isinstance(list, tuple) else (item,)

class Status(int):
    """This object is an integer representing the status of the `Node`."""

    # Possible status of the node. See monty.termocolor for the meaning of color, on_color and attrs.
    _STATUS_INFO = [
        #(value, name, color, on_color, attrs)
        (1,  "Initialized",   None     , None, None),         # Node has been initialized
        (2,  "Locked",        None     , None, None),         # Task is locked an must be explicitly unlocked by an external subject (Workflow).
        (3,  "Ready",         None     , None, None),         # Node is ready i.e. all the depencies of the node have status S_OK
        (4,  "Submitted",     "blue"   , None, None),         # Node has been submitted (The `Task` is running or we have started to finalize the Workflow)
        (5,  "Running",       "magenta", None, None),         # Node is running.
        (6,  "Done",          None     , None, None),         # Node done, This does not imply that results are ok or that the calculation completed successfully
        (7,  "AbiCritical",   "red"    , None, None),         # Node raised an Error by ABINIT.
        (8,  "QueueCritical", "red"    , "on_white", None),   # Node raised an Error by submitting submission script, or by executing it
        (9,  "Unconverged",   "red"    , "on_yellow", None),  # This usually means that an iterative algorithm didn't converge.
        (10, "Error",         "red"    , None, None),         # Node raised an unrecoverable error, usually raised when an attempt to fix one of other types failed.
        (11, "Completed",     "green"  , None, None),         # Execution completed successfully.
        #(11, "Completed",     "green"  , None, "underline"),   
    ]
    _STATUS2STR = collections.OrderedDict([(t[0], t[1]) for t in _STATUS_INFO])
    _STATUS2COLOR_OPTS = collections.OrderedDict([(t[0], {"color": t[2], "on_color": t[3], "attrs": _2attrs(t[4])}) for t in _STATUS_INFO])

    def __repr__(self):
        return "<%s: %s, at %s>" % (self.__class__.__name__, str(self), id(self))

    def __str__(self):
        """String representation."""
        return self._STATUS2STR[self]

    @classmethod
    def as_status(cls, obj):
        """Convert obj into Status."""
        if isinstance(obj, cls):
            return obj
        else:
            # Assume string
            return cls.from_string(obj)

    @classmethod
    def from_string(cls, s):
        """Return a `Status` instance from its string representation."""
        for num, text in cls._STATUS2STR.items():
            if text == s:
                return cls(num)
        else:
            raise ValueError("Wrong string %s" % s)

    @property
    def is_critical(self):
        """True if status is critical."""
        return str(self) in ("AbiCritical", "QueueCritical", "Uncoverged", "Error") 

    @property
    def colored(self):
        """Return colorized text used to print the status if the stream supports it."""
        return colored(str(self), **self._STATUS2COLOR_OPTS[self]) 


class Node(six.with_metaclass(abc.ABCMeta, object)):
    """
    Abstract base class defining the interface that must be 
    implemented by the nodes of the calculation.

    Nodes are hashable and can be tested for equality
    (hash uses the node identifier, whereas eq uses workdir).
    """
    Results = NodeResults

    # Possible status of the node.
    S_INIT = Status.from_string("Initialized")
    S_LOCKED = Status.from_string("Locked")
    S_READY = Status.from_string("Ready")
    S_SUB = Status.from_string("Submitted")
    S_RUN = Status.from_string("Running")
    S_DONE = Status.from_string("Done")
    S_ABICRITICAL = Status.from_string("AbiCritical")
    S_QUEUECRITICAL = Status.from_string("QueueCritical")
    S_UNCONVERGED = Status.from_string("Unconverged")
    S_ERROR = Status.from_string("Error")
    S_OK = Status.from_string("Completed")

    ALL_STATUS = [
        S_INIT,
        S_LOCKED,
        S_READY,
        S_SUB,
        S_RUN,
        S_DONE,
        S_ABICRITICAL,
        S_QUEUECRITICAL,
        S_UNCONVERGED,
        S_ERROR,
        S_OK,
    ]

    def __init__(self):
        # Node identifier.
        self._node_id = get_newnode_id()

        # List of dependencies
        self._deps = []

        # List of files (products) needed by this node.
        self._required_files = []

        # Used to push additional info during the execution. 
        self.history = collections.deque(maxlen=100)

        # Set to true if the node has been finalized.
        self._finalized = False

        self._status = self.S_INIT

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        #return self.node_id == other.node_id and 
        return (self.__class__ == other.__class__ and 
                self.workdir == other.workdir)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.node_id)

    def __repr__(self):
        try:
            return "<%s, node_id %s, workdir=%s>" % (
                self.__class__.__name__, self.node_id, os.path.relpath(self.workdir))

        except AttributeError:
            # this usually happens when workdir has not been initialized
            return "<%s, node_id %s, workdir=None>" % (self.__class__.__name__, self.node_id)
                                                                                            
    def __str__(self):
        try:
            return "<%s, workdir=%s>" % (self.__class__.__name__, os.path.relpath(self.workdir))
        except AttributeError:
            # this usually happens when workdir has not been initialized
            return "<%s, workdir=None>" % self.__class__.__name__

    @classmethod
    def as_node(cls, obj):
        """
        Convert obj into a Node instance.

        Return:
            obj if obj is a Node instance,
            cast obj to `FileNode` instance of obj is a string.
            None if obj is None
        """
        if isinstance(obj, cls):
            return obj
        elif is_string(obj):
            # Assume filepath.
            return FileNode(obj)
        elif obj is None:
            return obj
        else:
            raise TypeError("Don't know how to convert %s to Node instance." % obj)

    @property
    def name(self):
        """
        The name of the node 
        (only used for facilitating its identification in the user interface).
        """
        try:
            return self._name
        except AttributeError:
            return os.path.relpath(self.workdir)

    def set_name(self, name):
        """Set the name of the Node."""
        self._name = name

    @property
    def node_id(self):
        """Node identifier."""
        return self._node_id
                                                         
    def set_node_id(self, node_id):
        """Set the node identifier. Use it carefully!"""
        self._node_id = node_id

    @property
    def finalized(self):
        """True if the `Workflow` has been finalized."""
        return self._finalized

    @finalized.setter
    def finalized(self, boolean):
        self._finalized = boolean
        self.history.append("Finalized on %s" % time.asctime())

    @property
    def str_history(self):
        """String representation of history."""
        return "\n".join(self.history)

    @property
    def is_file(self):
        """True if this node is a file"""
        return isinstance(self, FileNode)

    @property
    def is_task(self):
        """True if this node is a Task"""
        return isinstance(self, Task)

    @property
    def is_workflow(self):
        """True if this node is a Workflow"""
        from .workflows import Workflow
        return isinstance(self, Workflow)

    @property
    def is_flow(self):
        """True if this node is a Flow"""
        from .flows import AbinitFlow
        return isinstance(self, AbinitFlow)

    @property
    def has_subnodes(self):
        """True if self contains sub-nodes e.g. `Workflow` object."""
        return isinstance(self, collections.Iterable)

    @property
    def deps(self):
        """
        List of `Dependency` objects defining the dependencies 
        of this `Node`. Empty list if this `Node` does not have dependencies.
        """
        return self._deps

    def add_deps(self, deps):
        """
        Add a list of dependencies to the `Node`.

        Args:
            deps:
                List of `Dependency` objects specifying the 
                dependencies of the node.
        """
        # We want a list
        if not isinstance(deps, (list, tuple)):
            deps = [deps]

        assert all(isinstance(d, Dependency) for d in deps)

        # Add the dependencies to the node
        self._deps.extend(deps)

        if self.has_subnodes:
            # This means that the node contains sub-nodes 
            # that should inherit the same dependency.
            for task in self:
                task.add_deps(deps)

    def remove_deps(self, deps):
        """
        Remove a list of dependencies from the `Node`.

        Args:
            deps:
                List of `Dependency` objects specifying the 
                dependencies of the node.
        """
        if not isinstance(deps, (list, tuple)):
            deps = [deps]
                                                                                      
        assert all(isinstance(d, Dependency) for d in deps)

        self._deps = [d for d in self._deps if d not in deps]
                                                                                      
        if self.has_subnodes:
            # This means that the node consists of sub-nodes 
            # that should remove the same list of dependencies.
            for task in self:
                task.remove_deps(deps)                                                                                                                                        

    @property
    def deps_status(self):
        """Returns a list with the status of the dependencies."""
        if not self.deps:
            return [self.S_OK]
                                                                  
        return [d.status for d in self.deps]

    def depends_on(self, other):
        """True if this node depends on the other node."""
        return other in [d.node for d in self.deps]

    def str_deps(self):
        """Return the string representation of the dependecies of the node."""
        lines = []
        app = lines.append

        app("Dependencies of node %s:" % str(self))
        for i, dep in enumerate(self.deps):
            app("%d) %s, status=%s" % (i, dep.info, str(dep.status)))

        return "\n".join(lines)

    @property
    def required_files(self):
        """
        List of `Product` objects with info on the files needed by the `Node`.
        """
        return self._required_files

    def add_required_files(self, files):
        """
        Add a list of paths to the list of files required by the `Node`.
        Note that the files must exist when the task is registered.

        Args:
            files:
                string or list of strings with the path of the files
        Raises:
            ValueError if at least one file does not exist.
        """
        # We want a list of absolute paths.
        files = map(os.path.abspath, list_strings(files))

        # Files must exist.
        if any(not os.path.exists(f) for f in files):
            err_msg = ("Cannot define a dependency on a file that does not exist!\n" + 
                       "The following files do not exist:\n" +
                       "\n".join(["\t" + f for f in files if not os.path.exists(f)]))
            raise ValueError(err_msg)

        # Convert to list of products.
        files = [Product.from_file(path) for path in files]

        # Add the dependencies to the node.
        self._required_files.extend(files)

    #@abc.abstractmethod
    #def set_status(self, status,  info_msg=None):
    #    """
    #    Set and return the status of the None
    #                                                                                     
    #    Args:
    #        status:
    #            Status object or string representation of the status
    #        info_msg:
    #            string with human-readable message used in the case of errors (optional)
    #    """

    @abc.abstractproperty
    def status(self):
        """The status of the `Node`."""

    @abc.abstractmethod
    def check_status(self):
        """Check the status of the `Node`."""


class FileNode(Node):
    """
    A Node that consists of an already existing file.

    Mainly used to connect Tasks to external files produced in previous runs
    """
    def __init__(self, filename):
        super(FileNode, self).__init__()
        self.filepath = os.path.abspath(filename)
        if not os.path.exists(self.filepath):
            raise ValueError("File %s must exists" % self.filepath)

        # Directories with input|output|temporary data.
        self.workdir = os.path.dirname(self.filepath)

        self.indir = Directory(self.workdir)
        self.outdir = Directory(self.workdir)
        self.tmpdir = Directory(self.workdir)

    @property
    def products(self):
        return [Product.from_file(self.filepath)]

    def opath_from_ext(self, ext):
        return self.filepath

    @property
    def status(self):
        return self.S_OK if os.path.exists(self.filepath) else self.S_ERROR

    def check_status(self):
        return self.status

    def get_results(self, **kwargs):
        results = super(FileNode, self).get_results(**kwargs)
        #results.add_gridfs_files(self.filepath=self.filepath)
        return results


class TaskError(Exception):
    """Base Exception for `Task` methods"""


class TaskRestartError(TaskError):
    """Exception raised while trying to restart the `Task`."""


class Task(six.with_metaclass(abc.ABCMeta, Node)):
    """A Task is a node that performs some kind of calculation."""
    # Use class attributes for TaskErrors so that we don't have to import them.
    Error = TaskError
    RestartError = TaskRestartError

    # List of `AbinitEvent` subclasses that are tested in the not_converged method. 
    # Subclasses should provide their own list if they need to check the converge status.
    CRITICAL_EVENTS = [
    ]

    # Prefixes for Abinit (input, output, temporary) files.
    Prefix = collections.namedtuple("Prefix", "idata odata tdata")
    pj = os.path.join

    prefix = Prefix(pj("indata", "in"), pj("outdata", "out"), pj("tmpdata", "tmp"))
    del Prefix, pj

    def __init__(self, strategy, workdir=None, manager=None, deps=None, required_files=None):
        """
        Args:
            strategy: 
                Input file or `Strategy` instance defining the calculation.
            workdir:
                Path to the working directory.
            manager:
                `TaskManager` object.
            deps:
                Dictionary specifying the dependency of this node.
                None means that this obj has no dependency.
            required_files:
                List of strings with the path of the files used by the task.
        """
        # Init the node
        super(Task, self).__init__()

        # Save the strategy to use to generate the input file.
        # FIXME
        #self.strategy = strategy.deepcopy()
        self.strategy = strategy
                                                               
        if workdir is not None:
            self.set_workdir(workdir)
                                                               
        if manager is not None:
            self.set_manager(manager)

        # Handle possible dependencies.
        if deps:
            deps = [Dependency(node, exts) for (node, exts) in deps.items()]
            self.add_deps(deps)

        if required_files:
            self.add_required_files(required_files)

        # Use to compute the wall-time
        self.start_datetime, self.stop_datetime = None, None

        # Number of restarts effectuated.
        self.num_restarts = 0

        self.queue_errors = []
        self.abi_errors = []

    def __getstate__(self):
        """
        Return state is pickled as the contents for the instance.
                                                                                      
        In this case we just remove the process since Subprocess objects cannot be pickled.
        This is the reason why we have to store the returncode in self._returncode instead
        of using self.process.returncode.
        """
        return {k: v for k, v in self.__dict__.items() if k not in ["_process"]}

    def set_workdir(self, workdir, chroot=False):
        """Set the working directory. Cannot be set more than once unless chroot is True"""
        if not chroot and hasattr(self, "workdir") and self.workdir != workdir:
                raise ValueError("self.workdir != workdir: %s, %s" % (self.workdir,  workdir))

        self.workdir = os.path.abspath(workdir)

        # Files required for the execution.
        self.input_file = File(os.path.join(self.workdir, "run.abi"))
        self.output_file = File(os.path.join(self.workdir, "run.abo"))
        self.files_file = File(os.path.join(self.workdir, "run.files"))
        self.job_file = File(os.path.join(self.workdir, "job.sh"))
        self.log_file = File(os.path.join(self.workdir, "run.log"))
        self.stderr_file = File(os.path.join(self.workdir, "run.err"))
        self.start_lockfile = File(os.path.join(self.workdir, "__startlock__"))

        # Directories with input|output|temporary data.
        self.indir = Directory(os.path.join(self.workdir, "indata"))
        self.outdir = Directory(os.path.join(self.workdir, "outdata"))
        self.tmpdir = Directory(os.path.join(self.workdir, "tmpdata"))

        # stderr and output file of the queue manager. Note extensions.
        self.qerr_file = File(os.path.join(self.workdir, "queue.qerr"))
        self.qout_file = File(os.path.join(self.workdir, "queue.qout"))

    def set_manager(self, manager):
        """Set the `TaskManager` to use to launch the Task."""
        self.manager = manager.deepcopy()

    @property
    def work(self):
        """The WorkFlow containing this `Task`."""
        return self._work

    def set_work(self, work):
        """Set the WorkFlow associated to this `Task`."""
        if not hasattr(self, "_work"):
            self._work = work
        else: 
            if self._work != work:
                raise ValueError("self._work != work")

    @property
    def flow(self):
        """The Flow containing this `Task`."""
        return self.work.flow

    @lazy_property
    def pos(self):
        """The position of the task in the Flow"""
        for i, task in enumerate(self.work):
            if self == task: 
                return (self.work.pos, i)
        raise ValueError("Cannot find the position of %s in flow %s" % (self, self.flow))

    def make_input(self):
        """Construct and write the input file of the calculation."""
        return self.strategy.make_input()

    def ipath_from_ext(self, ext):
        """
        Returns the path of the input file with extension ext.
        Use it when the file does not exist yet.
        """
        return os.path.join(self.workdir, self.prefix.idata + "_" + ext)

    def opath_from_ext(self, ext):
        """
        Returns the path of the output file with extension ext.
        Use it when the file does not exist yet.
        """
        return os.path.join(self.workdir, self.prefix.odata + "_" + ext)

    @abc.abstractproperty
    def executable(self):
        """
        Path to the executable associated to the task (internally stored in self._executable).
        """

    def set_executable(self, executable):
        """Set the executable associate to this task."""
        self._executable = executable

    @property
    def process(self):
        try:
            return self._process
        except AttributeError:
            # Attach a fake process so that we can poll it.
            return FakeProcess()

    @property
    def is_completed(self):
        """True if the task has been executed."""
        return self.status >= self.S_DONE

    @property
    def can_run(self):
        """The task can run if its status is < S_SUB and all the other dependencies (if any) are done!"""
        all_ok = all([stat == self.S_OK for stat in self.deps_status])
        #print("can_run: all_ok ==  ",all_ok)
        return self.status < self.S_SUB and all_ok

    def not_converged(self):
        """Return True if the calculation is not converged."""
        report = self.get_event_report()
        return report.filter_types(self.CRITICAL_EVENTS)

    def run_etime(self):
        """
        String with the wall-time

        ...note:
            The clock starts when self.status becomes S_RUN.
            thus run_etime does not correspond to the effective wall-time.
        """
        s = "None"
        if self.start_datetime is not None:
            stop = self.stop_datetime
            if stop is None:
                stop = datetime.datetime.now()

            # Compute time-delta, convert to string and remove microseconds (in any)
            s = str(stop - self.start_datetime)
            microsec = s.find(".")
            if microsec != -1: s = s[:microsec]

        return s

    def cancel(self):
        """
        Cancel the job. Returns 1 if job was cancelled.
        """
        if self.queue_id is None: return 0 
        if self.status >= self.S_DONE: return 0 

        exit_status = self.manager.cancel(self.queue_id)
        if exit_status != 0: return 0

        # Remove output files and reset the status.
        self.reset()
        return 1

    def _on_done(self):
        self.fix_ofiles()

    def _on_ok(self):
        # Read timing data.
        #self.read_timing()
        # Fix output file names.
        self.fix_ofiles()
        # Get results
        results = self.on_ok()
        # Set internal flag.
        self._finalized = True

        return results

    def on_ok(self):
        """
        This method is called once the `Task` has reached status S_OK. 
        Subclasses should provide their own implementation

        Returns:
            Dictionary that must contain at least the following entries:
                returncode:
                    0 on success. 
                message: 
                    a string that should provide a human-readable description of what has been performed.
        """
        return dict(returncode=0, 
                    message="Calling on_all_ok of the base class!")

    def fix_ofiles(self):
        """
        This method is called when the task reaches S_OK.
        It changes the extension of particular output files
        produced by Abinit so that the 'official' extension
        is preserved e.g. out_1WF14 --> out_1WF
        """
        filepaths = self.outdir.list_filepaths()
        logger.info("in fix_ofiles with filepaths %s" % filepaths) 

        old2new = FilepathFixer().fix_paths(filepaths)

        for old, new in old2new.items():
            logger.debug("will rename old %s to new %s" % (old, new))
            os.rename(old, new)

    def _restart(self, no_submit=False):
        """
        Called by restart once we have finished preparing the task for restarting.

        Return True if task has been restarted
        """
        self.set_status(self.S_READY, info_msg="Restarted on %s" % time.asctime())

        # Increase the counter.
        self.num_restarts += 1
        self.history.append("Restarted on %s, num_restarts %d" % (time.asctime(), self.num_restarts))

        if not no_submit:
            # Remove the lock file
            self.start_lockfile.remove()
            # Relaunch the task.
            fired = self.start()
            if not fired:
                self.history.append("[%s], restart failed" % time.asctime())
        else:
            fired = False

        return fired

    def restart(self):
        """
        Restart the calculation.  Subclasses should provide a concrete version that 
        performs all the actions needed for preparing the restart and then calls self._restart
        to restart the task. The default implementation is empty.

        Returns:
            1 if job was restarted, 0 otherwise.
        """
        logger.debug("Calling the **empty** restart method of the base class")
        return 0

    def poll(self):
        """Check if child process has terminated. Set and return returncode attribute."""
        self._returncode = self.process.poll()

        if self._returncode is not None:
            self.set_status(self.S_DONE)

        return self._returncode

    def wait(self):
        """Wait for child process to terminate. Set and return returncode attribute."""
        self._returncode = self.process.wait()
        self.set_status(self.S_DONE)

        return self._returncode

    def communicate(self, input=None):
        """
        Interact with process: Send data to stdin. Read data from stdout and stderr, until end-of-file is reached. 
        Wait for process to terminate. The optional input argument should be a string to be sent to the 
        child process, or None, if no data should be sent to the child.

        communicate() returns a tuple (stdoutdata, stderrdata).
        """
        stdoutdata, stderrdata = self.process.communicate(input=input)
        self._returncode = self.process.returncode
        self.set_status(self.S_DONE)

        return stdoutdata, stderrdata 

    def kill(self):
        """Kill the child."""
        self.process.kill()
        self.set_status(self.S_ERROR)
        self._returncode = self.process.returncode

    @property
    def returncode(self):
        """
        The child return code, set by poll() and wait() (and indirectly by communicate()). 
        A None value indicates that the process hasn't terminated yet.
        A negative value -N indicates that the child was terminated by signal N (Unix only).
        """
        try: 
            return self._returncode
        except AttributeError:
            return 0

    def reset(self):
        """
        Reset the task status. Mainly used if we made a silly mistake in the initial
        setup of the queue manager and we want to fix it and rerun the task.

        Returns:
            0 on success, 1 if reset failed.
        """
        # Can only reset tasks that are done.
        if self.status < self.S_DONE: return 1

        self.set_status(self.S_INIT, info_msg="Reset on %s" % time.asctime())
        self.set_queue_id(None)

        # Remove output files otherwise the EventParser will think the job is still running
        self.output_file.remove()
        self.log_file.remove()
        self.stderr_file.remove()
        self.start_lockfile.remove()
        self.qerr_file.remove()
        self.qout_file.remove()

        # TODO send a signal to the flow 
        #self.workflow.check_status()
        return 0

    @property
    def queue_id(self):
        """Queue identifier returned by the Queue manager. None if not set"""
        try:
            return self._queue_id
        except AttributeError:
            return None

    def set_queue_id(self, queue_id):
        """Set the task identifier."""
        self._queue_id = queue_id

    @property
    def has_queue(self):
        """True if we are submitting jobs via a queue manager."""
        return self.manager.qadapter.QTYPE.lower() != "shell"

    @property
    def tot_cores(self):
        """Total number of CPUs used to run the task."""
        return self.manager.tot_cores
                                                         
    @property
    def mpi_procs(self):
        """Number of CPUs used for MPI."""
        return self.manager.mpi_procs
                                                         
    @property
    def omp_threads(self):
        """Number of CPUs used for OpenMP."""
        return self.manager.omp_threads

    @property
    def status(self):
        """Gives the status of the task."""
        return self._status

    def set_status(self, status, info_msg=None):
        """
        Set and return the status of the task.

        Args:
            status:
                Status object or string representation of the status
            info_msg:
                string with human-readable message used in the case of errors (optional)
        """
        status = Status.as_status(status)

        changed = True
        if hasattr(self, "_status"):
            changed = (status != self._status)

        self._status = status

        if status == self.S_RUN:
            # Set start_datetime when the task enters S_RUN
            if self.start_datetime is None:
                self.start_datetime = datetime.datetime.now()

        # Add new entry to history only if the status has changed.
        if changed:
            if status == self.S_SUB: 
                self._submission_time = time.time()
                self.history.append("Submitted on %s" % time.asctime())

            if status == self.S_OK:
                self.history.append("Completed on %s" % time.asctime())

            if status == self.S_ABICRITICAL:
                self.history.append("Error info:\n %s" % str(info_msg))

        if status == self.S_DONE:
            self.stop_datetime = datetime.datetime.now()

            # Execute the callback
            self._on_done()
                                                                                
        if status == self.S_OK:
            #if status == self.S_UNCONVERGED:
            #    logger.debug("Task %s broadcasts signal S_UNCONVERGED" % self)
            #    dispatcher.send(signal=self.S_UNCONVERGED, sender=self)

            # Finalize the task.
            if not self.finalized:
                self._on_ok()
                                                                                
            logger.debug("Task %s broadcasts signal S_OK" % self)
            dispatcher.send(signal=self.S_OK, sender=self)

        return status

    def check_status(self):
        """
        This function checks the status of the task by inspecting the output and the
        error files produced by the application and by the queue manager.

        The process
        1) see it the job is blocked
        2) see if an error occured at submitting the job the job was submitted, TODO these problems can be solved
        3) see if there is output
            4) see if abinit reports problems
            5) see if both err files exist and are empty
        6) no output and no err files, the job must still be running
        7) try to find out what caused the problems
        8) there is a problem but we did not figure out what ...
        9) the only way of landing here is if there is a output file but no err files...
        """

        # 1) A locked task can only be unlocked by calling set_status explicitly.
        # an errored task, should not end up here but just to be sure
        black_list = [self.S_LOCKED, self.S_ERROR]
        if self.status in black_list:
            return

        # 2) Check the returncode of the process (the process of submitting the job) first.
        # this point type of problem should also be handled by the scheduler error parser
        if self.returncode != 0:
            # The job was not submitter properly
            info_msg = "return code %s" % self.returncode
            return self.set_status(self.S_QUEUECRITICAL, info_msg=info_msg)           

#        err_msg = None
#=======
#            if not self.stderr_file.exists and not self.qerr_file.exists:
#                # The job is still in the queue.
#                return self.status
#
#            else:
#                # Analyze the standard error of the executable:
#                if self.stderr_file.exists:
#                    err_msg = self.stderr_file.read()
#                    if err_msg:
#                        logger.critical("%s: executable stderr:\n %s" % (self, err_msg))
#                        return self.set_status(self.S_ERROR, info_msg=err_msg)
#
#                # Analyze the error file of the resource manager.
#                if self.qerr_file.exists:
#                    err_msg = self.qerr_file.read()
#                    if err_msg:
#                        logger.critical("%s: queue stderr:\n %s" % (self, err_msg))
#                        return self.set_status(self.S_ERROR, info_msg=err_msg)
#
#                return self.status
#
#        # Check if the run completed successfully.
#        report = self.get_event_report()
#
#        if report.run_completed:
#            # Check if the calculation converged.
#            not_ok = self.not_converged()

#            if not_ok:
#                return self.set_status(self.S_UNCONVERGED)
#            else:
#                return self.set_status(self.S_OK)

#       # This is the delicate part since we have to discern among different possibilities:
        #
        # 1) Calculation stopped due to an Abinit Error or Bug.
        #
        # 2) Segmentation fault that (by definition) was not handled by ABINIT.
        #    In this case we check if the ABINIT standard error is not empty.
        #    hoping that nobody has written to stderr (e.g. libraries in debug mode)
        #
        # 3) Problem with the resource manager and/or the OS (walltime error, resource error, phase of the moon ...)
        #    In this case we check if the error file of the queue manager is not empty.
        #    Also in this case we *assume* that there's something wrong if the stderr of the queue manager is not empty
        # 
        # 4) Calculation is still running!
        #
        # Point 2) and 3) are the most complicated since there's no standard!

        # 1) Search for possible errors or bugs in the ABINIT **output** file.
#        if report.errors or report.bugs:
#            logger.critical("%s: Found Errors or Bugs in ABINIT main output!" % self)
#            return self.set_status(self.S_ERROR, info_msg=str(report.errors) + str(report.bugs))

        # 2) Analyze the stderr file for Fortran runtime errors.
#       >>>>>>> pymatgen-matteo/master

        err_msg = None
        if self.stderr_file.exists:
            err_msg = self.stderr_file.read()

        err_info = None
        if self.qerr_file.exists:
            err_info = self.qerr_file.read()

        # 3) Start to check if the output file has been created.
        if self.output_file.exists:
            report = self.get_event_report()
            if report.run_completed:
                # Check if the calculation converged.
                not_ok = self.not_converged()
                if not_ok:
                    return self.set_status(self.S_UNCONVERGED)
                else:
                    return self.set_status(self.S_OK)

            # 4)
            if report.errors or report.bugs:
                if report.errors:
                    logger.debug('"Found errors in report')
                    for error in report.errors:
                        logger.debug(str(error))
                        try:
                            self.abi_errors.append(error)
                        except AttributeError:
                            self.abi_errors = [error]
                if report.bugs:
                    logger.debug('Found bugs in report:')
                    for bug in report.bugs:
                        logger.debug(str(bug))
                # Abinit reports problems
                logger.critical("%s: Found Errors or Bugs in ABINIT main output!" % self)
                info_msg = str(report.errors) + str(report.bugs)
                return self.set_status(self.S_ABICRITICAL, info_msg=info_msg)
                # The job is unfixable due to ABINIT errors

            # 5)
            if self.stderr_file.exists and not err_info:
                if self.qerr_file.exists and not err_msg:
                    # there is output and no errors
                    # Check if the run completed successfully.
#                    if report.run_completed:
#                        # Check if the calculation converged.
#                        not_ok = self.not_converged()
#                        if not_ok:
#                            return self.set_status(self.S_UNCONVERGED)
#                            # The job finished but did not converge
#                        else:
#                            return self.set_status(self.S_OK)
#                            # The job finished properly

                    return self.set_status(self.S_RUN)
                    # The job still seems to be running

        # 6)
        if not self.output_file.exists:
            logger.debug("output_file does not exists")
            if not self.stderr_file.exists and not self.qerr_file.exists:     # No output at all
                return self.status
                # The job is still in the queue.

        # 7) Analyze the files of the resource manager and abinit and execution err (mvs)
        if self.qerr_file.exists:
            from pymatgen.io.gwwrapper.scheduler_error_parsers import get_parser
            scheduler_parser = get_parser(self.manager.qadapter.QTYPE, err_file=self.qerr_file.path,
                                          out_file=self.qout_file.path, run_err_file=self.stderr_file.path)
            scheduler_parser.parse()

            if scheduler_parser.errors:
                # the queue errors in the task
                logger.debug('scheduler errors found:')
                logger.debug(str(scheduler_parser.errors))
                self.queue_errors = scheduler_parser.errors
                return self.set_status(self.S_QUEUECRITICAL)
                # The job is killed or crashed and we know what happened
            else:
                if len(err_info) > 0:
                    logger.debug('found unknown queue error: %s' % str(err_info))
                    return self.set_status(self.S_QUEUECRITICAL, info_msg=err_info)
                    # The job is killed or crashed but we don't know what happened
                    # it is set to queuecritical, we will attempt to fix it by running on more resources

        # 8) analizing the err files and abinit output did not identify a problem
        # but if the files are not empty we do have a problem but no way of solving it:
        if err_msg is not None and len(err_msg) > 0:
            logger.debug('found error message:\n %s' % str(err_msg))
            return self.set_status(self.S_QUEUECRITICAL, info_msg=err_info)
            # The job is killed or crashed but we don't know what happend
            # it is set to queuecritical, we will attempt to fix it by running on more resources

        # 9) if we still haven't returned there is no indication of any error and the job can only still be running
        # but we should actually never land here, or we have delays in the file system ....
        # print('the job still seems to be running maybe it is hanging without producing output... ')

        return self.set_status(self.S_RUN)

    def reduce_memory_demand(self):
        """
        Method that can be called by the flow to decrease the memory demand of a specific task.
        Returns True in case of success, False in case of Failure.
        Should be overwritten by specific tasks.
        """
        return False

    def speed_up(self):
        """
        Method that can be called by the flow to decrease the time needed for a specific task.
        Returns True in case of success, False in case of Failure
        Should be overwritten by specific tasks.
        """
        return False

    def out_to_in(self, out_file):
        """
        Move an output file to the output data directory of the `Task` 
        and rename the file so that ABINIT will read it as an input data file.

        Returns:
            The absolute path of the new file in the indata directory.
        """
        in_file = os.path.basename(out_file).replace("out", "in", 1)
        dest = os.path.join(self.indir.path, in_file)
                                                                           
        if os.path.exists(dest) and not os.path.islink(dest):
            logger.warning("Will overwrite %s with %s" % (dest, out_file))
                                                                           
        os.rename(out_file, dest)
        return dest

    def inlink_file(self, filepath):
        """
        Create a symbolic link to the specified file in the 
        directory containing the input files of the task.
        """
        if not os.path.exists(filepath): 
            logger.debug("Creating symbolic link to not existent file %s" % filepath)

        # Extract the Abinit extension and add the prefix for input files.
        root, abiext = abi_splitext(filepath)

        infile = "in_" + abiext
        infile = self.indir.path_in(infile)

        # Link path to dest if dest link does not exist.
        # else check that it points to the expected file.
        logger.debug("Linking path %s --> %s" % (filepath, infile))

        if not os.path.exists(infile):
            os.symlink(filepath, infile)
        else:
            if os.path.realpath(infile) != filepath:
                raise self.Error("infile %s does not point to filepath %s" % (infile, filepath))

    def make_links(self):
        """
        Create symbolic links to the output files produced by the other tasks.

        ..warning:
            
            This method should be called only when the calculation is READY because
            it uses a heuristic approach to find the file to link.
        """
        for dep in self.deps:
            filepaths, exts = dep.get_filepaths_and_exts()

            for path, ext in zip(filepaths, exts):
                logger.info("Need path %s with ext %s" % (path, ext))
                dest = self.ipath_from_ext(ext)

                if not os.path.exists(path): 
                    # Try netcdf file. TODO: this case should be treated in a cleaner way.
                    path += "-etsf.nc"
                    if os.path.exists(path): dest += "-etsf.nc"

                if not os.path.exists(path):
                    err_msg = "%s: %s is needed by this task but it does not exist" % (self, path)
                    logger.critical(err_msg)
                    raise self.Error(err_msg)

                # Link path to dest if dest link does not exist.
                # else check that it points to the expected file.
                logger.debug("Linking path %s --> %s" % (path, dest))

                if not os.path.exists(dest):
                    os.symlink(path, dest)
                else:
                    if os.path.realpath(dest) != path:
                        raise self.Error("dest %s does not point to path %s" % (dest, path))

        for f in self.required_files:
            path, dest = f.filepath, self.ipath_from_ext(f.ext)
      
            # Link path to dest if dest link does not exist.
            # else check that it points to the expected file.
            logger.debug("Linking path %s --> %s" % (path, dest))
                                                                                         
            if not os.path.exists(dest):
                os.symlink(path, dest)
            else:
                if os.path.realpath(dest) != path:
                    raise self.Error("dest %s does not point to path %s" % (dest, path))

    @abc.abstractmethod
    def setup(self):
        """Public method called before submitting the task."""

    def _setup(self):
        """
        This method calls self.setup after having performed additional operations
        such as the creation of the symbolic links needed to connect different tasks.
        """
        self.make_links()
        self.setup()

    # TODO: For the time being, we inspect the log file,
    # We will start to use the output file when the migration to YAML is completed
    def get_event_report(self, source="log"):
        """
        Analyzes the main output file for possible Errors or Warnings.

        Args:
            source:
                "output" for the main output file.
                "log" for the log file.

        Returns:
            `EventReport` instance or None if the main output file does not exist.
        """
        ofile = {
            "output": self.output_file,
            "log": self.log_file}[source]

        if not ofile.exists:
            return None

        parser = events.EventsParser()
        try:
            return parser.parse(ofile.path)

        except parser.Error as exc:
            # Return a report with an error entry with info on the exception.
            logger.critical("%s: Exception while parsing ABINIT events:\n %s" % (ofile, str(exc)))
            self.set_status(self.S_ABICRITICAL, info_msg=str(exc))
            return parser.report_exception(ofile.path, exc)

    def get_results(self, **kwargs):
        """
        Returns `NodeResults` instance.
        Subclasses should extend this method (if needed) by adding 
        specialized code that performs some kind of post-processing.
        """
        # Check whether the process completed.
        if self.returncode is None:
            raise self.Error("return code is None, you should call wait, communitate or poll")

        if self.status is None or self.status < self.S_DONE:
            raise self.Error("Task is not completed")

        return self.Results.from_node(self)

    def move(self, dest, is_abspath=False):
        """
        Recursively move self.workdir to another location. This is similar to the Unix "mv" command.
        The destination path must not already exist. If the destination already exists
        but is not a directory, it may be overwritten depending on os.rename() semantics.

        Be default, dest is located in the parent directory of self.workdir.
        Use is_abspath=True to specify an absolute path.
        """
        if not is_abspath:
            dest = os.path.join(os.path.dirname(self.workdir), dest)

        shutil.move(self.workdir, dest)

    def in_files(self):
        """Return all the input data files used."""
        return self.indir.list_filepaths()

    def out_files(self):
        """Return all the output data files produced."""
        return self.outdir.list_filepaths()

    def tmp_files(self):
        """Return all the input data files produced."""
        return self.tmpdir.list_filepaths()

    def path_in_workdir(self, filename):
        """Create the absolute path of filename in the top-level working directory."""
        return os.path.join(self.workdir, filename)

    def rename(self, src_basename, dest_basename, datadir="outdir"):
        """
        Rename a file located in datadir.

        src_basename and dest_basename are the basename of the source file
        and of the destination file, respectively.
        """
        directory = {
            "indir": self.indir,
            "outdir": self.outdir,
            "tmpdir": self.tmpdir,
        }[datadir]

        src = directory.path_in(src_basename)
        dest = directory.path_in(dest_basename)

        os.rename(src, dest)

    def build(self, *args, **kwargs):
        """
        Creates the working directory and the input files of the `Task`.
        It does not overwrite files if they already exist.
        """
        # Create dirs for input, output and tmp data.
        self.indir.makedirs()
        self.outdir.makedirs()
        self.tmpdir.makedirs()

        # Write files file and input file.
        if not self.files_file.exists:
            self.files_file.write(self.filesfile_string)

        self.input_file.write(self.make_input())

        self.manager.write_jobfile(self)

    def rmtree(self, exclude_wildcard=""):
        """
        Remove all files and directories in the working directory

        Args:
            exclude_wildcard:
                Optional string with regular expressions separated by |.
                Files matching one of the regular expressions will be preserved.
                example: exclude_wildcard="*.nc|*.txt" preserves all the files
                whose extension is in ["nc", "txt"].
        """
        if not exclude_wildcard:
            shutil.rmtree(self.workdir)

        else:
            w = WildCard(exclude_wildcard)

            for dirpath, dirnames, filenames in os.walk(self.workdir):
                for fname in filenames:
                    filepath = os.path.join(dirpath, fname)
                    if not w.match(fname):
                        os.remove(filepath)

    def remove_files(self, *filenames):
        """Remove all the files listed in filenames."""
        filenames = list_strings(filenames)

        for dirpath, dirnames, fnames in os.walk(self.workdir):
            for fname in fnames:
                if fname in filenames:
                    filepath = os.path.join(dirpath, fname)
                    os.remove(filepath)

    def setup(self):
        """Base class does not provide any hook."""

    def start(self):
        """
        Starts the calculation by performing the following steps:

            - build dirs and files
            - call the _setup method
            - execute the job file by executing/submitting the job script.

        Returns:
            1 if task was started, 0 otherwise.
            
        """
        if self.status >= self.S_SUB:
            raise self.Error("Task status: %s" % str(self.status))

        if self.start_lockfile.exists:
            logger.warning("Found lock file: %s" % self.start_lockfile.relpath)
            return 0

        self.start_lockfile.write("Started on %s" % time.asctime())

        self.build()
        self._setup()

        # Add the variables needed to connect the node.
        for d in self.deps:
            vars = d.connecting_vars()
            logger.debug("Adding connecting vars %s " % vars)
            self.strategy.add_extra_abivars(vars)

        # Add the variables needed to read the required files
        for f in self.required_files:
            #raise NotImplementedError("")
            vars = irdvars_for_ext("DEN")
            logger.debug("Adding connecting vars %s " % vars)
            self.strategy.add_extra_abivars(vars)

        # Automatic parallelization
        if hasattr(self, "autoparal_fake_run"):
            try:
                self.autoparal_fake_run()
            except:
                # Log the exception and continue with the parameters specified by the user.
                logger.critical("autoparal_fake_run raised:\n%s" % straceback())
                self.set_status(self.S_ABICRITICAL)
                return 0

        # Start the calculation in a subprocess and return.
        self._process = self.manager.launch(self)

        return 1

    def start_and_wait(self, *args, **kwargs):
        """
        Helper method to start the task and wait for completetion.

        Mainly used when we are submitting the task via the shell
        without passing through a queue manager.
        """
        self.start(*args, **kwargs)
        retcode = self.wait()
        return retcode


class AbinitTask(Task):
    """
    Base class defining an ABINIT calculation
    """
    Results = AbinitTaskResults

    @classmethod
    def from_input(cls, abinit_input, workdir=None, manager=None):
        """
        Create an instance of `AbinitTask` from an ABINIT input.
    
        Args:
            abinit_input:
                `AbinitInput` object.
            workdir:
                Path to the working directory.
            manager:
                `TaskManager` object.
        """
        # TODO: Find a better way to do this. I will likely need to refactor the Strategy object
        strategy = StrategyWithInput(abinit_input)

        return cls(strategy, workdir=workdir, manager=manager)

    def setup(self):
        """
        Abinit has the very *bad* habit of changing the file extension by appending the characters in [A,B ..., Z] 
        to the output file, and this breaks a lot of code that relies of the use of a unique file extension.
        Here we fix this issue by renaming run.abo to run.abo_[number] if the output file "run.abo" already
        exists. A few lines of code in python, a lot of problems if you try to implement this trick in Fortran90. 
        """
        if self.output_file.exists:
            # Find the index of the last file (if any) and push.
            # TODO: Maybe it's better to use run.abo --> run(1).abo
            fnames = [f for f in os.listdir(self.workdir) if f.startswith(self.output_file.basename)]
            nums = [int(f) for f in [f.split("_")[-1] for f in fnames] if f.isdigit()]
            last = max(nums) if nums else 0
            new_path = self.output_file.path + "_" + str(last+1)

            logger.info("Will rename %s to %s" % (self.output_file.path, new_path))
            os.rename(self.output_file.path, new_path)

    @property
    def executable(self):
        """Path to the executable required for running the Task."""
        try:
            return self._executable
        except AttributeError:
            return "abinit"

    @property
    def pseudos(self):
        """List of pseudos used in the calculation."""
        return self.strategy.pseudos

    @property
    def isnc(self):
        """True if norm-conserving calculation."""
        return all(p.isnc for p in self.pseudos)

    @property
    def ispaw(self):
        """True if PAW calculation"""
        return all(p.ispaw for p in self.pseudos)

    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append
        pj = os.path.join

        app(self.input_file.path)                 # Path to the input file
        app(self.output_file.path)                # Path to the output file
        app(pj(self.workdir, self.prefix.idata))  # Prefix for input data
        app(pj(self.workdir, self.prefix.odata))  # Prefix for output data
        app(pj(self.workdir, self.prefix.tdata))  # Prefix for temporary data

        # Paths to the pseudopotential files.
        # Note that here the pseudos **must** be sorted according to znucl.
        for pseudo in self.pseudos:
            app(pseudo.path)

        return "\n".join(lines)

    def autoparal_fake_run(self):
        """
        Find an optimal set of parameters for the execution of the task 
        using the options specified in `TaskPolicy`.
        This method can change the ABINIT input variables and/or the 
        parameters passed to the `TaskManager` e.g. the number of CPUs for MPI and OpenMp.

        Returns:
           confs, optimal 
           where confs is a `ParalHints` object with the configuration reported by 
           autoparal and optimal is the optimal configuration selected.
           Returns (None, None) if some problem occurred.
        """
        logger.info("in autoparal_fake_run")
        policy = self.manager.policy

        if policy.autoparal == 0 or policy.max_ncpus in [None, 1]:
            logger.info("Nothing to do in autoparal, returning (None, None)")
            return None, None

        if policy.autoparal != 1:
            raise NotImplementedError("autoparal != 1")

        ############################################################################
        # Run ABINIT in sequential to get the possible configurations with max_ncpus
        ############################################################################

        # Set the variables for automatic parallelization
        autoparal_vars = dict(
            autoparal=policy.autoparal,
            max_ncpus=policy.max_ncpus)

        self.strategy.add_extra_abivars(autoparal_vars)

        # Build a simple manager to run the job in a shell subprocess on the frontend
        # we don't want to make a request to the queue manager for this simple job!
        seq_manager = self.manager.to_shell_manager(mpi_procs=1)

        # Return code is always != 0 
        process = seq_manager.launch(self)
        logger.info("fake run launched")
        retcode = process.wait()  

        # Remove the variables added for the automatic parallelization
        self.strategy.remove_extra_abivars(autoparal_vars.keys())

        ##############################################################
        # Parse the autoparal configurations from the main output file
        ##############################################################
        parser = ParalHintsParser()

        try:
            confs = parser.parse(self.output_file.path)
            #self.all_autoparal_confs = confs
            logger.info('speedup hints: \n' + str(confs) + '\n')
            # print("confs", confs)
        except parser.Error:
            logger.critical("Error while parsing Autoparal section:\n%s" % straceback())
            return None, None

        ######################################################
        # Select the optimal configuration according to policy
        ######################################################
        optconf = confs.select_optimal_conf(policy)
        #print("optimal autoparal conf:\n %s" % optconf)

        # Select the partition on which we'll be running
        #for i, c in enumerate(optconfs):
        #    self.manager.select_partition(optconfs) is not None:
        #        optconf = optconfs[i]
        #        break
        #else:
        #    raise RuntimeError("cannot find partition for this run!")

        # Write autoparal configurations to JSON file.
        d = confs.as_dict()
        d["optimal_conf"] = optconf
        json_pretty_dump(d, os.path.join(self.workdir, "autoparal.json"))

        ####################################################
        # Change the input file and/or the submission script
        ####################################################
        self.strategy.add_extra_abivars(optconf.vars)
                                                                  
        # Change the number of MPI/OMP cores.
        self.manager.set_mpi_procs(optconf.mpi_procs)
        if self.manager.has_omp:
            self.manager.set_omp_threads(optconf.omp_threads)

        # Change the memory per node if automemory evaluates to True.
        if policy.automemory and optconf.mem_per_cpu:
            # mem_per_cpu = max(mem_per_cpu, policy.automemory)
            self.manager.set_mem_per_cpu(optconf.mem_per_cpu)

        ##############
        # Finalization
        ##############
        # Reset the status, remove garbage files ...
        self.set_status(self.S_INIT)

        # Remove the output file since Abinit likes to create new files 
        # with extension .outA, .outB if the file already exists.
        os.remove(self.output_file.path)
        os.remove(self.log_file.path)
        os.remove(self.stderr_file.path)

        return confs, optconf

    def restart(self):
        """
        general restart used when scheduler problems have been taken care of
        """
        return self._restart()

    def reset_from_scratch(self):
        """
        restart from scratch, reuse of output
        this is to be used if a job is restarted with more resources after a crash
        """
        # remove all 'error', else the job will be seen as crashed in the next check status
        # even if the job did not run
        self.output_file.remove()
        self.log_file.remove()
        self.stderr_file.remove()
        self.start_lockfile.remove()

        return self._restart(no_submit=True)

    def fix_abicritical(self):
        """
        method to fix crashes/error caused by abinit
        currently:
            try to rerun with more resources, last resort if all else fails
        ideas:
            upon repetative no converging iscf > 2 / 12

        """
        # the crude, no idea what to do but this may work, solution.
        if self.manager.increase_resources():
            self.reset_from_scratch()
            return True
        else:
            self.set_status(self.S_ERROR, info_msg='could not increase resources any further')
            return False

    #@property
    #def timing(self):
    #    """Object with timing data. None if timing is not available"""
    #    try:
    #        return self._timing
    #    except AttributeError:
    #        return None

    #def read_timig(self):
    #    """
    #    Read timing data from the main output file and store it in self.timing if available.
    #    """
    #    from pymatgen.io.abitimer import AbinitTimerParser
    #    _timing = AbinitTimerParser()
    #    retval = timing.parse(self.output_file) 
    #    if retval == self.output_file: self._timing = _timing

# TODO
# Enable restarting capabilites:
# Before doing so I need:
#   1) Preliminary standardization of the ABINT events and critical WARNINGS (YAML)
#   2) Change the parser so that we can use strings in the input file.
#      We need this change for restarting structural relaxations so that we can read 
#      the initial structure from file.


class ScfTask(AbinitTask):
    """
    Self-consistent ground-state calculations.
    Provide support for in-place restart via (WFK|DEN) files
    """
    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def restart(self):
        """SCF calculations can be restarted if we have either the WFK file or the DEN file."""
        # Prefer WFK over DEN files since we can reuse the wavefunctions.
        restart_file = None
        for ext in ("WFK", "DEN"):
            restart_file = self.outdir.has_abiext(ext)
            irdvars = irdvars_for_ext(ext)
            if restart_file:
                break

        if not restart_file:
            raise self.RestartError("Cannot find WFK or DEN file to restart from.")

        # Move out --> in.
        self.out_to_in(restart_file)

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def inspect(self, **kwargs):
        """
        Plot the SCF cycle results with matplotlib.

        Returns
            `matplotlib` figure, None if some error occurred.
        """
        scf_cycle = abiinspect.GroundStateScfCycle.from_file(self.output_file.path)
        if scf_cycle is not None:
            return scf_cycle.plot(**kwargs)

    def get_results(self, **kwargs):
        results = super(ScfTask, self).get_results(**kwargs)

        # Open the GRS file and add its data to results.out
        from abipy.electrons.gsr import GSR_File
        gsr = GSR_File(self.outdir.has_abiext("GSR"))
        results["out"].update(gsr.as_dict())

        # Add files to GridFS
        return results.add_gridfs_files(GSR=gsr.filepath)


class NscfTask(AbinitTask):
    """
    Non-Self-consistent GS calculation.
    Provide in-place restart via WFK files
    """
    CRITICAL_EVENTS = [
        events.NscfConvergenceWarning,
    ]

    def restart(self):
        """NSCF calculations can be restarted only if we have the WFK file."""
        ext = "WFK"
        restart_file = self.outdir.has_abiext(ext)
        irdvars = irdvars_for_ext(ext)

        if not restart_file:
            raise self.RestartError("Cannot find the WFK file to restart from.")

        # Move out --> in.
        self.out_to_in(restart_file)

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def get_results(self, **kwargs):
        results = super(NscfTask, self).get_results(**kwargs)

        # Open the GRS file and add its data to results.out
        from abipy.electrons.gsr import GSR_File
        gsr = GSR_File(self.outdir.has_abiext("GSR"))
        results["out"].update(gsr.as_dict())

        # Add files to GridFS
        return results.add_gridfs_files(GSR=gsr.filepath)


class RelaxTask(AbinitTask):
    """
    Task for structural optimizations.
    """
    # What about a possible ScfConvergenceWarning?
    CRITICAL_EVENTS = [
        events.RelaxConvergenceWarning,
    ]

    def change_structure(self, structure):
        """Change the input structure."""
        print("changing structure")
        print("old:\n" + str(self.strategy.abinit_input.structure) + "\n")
        print("new:\n" + str(structure) + "\n")
        self.strategy.abinit_input.set_structure(structure)

    def read_final_structure(self):
        """Read the final structure from the GSR file."""
        gsr_file = self.outdir.has_abiext("GSR")
        if not gsr_file:
            raise self.RestartError("Cannot find the GSR file with the final structure to restart from.")

        with ETSF_Reader(gsr_file) as r:
            return r.read_structure()

    def restart(self):
        """
        Restart the structural relaxation.

        Structure relaxations can be restarted only if we have the WFK file or the DEN or the GSR file.
        from which we can read the last structure (mandatory) and the wavefunctions (not mandatory but useful).
        Prefer WFK over other files since we can reuse the wavefunctions.

        .. note:
            The problem in the present approach is that some parameters in the input
            are computed from the initial structure and may not be consisten with
            the modification of the structure done during the structure relaxation.
        """
        ofile = None
        for ext in ["WFK", "DEN"]:
            ofile = self.outdir.has_abiext(ext)
            if ofile:
                irdvars = irdvars_for_ext(ext)
                infile = self.out_to_in(ofile)
                break

        if not ofile:
            raise self.RestartError("Cannot find the WFK|DEN file to restart from.")

        # Read the relaxed structure from the GSR file.
        structure = self.read_final_structure()
                                                           
        # Change the structure.
        self.change_structure(structure)

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def inspect(self, **kwargs):
        """
        Plot the evolution of the structural relaxation with matplotlib.

        Returns
            `matplotlib` figure, None is some error occurred. 
        """
        relaxation = abiinspect.Relaxation.from_file(self.output_file.path)
        if relaxation is not None:
            return relaxation.plot(**kwargs)

    def get_results(self, **kwargs):
        results = super(RelaxTask, self).get_results(**kwargs)

        # Open the GRS file and add its data to results.out
        from abipy.electrons.gsr import GSR_File
        gsr = GSR_File(self.outdir.has_abiext("GSR"))
        results["out"].update(gsr.as_dict())

        # Add files to GridFS
        return results.add_gridfs_files(GSR=gsr.filepath)


class DdkTask(AbinitTask):
    """Task for DDK calculations."""

    def get_results(self, **kwargs):
        results = super(DdkTask, self).get_results(**kwargs)
        return results.add_gridfs_file(DDB=(self.outdir.has_abiext("DDB"), "t"))


class PhononTask(AbinitTask):
    """
    DFPT calculations for a single atomic perturbation.
    Provide support for in-place restart via (1WF|1DEN) files
    """
    # TODO: 
    # for the time being we don't discern between GS and PhononCalculations.
    # Restarting Phonon calculation is more difficult due to the crazy rules employed in ABINIT 
    CRITICAL_EVENTS = [
        events.ScfConvergenceWarning,
    ]

    def restart(self):
        """
        Phonon calculations can be restarted only if we have the 1WF file or the 1DEN file.
        from which we can read the first-order wavefunctions or the first order density.
        Prefer 1WF over 1DEN since we can reuse the wavefunctions.
        """
        #self.fix_ofiles()
        restart_file = None
        for ext in ["1WF", "1DEN"]:
            restart_file = self.outdir.has_abiext(ext)
            irdvars = irdvars_for_ext(ext)
            if restart_file:
                break

        if not restart_file:
            raise self.RestartError("Cannot find the 1WF|1DEN|file to restart from.")

        self.out_to_in(restart_file)

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def inspect(self, **kwargs):
        """
        Plot the Phonon SCF cycle results with matplotlib.

        Returns
            `matplotlib` figure, None if some error occurred.
        """
        scf_cycle = abiinspect.PhononScfCycle.from_file(self.output_file.path)
        if scf_cycle is not None:
            return scf_cycle.plot(**kwargs)

    def get_results(self, **kwargs):
        results = super(PhononTask, self).get_results(**kwargs)
        return results.add_gridfs_file(DDB=(self.outdir.has_abiext("DDB"), "t"))


class SigmaTask(AbinitTask):
    """
    Tasks for SIGMA calculations employing the self-consistent G approximation 
    Provide support for in-place restart via QPS files
    """
    CRITICAL_EVENTS = [
        events.QPSConvergenceWarning,
    ]

    def restart(self):
        # G calculations can be restarted only if we have the QPS file 
        # from which we can read the results of the previous step.
        ext = "QPS"
        restart_file = self.outdir.has_abiext(ext)
        irdvars = irdvars_for_ext(ext)

        if not restart_file:
            raise self.RestartError("Cannot find the QPS file to restart from.")

        self.out_to_in(restart_file)

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def get_results(self, **kwargs):
        results = super(SigmaTask, self).get_results(**kwargs)

        # Open the SIGRES file and add its data to results.out
        from abipy.electrons.gsr import GSR_File
        #sigres = SIGRES_File(self.outdir.has_abiext("SIGRES"))
        #results["out"].update(sigres.as_dict())
        #return results.add_gridfs_files(SIGRES=sigres.filepath)
        return results


class BseTask(AbinitTask):
    """
    Task for Bethe-Salpeter calculations.

    .. note:

        The BSE codes provides both iterative and direct schemes
        for the computation of the dielectric function. 
        The direct diagonalization cannot be restarted whereas 
        Haydock and CG support restarting.

        Bethe-Salpeter calculations with Haydock iterative scheme.
        Provide in-place restart via (BSR|BSC) files
    """
    CRITICAL_EVENTS = [
        events.HaydockConvergenceWarning,
        #events.BseIterativeDiagoConvergenceWarning,
    ]

    def restart(self):
        """
        BSE calculations with Haydock can be restarted only if we have the
        excitonic Hamiltonian and the HAYDR_SAVE file.
        """
        # TODO: This version seems to work but the main output file is truncated
        # TODO: Handle restart if CG method is used
        # TODO: restart should receive a list of critical events
        # the log file is complete though.
        irdvars = {}

        # Move the BSE blocks to indata.
        # This is done only once at the end of the first run.
        # Successive restarts will use the BSR|BSC files in the indir directory
        # to initialize the excitonic Hamiltonian
        count = 0
        for ext in ["BSR", "BSC"]:
            ofile = self.outdir.has_abiext(ext)
            if ofile:
                count += 1
                irdvars.update(irdvars_for_ext(ext))
                self.out_to_in(ofile)

        if not count:
            # outdir does not contain the BSR|BSC file.
            # This means that num_restart > 1 and the files should be in task.indir
            count = 0
            for ext in ["BSR", "BSC"]:
                ifile = self.indir.has_abiext(ext)
                if ifile:
                    count += 1

            if not count:
                raise self.RestartError("Cannot find BSR|BSC files in %s" % self.indir)

        # Rename HAYDR_SAVE files
        count = 0
        for ext in ["HAYDR_SAVE", "HAYDC_SAVE"]:
            ofile = self.outdir.has_abiext(ext)
            if ofile:
                count += 1
                irdvars.update(irdvars_for_ext(ext))
                self.out_to_in(ofile)

        if not count:
            raise self.RestartError("Cannot find the HAYDR_SAVE file to restart from.")

        # Add the appropriate variable for restarting.
        self.strategy.add_extra_abivars(irdvars)

        # Now we can resubmit the job.
        return self._restart()

    def get_results(self, **kwargs):
        results = super(BseTask, self).get_results(**kwargs)

        #mdf = MDF_File(self.outdir.has_abiext("MDF"))
        #results["out"].update(mdf.as_dict())
        #    out=mdf.as_dict(),
        #    epsilon_infinity
        #    optical_gap
        #)
        #return results.add_gridfs_files(MDF=mdf.filepath)
        return results


class OpticTask(Task):
    """
    Task for the computation of optical spectra with optic i.e.
    RPA without local-field effects and velocity operator computed from DDK files.
    """
    def __init__(self, optic_input, nscf_node, ddk_nodes, workdir=None, manager=None):
        """
        Create an instance of `OpticTask` from an string containing the input.
    
        Args:
            optic_input:
                string with the optic variables (filepaths will be added at run time).
            nscf_node:
                The NSCF task that will produce thw WFK file or string with the path of the WFK file.
            ddk_nodes:
                List of `DdkTask` nodes that will produce the DDK files or list of DDF paths.
            workdir:
                Path to the working directory.
            manager:
                `TaskManager` object.
        """
        # Convert paths to FileNodes
        self.nscf_node = Node.as_node(nscf_node)
        self.ddk_nodes = [Node.as_node(n) for n in ddk_nodes]
        assert len(ddk_nodes) == 3
        #print(self.nscf_node, self.ddk_nodes)

        deps = {n: "1WF" for n in self.ddk_nodes}
        deps.update({self.nscf_node: "WFK"})
        #print("deps", deps)

        strategy = OpticInput(optic_input)
        super(OpticTask, self).__init__(strategy=strategy, workdir=workdir, manager=manager, deps=deps)

    def set_workdir(self, workdir):
        super(OpticTask, self).set_workdir(workdir)
        # Small hack: the log file of optics is actually the main output file. 
        self.output_file = self.log_file

    @property
    def executable(self):
        """Path to the executable required for running the `OpticTask`."""
        try:
            return self._executable
        except AttributeError:
            return "optic"

    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append

        #optic.in     ! Name of input file
        #optic.out    ! Unused
        #optic        ! Root name for all files that will be produced
        app(self.input_file.path)                 # Path to the input file
        app(os.path.join(self.workdir, "unused"))           # Path to the output file
        app(os.path.join(self.workdir, self.prefix.odata))  # Prefix for output data

        return "\n".join(lines)

    @property
    def wfk_filepath(self):
        """Returns (at runtime) the absolute path of the WFK file produced by the NSCF run."""
        return self.nscf_node.outdir.has_abiext("WFK")

    @property
    def ddk_filepaths(self):
        """Returns (at runtime) the absolute path of the DDK files produced by the DDK runs."""
        return [ddk_task.outdir.has_abiext("1WF") for ddk_task in self.ddk_nodes]

    def make_input(self):
        """Construct and write the input file of the calculation."""
        # Set the file paths.
        files = "\n".join(self.ddk_filepaths + [self.wfk_filepath]) + "\n"

        # Get the input specified by the user
        user_inp = self.strategy.make_input()

        # Join them.
        return files + user_inp

    def setup(self):
        """Public method called before submitting the task."""

    def make_links(self):
        """
        Optic allows the user to specify the paths of the input file.
        hence we don't need to create symbolic links.
        """

    def get_results(self, **kwargs):
        results = super(OpticTask, self).get_results(**kwargs)
        #results.update(
        #"epsilon_infinity":
        #))
        return results


class AnaddbTask(Task):
    """Task for Anaddb runs (post-processing of DFPT calculations)."""
    def __init__(self, anaddb_input, ddb_node,
                 gkk_node=None, md_node=None, ddk_node=None, workdir=None, manager=None):
        """
        Create an instance of `AnaddbTask` from an string containing the input.

        Args:
            anaddb_input:
                string with the anaddb variables.
            ddb_node:
                The node that will produce the DDB file. Accept `Task`, `Workflow` or filepath.
            gkk_node:
                The node that will produce the GKK file (optional). Accept `Task`, `Workflow` or filepath.
            md_node:
                The node that will produce the MD file (optional). Accept `Task`, `Workflow` or filepath.
            gkk_node:
                The node that will produce the GKK file (optional). Accept `Task`, `Workflow` or filepath.
            workdir:
                Path to the working directory (optional).
            manager:
                `TaskManager` object (optional).
        """
        # Keep a reference to the nodes.
        self.ddb_node = Node.as_node(ddb_node)
        deps = {self.ddb_node: "DDB"}

        self.gkk_node = Node.as_node(gkk_node)
        if self.gkk_node is not None:
            deps.update({self.gkk_node: "GKK"})

        # I never used it!
        self.md_node = Node.as_node(md_node)
        if self.md_node is not None:
            deps.update({self.md_node: "MD"})

        self.ddk_node = Node.as_node(ddk_node)
        if self.ddk_node is not None:
            deps.update({self.ddk_node: "DDK"})

        super(AnaddbTask, self).__init__(strategy=anaddb_input, workdir=workdir, manager=manager, deps=deps)

    @property
    def executable(self):
        """Path to the executable required for running the `AnaddbTask`."""
        try:
            return self._executable
        except AttributeError:
            return "anaddb"

    @property
    def filesfile_string(self):
        """String with the list of files and prefixes needed to execute ABINIT."""
        lines = []
        app = lines.append

        app(self.input_file.path)          # 1) Path of the input file
        app(self.output_file.path)         # 2) Path of the output file
        app(self.ddb_filepath)             # 3) Input derivative database e.g. t13.ddb.in
        app(self.md_filepath)              # 4) Output molecular dynamics e.g. t13.md
        app(self.gkk_filepath)             # 5) Input elphon matrix elements  (GKK file)
        # FIXME check this one
        app(self.outdir.path_join("out"))  # 6) Base name for elphon output files e.g. t13
        app(self.ddk_filepath)             # 7) File containing ddk filenames for elphon/transport.

        return "\n".join(lines)

    @property
    def ddb_filepath(self):
        """Returns (at runtime) the absolute path of the input DDB file."""
        path = self.ddb_node.outdir.has_abiext("DDB")
        return path if path else "DDB_FILE_DOES_NOT_EXIST"

    @property
    def md_filepath(self):
        """Returns (at runtime) the absolute path of the input MD file."""
        if self.md_node is None:
            return "MD_FILE_DOES_NOT_EXIST"

        path = self.md_node.outdir.has_abiext("MD")
        return path if path else "MD_FILE_DOES_NOT_EXIST"

    @property
    def gkk_filepath(self):
        """Returns (at runtime) the absolute path of the input GKK file."""
        if self.gkk_node is None:
            return "GKK_FILE_DOES_NOT_EXIST"

        path = self.gkk_node.outdir.has_abiext("GKK")
        return path if path else "GKK_FILE_DOES_NOT_EXIST"

    @property
    def ddk_filepath(self):
        """Returns (at runtime) the absolute path of the input DKK file."""
        if self.ddk_node is None:
            return "DDK_FILE_DOES_NOT_EXIST"

        path = self.ddk_node.outdir.has_abiext("DDK")
        return path if path else "DDK_FILE_DOES_NOT_EXIST"

    def setup(self):
        """Public method called before submitting the task."""

    def make_links(self):
        """
        Anaddb allows the user to specify the paths of the input file.
        hence we don't need to create symbolic links.
        """

    def get_results(self, **kwargs):
        results = super(AnaddbTask, self).get_results(**kwargs)
        return results
