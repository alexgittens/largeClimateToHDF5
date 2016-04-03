# to test speed
# salloc -N 15 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 75 -u python-mpi -u ./concat.py
#
# to run in production
# salloc -N 60 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 300 -u python-mpi -u ./concat.py
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 72 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
# - set the buffer size
#
# Note: the memory usage should be determined by rawvar and reshapedrawvar,
# each of which are < 2GB in this case, so (ignoring MPI-IO aggregator needs
# and other overhead), each node should be able to support 32 processes (128 GB/node)
# Being generous, lets use 5 per node (32 cores/node => 6 cores per process)
# 
# Note: the number of aggregator processes is set to the number of OSTs

from mpi4py import MPI
from netCDF4 import Dataset
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import time
import math, sys

rank = MPI.COMM_WORLD.Get_rank()
mpi_info = MPI.Info.Create()
numProcs = MPI.COMM_WORLD.Get_size()

procslist = np.arange(numProcs)

numWriters = 7

def status(message, ranks=procslist):
    if rank in ranks:
        print "%s, process %d: %s" % (time.asctime(time.localtime()), rank, message)

def report(message):
    status(message, [0])

def reportbarrier(message):
    MPI.COMM_WORLD.Barrier()
    report(message)

datapath = "/global/cscratch1/sd/gittens/large-climate-dataset/data"
filelist = [fname for fname in listdir(datapath) if fname.endswith(".nc")]

REDUCEDNUMFILES = 7
filelist = [filelist[idx] for idx in np.arange(REDUCEDNUMFILES)]

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))

myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
#status("%d input files" % len(myfiles))
myhandles = [None]*len(myfiles) 
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r")
mynumfiles = len(myhandles)

colfilenames = MPI.COMM_WORLD.gather(myfiles, root=0)
if rank == 0:
   colnames = [item for sublist in colfilenames for item in sublist] 
   print colnames
   with open("colfilenames.txt", "w") as colfout:
       np.save(colfout, np.array(colnames))

reportbarrier("Finished opening all files")

numtimeslices = 8 # each variable comes in 8 time slices, and one time slice becomes one column

#varnames = ["T", "U", "V", "Q", "Z3"]
varnames = ["T"]
numvars = len(varnames)
numlevels = 30
numlats = 768
numlongs = 1152
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices

report("Creating file and dataset")

foutname = "superstrided/atmosphere"
#Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
if rank in np.arange(numWriters):
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    propfaid.set_alignment(1024, 1024*1024)
    propfaid.set_sieve_buf_size(numCols*8*20) # be able to store 20 rows worth of data
    fid = h5py.h5f.create(join(datapath, foutname + str(rank) + ".hdf5"), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
    fout = h5py.File(fid)
    #fout = h5py.File(join(datapath, foutname), "w")

# create the rows dataset using the low-level api so I can force it to not do zero-filling, then convert to a high level object
if rank in np.arange(numWriters):
    mynumrowspervar = len(filter( lambda r: r % numWriters == rank, np.arange(flattenedlength)))
    spaceid = h5py.h5s.create_simple((mynumrowspervar*numvars, numCols))
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
    rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating file and dataset")

# write output files one row at a time. to save on IO time, assume that we have enough memory to keep the
# current variable for each one of the files assigned to the current process in memory

if rank in np.arange(numWriters):
    collectedvals = np.ascontiguousarray(np.empty((numCols,), dtype=np.float64))
else:
    collectedvals = None
mynumcols = numtimeslices * len(myhandles)
rawvar = np.empty((numtimeslices, numlevels, numlats, numlongs), dtype=np.float64)
curvals = np.empty((flattenedlength, mynumcols), dtype=np.float64)

localcolumncount = mynumfiles*numtimeslices
columncounts = MPI.COMM_WORLD.gather(localcolumncount, root=0)
columncounts = MPI.COMM_WORLD.bcast(columncounts, root=0)

if rank in np.arange(numWriters):
    print columncounts
    columndisplacements = np.r_[0, np.cumsum(columncounts[:-1])]
else:
    columndisplacements = None

for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    status("Loading %s from %d files" % (curvar, mynumfiles))
    for (hindex, fh) in enumerate(myhandles):
        rawvar[:] = fh[curvar][:]
        curvals[:, (hindex * numtimeslices):((hindex + 1)*numtimeslices)] = \
                np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
        if (hindex % 1 == 0):
            status("... loaded %d of %d" % (hindex + 1, mynumfiles))
    status("Done loading");

    reportbarrier("Gathering rows and writing") 
    currow = 0
    for rowoffset in np.arange(flattenedlength):
        MPI.COMM_WORLD.Gatherv([curvals[rowoffset, :], mynumcols], 
            [collectedvals, columncounts, columndisplacements, MPI.DOUBLE], root=(rowoffset % numWriters))
        if rank in np.arange(numWriters):
            rows[varidx*mynumrowspervar + currow, :] = collectedvals
            currow = currow + 1
        report("Wrote row %d/%d" % (rowoffset, flattenedlength))

for fh in enumerate(myhandles):
    fh.close()
fout.close()

