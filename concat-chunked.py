# to run
# salloc -N 30 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 16 -n 60 -u python-mpi -u ./concat-chunked.py
#
# DONT FORGET TO MAKE THE OUTPUT USE MANY OSTS

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
numProcs = MPI.COMM_WORLD.Get_size()

procslist = np.arange(numProcs)

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

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))

# open all the files associated with this process and keep handles around (avoid metadata costs)
myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
status("%d input files" % len(myfiles))
#myhandles = [None]*len(myfiles) 
myhandles = [None]*int(math.ceil(len(filelist)/numProcs))
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r")

reportbarrier("Finished opening all files")

# to do efficient collective IO, note that we want to make sure all the
# processes are writing to the same contiguous blocks of rows
numtimeslices = 8 # each variable comes in 8 time slices, and one time slice becomes one column
allcoloffsets = np.arange(0, numtimeslices*len(filelist), numtimeslices)
mycoloffsets = [coloffset for (index, coloffset) in enumerate(allcoloffsets) if (index % numProcs == rank)]

numlevels = 30
numlats = 768
numlongs = 1152
numvars = 1
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices

# means about 3.75 MB for 60 processes
# and will have 129600 chunks in rows, 170 chunks in columns
rowchunk = 1024
colchunk = 8*numProcs

report("Creating file")
fout = h5py.File(join(datapath, "atmosphere-chunked.hdf5"), "w", driver="mpio", comm=MPI.COMM_WORLD)

report("Creating dataset")
#rows = fout.create_dataset("rows", (numRows, numCols), dtype=np.float64, chunks=(colchunk, rowchunk))

# create the rows dataset using the low-level api so I can force it to not do zero-filling, then convert to a high level object
spaceid = h5py.h5s.create_simple((numRows, numCols))
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
plist.set_chunk((rowchunk, colchunk))
datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating dataset")
reportbarrier("Writing T")

varidx = 0
for (hindex, fh) in enumerate(myhandles):
    reportbarrier("Writing %d/%d chunks for the current variable" % (hindex + 1, len(myhandles)))

    if fh is None:
        with rows.collective:
            pass
    else:
        myoffset = mycoloffsets[hindex]
        reportbarrier("Starting to load variable chunks")
        T = fh["T"][:]
        Treshaped = np.reshape(T, (flattenedlength, numtimeslices)) # uses C ordering
        reportbarrier("Finished loading variable chunks")
        status("Starting to write out variable chunks")
        with rows.collective:
            rows[varidx*flattenedlength:(varidx+1)*flattenedlength, myoffset:(myoffset + numtimeslices)] = Treshaped
        status("Finished writing out variable chunks")

for fh in enumerate(myhandles):
    if fh is not None:
        fh.close()
fout.close()

