# to test speed
# salloc -N 32 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 2 -n 510 -u python-mpi -u ./concat.py
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 140 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
#
# Note: apparently, the number of aggregator processes is set to the number of OSTs

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

filelist = [fname for fname in filelist[:400]]

report("Using %d processes" % numProcs)
report("Found %d input files, starting to open" % len(filelist))

assert( len(filelist) % numProcs == 0)

# open all the files associated with this process and keep handles around (avoid metadata costs)
myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
numFilesPerProc = len(myfiles)
myhandles = [None]*len(myfiles)
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r") 

reportbarrier("Finished opening all files")

varnames = ["T", "U", "V", "Q", "Z3"]
#varnames = ["T"]
numvars = len(varnames)
numtimeslices = 8
numlevels = 30
numlats = 768
numlongs = 1152
numlevdivs = 6
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices
rowChunkSize = numlats*numlongs/numlevdivs

assert ((numlats * numlongs) % numlevdivs == 0)

report("Creating file and dataset")

foutname = "superstrided/atmosphere.hdf5"
#Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
propfaid.set_fapl_mpio(MPI.COMM_WORLD, mpi_info)
propfaid.set_alignment(1024*1024*1024, 1024*1024)
#propfaid.set_sieve_buf_size(1024*1024)
fid = h5py.h5f.create(join(datapath, foutname), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
fout = h5py.File(fid)

# create the rows dataset using the low-level api so I can force it to not do zero-filling, then convert to a high level object
spaceid = h5py.h5s.create_simple((numRows, numCols))
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating file and dataset")

curlevdata = np.empty((numlats*numlongs, numFilesPerProc*numtimeslices), \
        dtype=np.float32)

for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    for curlev in np.arange(numlevels):

        # load the data for this level from my files
        reportbarrier("Loading data for level %d/%d" % (curlev + 1, numlevels))
        for (fhidx, fh) in enumerate(myhandles):
            if fh[curvar].shape[0] < numtimeslices:
                status("File %s has only %d timesteps for variable %s, simply repeating the first timestep" % (myfiles[fhidx], myfiles[fhidx].shape[0], curvar))
                for idx in np.arange(numtimeslices):
                    curlevdata[:, idx] = fh[curvar][1, curlev, ...].reshape(numlats*numlongs, 1)
            else:
                curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices] = \
                    fh[curvar][:, curlev, ...].reshape(numlats*numlongs, numtimeslices)
        reportbarrier("Done loading data for this level")
        
        # write out this level in several chunks of rows
        for chunkidx in np.arange(numlevdivs):
            reportbarrier("Writing row chunk %d/%d for this level" % \
                    (chunkidx + 1, numlevdivs))

            inputstartrow = chunkidx*rowChunkSize
            inputendrow = inputstartrow + rowChunkSize

            outputstartrow = varidx * flattenedlength + \
                    curlev * flattenedlength + chunkidx*rowChunkSize
            outputendrow = outputstartrow + rowChunkSize
            outputstartcol = rank*numFilesPerProc*numtimeslices
            outputendcol = outputstartcol + numFilesPerProc*numtimeslices

            with rows.collective:
                rows[outputstartrow:outputendrow, outputstartcol:outputendcol] = \
                        curlevdata[inputstartrow:inputendrow, :]



            # # use low-level collective writing interface b/c high-level might incur context management overhead?
            # # see https://groups.google.com/forum/#!topic/h5py/wDgTjGho0dY
            # dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            # dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE) 
            # slab_shape = (flattenedlength, numtimeslices)
            # slab_offset = (varidx*flattenedlength, mypreviousoffset)
            # memory_space = h5py.h5s.create_simple(slab_shape)
            # file_space = rows.id.get_space()
            # file_space.select_hyperslab(slab_offset, slab_shape)
            # rows.id.write(memory_space, file_space, flatdata, dxpl = dxpl)

for fh in myhandles:
    fh.close()
fout.close()

