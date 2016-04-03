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

filelist = [filelist[idx] for idx in np.arange(351)]

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))

# open all the files associated with this process and keep handles around (avoid metadata costs)
myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
status("%d input files" % len(myfiles))
#myhandles = [None]*len(myfiles) 
myhandles = [None]*int(math.ceil(float(len(filelist))/numProcs))
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r")

reportbarrier("Finished opening all files")

# to do efficient collective IO, note that we want to make sure all the
# processes are writing to the same contiguous blocks of rows
numtimeslices = 8 # each variable comes in 8 time slices, and one time slice becomes one column
allcoloffsets = np.arange(0, numtimeslices*len(filelist), numtimeslices)
mycoloffsets = [coloffset for (index, coloffset) in enumerate(allcoloffsets) if (index % numProcs == rank)]

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

foutname = "superstrided/atmosphere.hdf5"
#Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
propfaid.set_fapl_mpio(MPI.COMM_WORLD, mpi_info)
propfaid.set_alignment(1024*1024*1024, 1024*1024)
propfaid.set_sieve_buf_size(1024*1024)
fid = h5py.h5f.create(join(datapath, foutname), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
fout = h5py.File(fid)
#fout = h5py.File(join(datapath, foutname), "w", driver="mpio", comm=MPI.COMM_WORLD)

# create the rows dataset using the low-level api so I can force it to not do zero-filling, then convert to a high level object
spaceid = h5py.h5s.create_simple((numRows, numCols))
plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating file and dataset")

for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    for (hindex, fh) in enumerate(myhandles):
        reportbarrier("Writing %d/%d chunks for %s" % (hindex + 1, len(myhandles), curvar))

        if fh is None:
            # even though you want a no-op, you have to write something or hdf5 hangs waiting
            # so rewrite the previous chunk of data
            previoushindex = hindex - 1
            previousfh = myhandles[previoushindex]
            mypreviousoffset = mycoloffsets[previoushindex]

            reportbarrier("Starting to load variable chunks")
            rawvar = previousfh[curvar][:]
            reshapedrawvar = np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
            flatdata = np.ascontiguousarray(reshapedrawvar)
            reportbarrier("Finished loading variable chunks")
            reportbarrier("Starting to write out variable chunks")

            # use low-level collective writing interface b/c high-level might incur context management overhead?
            # see https://groups.google.com/forum/#!topic/h5py/wDgTjGho0dY
            dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE) 
            slab_shape = (flattenedlength, numtimeslices)
            slab_offset = (varidx*flattenedlength, mypreviousoffset)
            memory_space = h5py.h5s.create_simple(slab_shape)
            file_space = rows.id.get_space()
            file_space.select_hyperslab(slab_offset, slab_shape)
            rows.id.write(memory_space, file_space, flatdata, dxpl = dxpl)

            #with rows.collective:
            #    rows[varidx*flattenedlength:(varidx+1)*flattenedlength, mypreviousoffset:(mypreviousoffset + numtimeslices)] = reshapedrawvar

            reportbarrier("Finished writing out variable chunks")

        else:
            # get the data, ensure it is contiguous in memory
            myoffset = mycoloffsets[hindex]
            reportbarrier("Starting to load variable chunks")
            rawvar = fh[curvar][:]
            reshapedrawvar = np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
            flatdata = np.ascontiguousarray(reshapedrawvar)
            reportbarrier("Finished loading variable chunks")
            reportbarrier("Starting to write out variable chunks")

            dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE) 
            
            numrowchunks = 120 # TODO: this should evenly divide flattenedlength
            rowchunksize = flattenedlength/numrowchunks
            rowchunkoffset = 0
            while rowchunkoffset < rowchunksize*numrowchunks:
                status("Writing a chunk within this chunk")
                slab_offset = (varidx*flattenedlength + rowchunkoffset, myoffset)
                slab_shape = (rowchunksize, numtimeslices)
                memory_space = h5py.h5s.create_simple(slab_shape)
                file_space = rows.id.get_space()
                file_space.select_hyperslab(slab_offset, slab_shape)

                rows.id.write(memory_space, file_space, 
                        flatdata[rowchunkoffset*numtimeslices:((rowchunkoffset+1)*numtimeslices)], 
                        dxpl = dxpl)
                fout.id.flush()

                rowchunkoffset = rowchunkoffset + numrowchunks

            #with rows.collective:
            #    rows[varidx*flattenedlength:(varidx+1)*flattenedlength, myoffset:(myoffset + numtimeslices)] = reshapedrawvar

            reportbarrier("Finished writing out variable chunks")

            """
            myoffset = mycoloffsets[hindex]
            rawvar = fh[curvar][:]
            reshapedrawvar = np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
            startrow = varidx*flattenedlength
            endrow = (varidx+1)*flattenedlength
            startcol = myoffset
            endcol = myoffset + numtimeslices
            status("Writing chunk %d/%d of %s to chunk [%d:%d, %d;%d] of output matrix" % (hindex + 1, len(myhandles), curvar, startrow, endrow, startcol, endcol))
            rows[startrow:endrow, startcol:endcol] = reshapedrawvar
            status("Done writing chunk %d/%d of %s to chunk [%d:%d, %d;%d] of output matrix" % (hindex + 1, len(myhandles), curvar, startrow, endrow, startcol, endcol))
            """

for fh in enumerate(myhandles):
    if fh is not None:
        fh.close()
fout.close()

