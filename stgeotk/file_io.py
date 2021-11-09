import os
import numpy as np
import re
import sys
import vtk

from . utility import *

from vtk.util.numpy_support import vtk_to_numpy
from io import StringIO


class StreamCapturer(list):

    def __enter__(self):
        # redirect the system stream
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout.flush()
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up memory
        sys.stdout = self._stdout


class VTKUnstructuredGridExtractor:
    '''
    Extract VTKUnstructutredGrid from a file or from 
    data input in Paraview Programmable Filter.
    '''

    def __init__(self, finput):
        '''
        Initialize the PointData with filename or with 
        '''
        if isinstance(finput, str):
            # initialize a reader to extract data from particle output file
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(finput)
            reader.Update()

            # if the file does not exist, redirect warning
            with StreamCapturer() as output:
                print(reader.GetOutput())

            point_number_entry = next(
                (x for x in output if "Number Of Points" in x), None)
            if (point_number_entry is None):
                raise RuntimeError("Number of Points entry not found")

            n_point = int(re.search(r"\d+", point_number_entry).group(0))
            if n_point == 0:
                raise RuntimeError(
                    f"File {finput} contains no point data or the file does not exist")
            self.grid_data = reader.GetOutput()
            log_info(f"Unstructured grid data is imported from file {finput}.")

        else:
            self.grid_data = finput
            log_info(f"Unstructured grid data is imported vtkUnstructuredGrid.")

    def get_dataset(self, dataset_name):
        '''
        Get the the named dataset vector and put in np.array
        '''
        point_data = self.grid_data.GetPointData()
        try:
            dataset = point_data.GetArray(dataset_name)

            if isinstance(dataset, np.ndarray):
                return dataset
            else:
                dataset_np = vtk_to_numpy(dataset)
                if len(dataset_np.shape) > 1 and dataset_np.shape[1] == 9:
                    n_entries = dataset_np.shape[0]
                    # note that tensor is column-major (Fortran) order
                    return dataset_np.reshape((n_entries, 3, 3), order='F')
                return dataset_np
        except:
            raise RuntimeError(
                f"Failed to extract the data set {dataset_name}")

    def get_xyz(self):
        try:
            point_coordinates = vtk_to_numpy(self.grid_data.GetPoints())
        except AttributeError:
            point_coordinates = vtk_to_numpy(
                self.grid_data.GetPoints().GetData())
        return point_coordinates
