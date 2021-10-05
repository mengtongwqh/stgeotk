import os
import numpy as np
import re
import sys
import vtk

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

    def __init__(self, file_name):
        # initialize a reader to extract data from particle output file
        self.reader = vtk.vtkXMLUnstructuredGridReader()
        self.reader.SetFileName(file_name)
        self.reader.Update()

        # # if the file does not exist...
        with StreamCapturer() as output:
            print(self.reader.GetOutput())

        point_number_entry = next(
            (x for x in output if "Number Of Points" in x), None)
        if (point_number_entry is None):
            raise RuntimeError("Number of Points entry not found")

        n_point = int(re.search(r"\d+", point_number_entry).group(0))
        if n_point == 0:
            raise RuntimeError(
                "File {0} contains no point data or the file does not exist".format(file_name))

    def get_dataset(self, dataset_name):
        """
        Get the the named dataset vector and put in np.array
        """
        point_data = self.reader.GetOutput().GetPointData()
        try:
            return vtk_to_numpy(point_data.GetArray(dataset_name))
        except:
            raise RuntimeError(
                "failed to extract the data set {0}". format(dataset_name))

    def get_xyz(self):
        point_coordinates = self.reader.GetOutput().GetPoints().GetData()
        return vtk_to_numpy(point_coordinates)


if __name__ == "__main__":
    extractor = VTKUnstructuredGridExtractor("test_ptcl.vtu")
    extractor.get_dataset("max_stretching")
