from stgeotk.dataset import LineData, PlaneData
from stgeotk.stereonet import LinePlot, Stereonet, ContourPlot, PlanePlot
from stgeotk.kinematics import *
from stgeotk.stereomath import *
from stgeotk.contouring import ContourData
from stgeotk.statistics import *
from stgeotk.utility import second_to_myr, meter_per_second_to_cm_per_year

# do not import file io
vtk_exists = True
try:
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError as e:
    vtk_exists = False
    print("FileIO module will not be loaded due to missing vtk package.")

if vtk_exists:
    from stgeotk.file_io import *


    

