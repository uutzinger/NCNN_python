# from setuptools import setup
from distutils.core import setup
from Cython.Build import cythonize
# setup(
#     ext_modules = cythonize(["utils.py", 
#                              "align.py", 
#                              "blazeface.py", 
#                              "retinaface.py",
#                              "scrfd.py",
#                              "arcface.py",
#                              "blur.py",
#                              "handpose.py",
#                              "ultraface.py",
#                              "yolo7.py"],
#                              language_level = "3"),
# )
setup(
    ext_modules = cythonize(["utils_object.py",
                             "utils_cnn.py",
                             "utils_image.py",
                             "utils_hand.py",
                             "blazeperson.py",
                             "blazehandpose.py",
                             "blazepose.py",
                             "blazepalm.py",
                             "live.py",
                             "blur.py",
                             "retinaface.py"],
                             language_level = "3"),
)

# py -3 setup.py build_ext --inplace
