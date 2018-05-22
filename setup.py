#from distutils.core import setup
from setuptools import setup, find_packages


setup(
        name = "cbrain",
        version = "0.0.1",
        author = "Cooper",
        author_email = "davidontech@gmail.com",
        description = (""),
        license = "BSD",
        keywords = "",
        url = "",
#        packages=['cbrain/'],
        packages=find_packages(),
        long_description='README',
        classifiers=[ ],
)
