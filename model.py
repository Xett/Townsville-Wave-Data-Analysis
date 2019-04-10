"""
The Model module contains utilities to manage runtime hierarchy of objects.
"""
import pandas as pd
class Model(object):
    """
    Base object inteded to be inherited from

    :param parent: The parent of the object
    :type parent: Object or None
    """
    __slots__=('controller', 'directory', 'wavedata')
    def __init__(self, controller, directory):
        super().__init__()
        #:The parent object of the object
        #:
        #::rtype: Object or None
        self.controller=controller
        self.directory=directory
    def preprocess(self):
        self.wavedata=pd.read_csv(self.directory+'/data files/townsville-wavedata-1975-2019.csv', low_memory=False)
        self.controller.log(self.wavedata)
