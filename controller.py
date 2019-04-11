"""
The Controller module contains utilities to manage runtime hierarchy of objects.
"""
import wx
import logging
import datetime
from events import EventManager
from view import View
from model import Model
import os
class Controller(object):
    """
    Base object inteded to be inherited from

    :param parent: The parent of the object
    :type parent: Object or None
    """
    __slots__=('parent', 'eventManager', 'view', 'model', 'dir', 'App', 'rootLogger', 'fileHandler', 'consoleHandler', 'formatter', 'directory', 'running')
    def __init__(self, parent=None):
        super().__init__()
        #:The parent object of the object
        #:
        #::rtype: Object or None
        self.App=wx.App()
        self.parent=parent
        self.running=True
        self.directory=os.getcwd()
        self.eventManager=EventManager()
        self.eventManager.createNewEventId(2, 'Wavedata updated')
        self.loggingInit()
        self.view=View(self)
        self.model=Model(self, self.directory)
        self.view.windows[0].MainPanel.Controls[0].Bind(wx.EVT_BUTTON, self.model.preprocess)
        self.eventManager.Bind('Wavedata updated', self.view.windows[0].updateWavedataListCtrl)
        self.main()
    def main(self):
        while self.running:
            self.App.MainLoop()
    def loggingInit(self):
        self.rootLogger=logging.getLogger()
        self.formatter      =   logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        t                   =   datetime.datetime.now()
        self.fileHandler    =   logging.FileHandler(self.directory+'[{}-{}-{}]-Log.log'.format(t.year, t.month, t.day))
        self.fileHandler.setFormatter(self.formatter)
        self.rootLogger.addHandler(self.fileHandler)
        self.consoleHandler =   logging.StreamHandler()
        self.consoleHandler.setFormatter(self.formatter)
        self.rootLogger.addHandler(self.consoleHandler)
        self.rootLogger.setLevel(logging.DEBUG)
    def log(self, msg, *args, **kw):
        """
        Wrapper function to manage logging

        :param msg: The message to be logged
        :type msg: string
        :param logger: The logger to log to. If None is passed, the root logger will be used.
        :type logger: logging.logger
        :param level: The level for the logger to log to. If None is passed, the level will be the currently set level.
        :type level: integer
        :param *args: Other arguments to be passed to the log function
        :param **kw: Other keywords to be passed to the log function
        """
        self.rootLogger.log(logging.DEBUG, msg, *args, **kw)
if __name__=="__main__":
    c=Controller()
    c.main()
    del c
