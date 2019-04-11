"""
The View module contains utilities to manage runtime hierarchy of objects.
"""
import wx
import wxWrapper as xet
class View(object):
    """
    Base object inteded to be inherited from

    :param parent: The parent of the object
    :type parent: Object or None
    """
    __slots__=('controller', 'windows')
    def __init__(self, controller):
        super().__init__()
        #:The parent object of the object
        #:
        #::rtype: Object or None
        self.controller=controller
        self.windows=[]
        mainFrame=MainWindow(self.controller)
        self.windows.append(mainFrame)
class MainWindow(xet.Window):
    __slots__=('controller')
    def __init__(self, controller, parent=None, id=-1, title='Window', pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Window"):
        super().__init__(None)
        self.controller=controller
        self.createMenuBar()
        self.createMenu('File')
        self.createMenuItem(self.menus[0], wx.ID_NEW, 'New', 'New', wx.ITEM_NORMAL, None)
        self.MainPanel.createButton('Preprocess')
        self.MainPanel.createListCtrl()
        listCtrl=self.MainPanel.Controls[1]
        listCtrl.InsertColumn(0, 'Date/Time')
        listCtrl.InsertColumn(1, 'Hs')
        listCtrl.InsertColumn(2, 'Hmax')
        listCtrl.InsertColumn(3, 'Tz')
        listCtrl.InsertColumn(4, 'Tp')
        listCtrl.InsertColumn(5, 'Dir_Tp TRUE')
        listCtrl.InsertColumn(6, 'SST')
        self.Show(True)
        self.Layout()
    def updateWavedataListCtrl(self):
        wavedata=self.controller.model.wavedata
        listCtrl=self.MainPanel.Controls[1]
        for index, row in wavedata.head(100).iterrows():
            listCtrl.InsertStringItem(index, str(row[0]))
            listCtrl.SetStringItem(index, 1, str(row[1]))
            listCtrl.SetStringItem(index, 2, str(row[2]))
            listCtrl.SetStringItem(index, 3, str(row[3]))
            listCtrl.SetStringItem(index, 4, str(row[4]))
            listCtrl.SetStringItem(index, 5, str(row[5]))
            listCtrl.SetStringItem(index, 6, str(row[6]))
