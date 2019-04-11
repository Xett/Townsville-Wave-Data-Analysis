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
    def __init__(self, controller, parent=None, id=-1, title='Window', pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Window"):
        super().__init__(None)
        self.controller=controller
        self.createSizer()
        self.createPanel(self)
        self.createMenuBar()
        self.createMenu('File')
        self.createMenuItem(self.menus[0], wx.ID_NEW, 'New', 'New', wx.ITEM_NORMAL, None)
        self.ButtonPanel=ButtonPanel(self.MainPanel)
        self.MainPanel.MainSizer.Add(self.ButtonPanel, 1, wx.EXPAND)
        self.ButtonPanel.createButton('Preprocess', 1, wx.SHAPED)
        self.WavedataCtrl=WavedataCtrl(self.MainPanel)
        self.MainPanel.MainSizer.Add(self.WavedataCtrl.MainPanel, 1, wx.EXPAND)
        listCtrl=self.WavedataCtrl.listCtrl
        listCtrl.InsertColumn(0, 'Date/Time')
        listCtrl.InsertColumn(1, 'Hs')
        listCtrl.InsertColumn(2, 'Hmax')
        listCtrl.InsertColumn(3, 'Tz')
        listCtrl.InsertColumn(4, 'Tp')
        listCtrl.InsertColumn(5, 'Dir_Tp TRUE')
        listCtrl.InsertColumn(6, 'SST')
        self.Show(True)
        self.Layout()
class ButtonPanel(xet.Panel):
    def __init__(self, parent):
        super().__init__(parent)
class ListCtrlPanel(xet.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.MainSizer=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.MainSizer)
class ListCtrlButtonsPanel(xet.Panel):
    def __init__(self, parent):
        super().__init__(parent)
class WavedataCtrl:
    def __init__(self, parent, controller):
        self.controller=controller
        self.MainPanel=xet.Panel(parent)
        self.MainPanel.MainSizer=wx.BoxSizer(wx.VERTICAL)
        self.MainPanel.SetSizer(self.MainPanel.MainSizer)
        self.ButtonsPanel=xet.Panel(self.MainPanel)
        self.listCtrl=wx.ListCtrl(self.MainPanel, style=wx.LC_REPORT|wx.BORDER_SUNKEN)
        self.MainPanel.MainSizer.Add(self.ButtonsPanel, 1, wx.EXPAND)
        self.MainPanel.MainSizer.Add(self.listCtrl, 1, wx.EXPAND)
        self.ButtonsPanel.createButton('<<')
        self.ButtonsPanel.createButton('<')
        self.ButtonsPanel.createButton('>')
        self.ButtonsPanel.createButton('>>')
        self.length=0
        self.current=0
        self.chunkSize=0
    def update(self):
        wavedata=self.controller.model.wavedata
        listCtrl=self.WavedataCtrl.listCtrl
        for index, row in wavedata.head(10000).iterrows():
            listCtrl.InsertItem(index, str(row[0]))
            listCtrl.SetItem(index, 1, str(row[1]))
            listCtrl.SetItem(index, 2, str(row[2]))
            listCtrl.SetItem(index, 3, str(row[3]))
            listCtrl.SetItem(index, 4, str(row[4]))
            listCtrl.SetItem(index, 5, str(row[5]))
            listCtrl.SetItem(index, 6, str(row[6]))
