"""
The View module contains utilities to manage runtime hierarchy of objects.
"""
import wx
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
        mainFrame=Window(controller=self.controller, parent=None, id=-1, title='Window', pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Window")
        self.windows.append(mainFrame)
        mainFrame.createMenuBar()
        mainFrame.createMenu('File')
        mainFrame.createMenuItem(mainFrame.menus[0], wx.ID_NEW, 'New', 'New', wx.ITEM_NORMAL, None)
class Window(wx.Frame):
    """
    #
    """
    __slots__=('controller', 'MenuBar', 'MainPanel', 'MainSizer', 'Panels', 'Sizers', 'menus', 'menuItems')
    def __init__(self, controller, parent, id=-1, title="Window", pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Window"):
        super().__init__(parent, id, title, pos, size, style, name)
        self.controller=controller
        self.menus=[]
        self.Panels=[]
        self.Sizers=[]
        self.menuItems=[]
        self.MenuBar=None
        self.MainSizer=None
        self.MainPanel=None
        self.addSizer()
        self.createPanel(self)
        self.MainPanel.createButton('Preprocess')
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Show(True)
        self.Layout()
    def createMenuBar(self, style=0):
        """
        #
        """
        if self.MenuBar==None:
            self.MenuBar=wx.MenuBar(style)
            self.SetMenuBar(self.MenuBar)
    def createMenu(self, label, style=0):
        """
        #
        """
        if self.MenuBar!=None:
            menu=wx.Menu(style)
            self.MenuBar.Append(menu, label)
            self.menus.append(menu)
    def createMenuItem(self, parentMenu, id, text="", helpString="", kind=wx.ITEM_NORMAL, subMenu=None):
        """
        #
        """
        menuItem=wx.MenuItem(parentMenu, id, text, helpString, kind, subMenu)
        self.menuItems.append(menuItem)
        parentMenu.Append(menuItem)
    def createPanel(self, parent):
        """
        #
        """
        panel=Panel(self)
        self.MainSizer.Add(panel, 1, wx.EXPAND)
        if self.MainPanel==None:
            self.MainPanel=panel
        else:
            self.Panels.append(panel)
    def addSizer(self):
        """
        #
        """
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        if self.MainSizer==None:
            self.MainSizer=sizer
        else:
            self.Sizers.append(sizer)
    def OnClose(self, event):
        """
        #
        """
        self.controller.running=False
        self.Destroy()
class Panel(wx.Panel):
    """
    #
    """
    __slots__=('MainSizer', 'Panels', 'Controls')
    def __init__(self, parent, id=-1, pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Panel"):
        super().__init__(parent, id, pos, size, style, name)
        self.MainSizer=None
        self.Panels=[]
        self.Controls=[]
        self.addSizer()
    def createButton(self, label):
        """
        #
        """
        button=wx.Button(self, -1, label)
        self.Controls.append(button)
        self.MainSizer.Add(button, 1, wx.EXPAND)
    def addSizer(self):
        """
        #
        """
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        if self.MainSizer==None:
            self.MainSizer=sizer
        else:
            self.Sizers.append(sizer)
