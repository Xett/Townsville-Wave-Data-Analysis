import wx

class Window(wx.Frame):
    """
    #
    """
    __slots__=('controller', 'MenuBar', 'MainPanel', 'MainSizer', 'Panels', 'Sizers', 'menus', 'menuItems')
    def __init__(self, parent, id=-1, title="Window", pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="Window"):
        super().__init__(parent, id, title, pos, size, style, name)
        self.menus=[]
        self.Panels=[]
        self.Sizers=[]
        self.menuItems=[]
        self.MenuBar=None
        self.MainSizer=None
        self.MainPanel=None
        self.addSizer()
        self.createPanel(self)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
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
            self.SetSizer(self.MainSizer)
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
    def addButton(self, button, proportion=1, flags=wx.EXPAND):
        self.Controls.append(button)
        self.MainSizer.Add(button, proportion, flags)
    def createButton(self, label, proportion=1, flags=wx.EXPAND):
        button=wx.Button(self, -1, label)
        self.addButton(button, proportion, flags)
    def addListCtrl(self, listCtrl, proportion=1, flags=wx.EXPAND):
        self.Controls.append(listCtrl)
        self.MainSizer.Add(listCtrl, proportion, flags)
    def createListCtrl(self, proportion=1, flags=wx.EXPAND):
        """
        #
        """
        listCtrl=wx.ListCtrl(self, style=wx.LC_REPORT|wx.BORDER_SUNKEN)
        self.addListCtrl(listCtrl, proportion, flags)
    def addSizer(self):
        """
        #
        """
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        if self.MainSizer==None:
            self.MainSizer=sizer
            self.SetSizer(self.MainSizer)
        else:
            self.Sizers.append(sizer)
