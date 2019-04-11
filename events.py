"""
The Events module contains utilities to manage runtime hierarchy of objects.
"""
from collections import  OrderedDict
class EventManager(object):
    __slots__=('ActivatedEvents', 'event_ids', 'event_callbacks')
    def __init__(self):
        super().__init__()
        self.event_ids={}
        self.event_callbacks={}
        self.createNewEventId(0, 'Start')
        self.createNewEventId(1, 'Quit')
    def Bind(self, name, func):
        self.event_ids[name].addCallback(func)
    def createNewEventId(self, ID, name):
        check_sentinal=False
        for key, value in self.event_ids.items():
            if name==value.name and ID==value.id:
                check_sentinal=True
        if not check_sentinal:
            self.event_ids[name]=Event(ID, name)
    def eventNotify(self, name):
        self.event_ids[name].doCallbacks()
class Event(object):
    __slots__=('id', 'name', 'callbacks')
    def __init__(self, ID, name):
        self.id=ID
        self.name=name
        self.callbacks=OrderedDict()
    def addCallback(self, func):
        self.callbacks[func]=1
    def delCallback(self, func):
        del self.callbacks[func]
    def doCallbacks(self):
        for func in self.callbacks:
            func()
