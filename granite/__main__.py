import os
import sys
import multiprocessing as mp

mp.freeze_support()

os.environ['QT_API'] = 'pyside2'

from qtpy import QtWidgets
from app import GraniteApp

os.environ['QT_API'] = 'pyside2'

app = QtWidgets.QApplication(sys.argv)
app.setStyle('fusion')
ex = GraniteApp()
ex.show()
app.exec_()