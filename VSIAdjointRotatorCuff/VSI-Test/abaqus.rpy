# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-13.57.30 176069
# Run by maherlam on Tue Feb  3 15:42:22 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.23958, 1.24074), width=182.467, 
    height=123.081)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('export_cube_from_odb.py', __main__.__dict__)
#* SyntaxError: ('invalid syntax', ('export_cube_from_odb.py', 35, 45, '        
#* f.write(f"{n},{u[0]},{u[1]},{u[2]}\\n")\n'))
