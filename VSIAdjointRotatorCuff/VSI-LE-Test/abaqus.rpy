# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-13.57.30 176069
# Run by maherlam on Tue Feb 17 10:03:04 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=130.189575195312, 
    height=191.022232055664)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
openMdb(pathName='C:/Users/maherlam.UMROOT/VSITest.cae')
#: The model database "C:\Users\maherlam.UMROOT\VSITest.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['Model-1'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p = mdb.models['Model-1'].parts['Cube']
c = p.cells
pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
p.deleteMesh(regions=pickedRegions)
p = mdb.models['Model-1'].parts['Cube']
c = p.cells
pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
elemType1 = mesh.ElemType(elemCode=C3D20R)
elemType2 = mesh.ElemType(elemCode=C3D15)
elemType3 = mesh.ElemType(elemCode=C3D10)
p = mdb.models['Model-1'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
p = mdb.models['Model-1'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
p = mdb.models['Model-1'].parts['Cube']
p.generateMesh()
a = mdb.models['Model-1'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
mdb.jobs.changeKey(fromName='TestVSI-1', toName='TestVSI-Tets')
mdb.models['Model-1'].rootAssembly.sets.changeKey(fromName='Corner 2', 
    toName='Corner2')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
a = mdb.models['Model-1'].rootAssembly
region = a.sets['Corner2']
mdb.models['Model-1'].boundaryConditions['BC-4'].setValues(region=region)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['TestVSI-Tets'].submit(consistencyChecking=OFF)
#: The job input file "TestVSI-Tets.inp" has been submitted for analysis.
#: Job TestVSI-Tets: Analysis Input File Processor completed successfully.
#: Job TestVSI-Tets: Abaqus/Standard completed successfully.
#: Job TestVSI-Tets completed successfully. 
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p1 = mdb.models['Model-1'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['Model-1'].materials['Linear Elastic'].elastic.setValues(table=((
    10.0, 0.3), ))
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.jobs['TestVSI-Tets'].submit(consistencyChecking=OFF)
#: The job input file "TestVSI-Tets.inp" has been submitted for analysis.
#: Job TestVSI-Tets: Analysis Input File Processor completed successfully.
#: Job TestVSI-Tets: Abaqus/Standard completed successfully.
#: Job TestVSI-Tets completed successfully. 
