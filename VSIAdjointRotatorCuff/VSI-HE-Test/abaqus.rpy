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
mdb.Job(name='TestVSI-Tets2', objectToCopy=mdb.jobs['TestVSI-Tets'])
mdb.Model(name='Model-2', objectToCopy=mdb.models['Model-1'])
#: The model "Model-2" has been created.
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.Job(name='TestVSI-Tets2-Copy', objectToCopy=mdb.jobs['TestVSI-Tets2'])
mdb.jobs['TestVSI-Tets2-Copy'].setValues(model='Model-2')
del mdb.jobs['TestVSI-Tets2']
mdb.jobs.changeKey(fromName='TestVSI-Tets2-Copy', toName='TestVSI-Tets2')
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9461, 
    farPlane=23.6393, width=11.9429, height=6.07513, viewOffsetX=-0.410194, 
    viewOffsetY=-0.204868)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.6402, 
    farPlane=23.3702, width=12.5833, height=6.40084, cameraPosition=(5.96249, 
    13.0399, 17.3115), cameraUpVector=(-0.684993, 0.443414, -0.578073), 
    cameraTarget=(2.54354, 2.5178, 2.74376), viewOffsetX=-0.432186, 
    viewOffsetY=-0.215852)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.7229, 
    farPlane=23.5263, width=12.6596, height=6.43969, cameraPosition=(14.0927, 
    5.47363, 16.7715), cameraUpVector=(-0.475183, 0.847192, -0.237628), 
    cameraTarget=(2.74783, 2.37757, 2.7597), viewOffsetX=-0.434807, 
    viewOffsetY=-0.217161)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.3892, 
    farPlane=23.7558, width=13.2743, height=6.75235, cameraPosition=(-8.88908, 
    3.57792, 17.7674), cameraUpVector=(-0.0349525, 0.85326, -0.520314), 
    cameraTarget=(2.35611, 2.4152, 3.38626), viewOffsetX=-0.455918, 
    viewOffsetY=-0.227705)
a = mdb.models['Model-2'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#8 ]', ), )
a.Set(faces=faces1, name='XMIN')
#: The set 'XMIN' has been created (1 face).
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.3426, 
    farPlane=23.8025, width=13.2313, height=6.73049, viewOffsetX=1.12737, 
    viewOffsetY=0.165863)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.8719, 
    farPlane=25.6933, width=14.6421, height=7.44815, cameraPosition=(12.5462, 
    6.30259, 20.3112), cameraUpVector=(-0.316507, 0.863746, -0.39213), 
    cameraTarget=(3.07412, 2.95131, 5.0249), viewOffsetX=1.24758, 
    viewOffsetY=0.183549)
a = mdb.models['Model-2'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#2 ]', ), )
a.Set(faces=faces1, name='XMAX')
#: The set 'XMAX' has been created (1 face).
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.8568, 
    farPlane=26.2719, width=15.5507, height=7.91032, cameraPosition=(21.151, 
    3.58381, -8.40452), cameraUpVector=(-0.455852, 0.889231, -0.0383106), 
    cameraTarget=(6.09174, 3.16802, 1.9718), viewOffsetX=1.32499, 
    viewOffsetY=0.194938)
a = mdb.models['Model-2'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#20 ]', ), )
a.Set(faces=faces1, name='ZMIN')
#: The set 'ZMIN' has been created (1 face).
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.3551, 
    farPlane=26.7198, width=15.0879, height=7.6749, cameraPosition=(14.8103, 
    7.23681, 19.6215), cameraUpVector=(-0.0657184, 0.80357, -0.591572), 
    cameraTarget=(3.19978, 3.78412, 5.914), viewOffsetX=1.28556, 
    viewOffsetY=0.189136)
a = mdb.models['Model-2'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#10 ]', ), )
a.Set(faces=faces1, name='ZMAX')
#: The set 'ZMAX' has been created (1 face).
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['Model-2'].rootAssembly
region = a.sets['XMIN']
mdb.models['Model-2'].DisplacementBC(name='BC-5', createStepName='Compression', 
    region=region, u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
a = mdb.models['Model-2'].rootAssembly
region = a.sets['XMAX']
mdb.models['Model-2'].DisplacementBC(name='BC-6', createStepName='Compression', 
    region=region, u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
a = mdb.models['Model-2'].rootAssembly
region = a.sets['ZMIN']
mdb.models['Model-2'].DisplacementBC(name='BC-7', createStepName='Compression', 
    region=region, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
a = mdb.models['Model-2'].rootAssembly
region = a.sets['ZMAX']
mdb.models['Model-2'].DisplacementBC(name='BC-8', createStepName='Compression', 
    region=region, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['TestVSI-Tets2'].submit(consistencyChecking=OFF)
#: The job input file "TestVSI-Tets2.inp" has been submitted for analysis.
#: Job TestVSI-Tets2: Analysis Input File Processor completed successfully.
#: Job TestVSI-Tets2: Abaqus/Standard completed successfully.
#: Job TestVSI-Tets2 completed successfully. 
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/TestVSI-Tets2.odb')
#: Model: C:/Users/maherlam.UMROOT/TestVSI-Tets2.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='E', outputPosition=INTEGRATION_POINT, refinement=(INVARIANT, 
    'Max. Principal'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=15 )
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.728, 
    farPlane=26.3469, width=15.4319, height=8.09377, viewOffsetX=2.25339, 
    viewOffsetY=-0.400525)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.729, 
    farPlane=26.3459, width=15.4328, height=8.09427, cameraPosition=(14.6502, 
    6.18263, 20.0227), cameraUpVector=(-0.347383, 0.862559, -0.367855), 
    cameraTarget=(3.03964, 2.72993, 6.31518), viewOffsetX=2.25352, 
    viewOffsetY=-0.400549)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.2512, 
    farPlane=26.5671, width=14.992, height=7.86309, cameraPosition=(15.3964, 
    11.7935, 17.001), cameraUpVector=(-0.572165, 0.687191, -0.447656), 
    cameraTarget=(3.3169, 3.48341, 6.06253), viewOffsetX=2.18916, 
    viewOffsetY=-0.389109)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.6384, 
    farPlane=26.561, width=15.3492, height=8.05044, cameraPosition=(17.6183, 
    8.57184, 16.8572), cameraUpVector=(-0.544708, 0.787533, -0.288245), 
    cameraTarget=(3.84459, 2.90117, 6.23873), viewOffsetX=2.24132, 
    viewOffsetY=-0.39838)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.9184, 
    farPlane=26.4547, width=15.6075, height=8.18591, cameraPosition=(17.5358, 
    5.94249, 17.8874), cameraUpVector=(-0.362234, 0.871975, -0.329312), 
    cameraTarget=(3.75143, 2.82288, 6.27343), viewOffsetX=2.27903, 
    viewOffsetY=-0.405084)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.3482, 
    farPlane=26.61, width=15.0815, height=7.91001, cameraPosition=(16.3249, 
    11.517, 16.3974), cameraUpVector=(-0.504528, 0.710595, -0.490414), 
    cameraTarget=(3.47634, 3.75856, 5.9406), viewOffsetX=2.20222, 
    viewOffsetY=-0.391431)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.4271, 
    farPlane=26.531, width=13.3903, height=7.02301, viewOffsetX=2.42685, 
    viewOffsetY=-0.482106)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.4711, 
    farPlane=26.487, width=13.4262, height=7.04182, viewOffsetX=1.92223, 
    viewOffsetY=0.0458292)
session.graphicsOptions.setValues(backgroundStyle=SOLID, 
    backgroundColor='#FFFFFF')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/TestVSI-Tets.odb')
#: Model: C:/Users/maherlam.UMROOT/TestVSI-Tets.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          6
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/TestVSI-Tets.odb'])
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9728, 
    farPlane=23.728, width=11.9676, height=6.08767, viewOffsetX=-0.0158095, 
    viewOffsetY=-0.860643)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.6198, 
    farPlane=24.081, width=17.9528, height=9.1322, viewOffsetX=-0.933997, 
    viewOffsetY=-0.293771)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.55, 
    farPlane=24.1509, width=17.8534, height=9.08165, viewOffsetX=0.486237, 
    viewOffsetY=-0.798644)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.7782, 
    farPlane=23.9226, width=14.1925, height=7.21943, viewOffsetX=0.834546, 
    viewOffsetY=-0.949508)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.8387, 
    farPlane=23.8621, width=14.2597, height=7.25363, viewOffsetX=1.30943, 
    viewOffsetY=-0.991638)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.7657, 
    farPlane=23.9352, width=16.0464, height=8.16249, viewOffsetX=1.20205, 
    viewOffsetY=-0.93116)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.7021, 
    farPlane=23.9987, width=15.9665, height=8.12184, viewOffsetX=2.7885, 
    viewOffsetY=-1.43216)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='E', outputPosition=INTEGRATION_POINT, refinement=(INVARIANT, 
    'Max. Principal'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='E', outputPosition=INTEGRATION_POINT, refinement=(INVARIANT, 
    'Min. Principal'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='E', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 
    'E11'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=HARMONIC)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.animationOptions.setValues(frameRate=28)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=HARMONIC)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.animationOptions.setValues(frameRate=48)
session.animationOptions.setValues(mode=PLAY_ONCE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.4986, 
    farPlane=23.9765, width=15.7107, height=7.99171, viewOffsetX=1.84103, 
    viewOffsetY=-1.06716)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.6374, 
    farPlane=23.8376, width=14.9321, height=7.59565, viewOffsetX=1.97164, 
    viewOffsetY=-1.08598)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.5666, 
    farPlane=23.9084, width=14.8484, height=7.55309, viewOffsetX=1.48004, 
    viewOffsetY=-1.03091)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.566, 
    farPlane=23.9091, width=14.8477, height=7.55272, viewOffsetX=2.78429, 
    viewOffsetY=-1.13862)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.566, 
    farPlane=23.9091, width=14.8477, height=7.5527, viewOffsetX=2.72544, 
    viewOffsetY=-1.19739)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.1721, 
    farPlane=23.4272, width=14.3823, height=7.316, cameraPosition=(11.3886, 
    11.6342, 14.9923), cameraUpVector=(-0.508215, 0.627922, -0.589433), 
    cameraTarget=(2.35918, 2.08948, 2.26478), viewOffsetX=2.64002, 
    viewOffsetY=-1.15986)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.3954, 
    farPlane=24.1627, width=15.8277, height=8.05124, cameraPosition=(15.0842, 
    7.10337, 15.7818), cameraUpVector=(-0.359337, 0.837072, -0.412537), 
    cameraTarget=(2.8743, 2.78035, 2.86469), viewOffsetX=2.90534, 
    viewOffsetY=-1.27642)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.5822, 
    farPlane=24.6024, width=16.0485, height=8.16352, cameraPosition=(16.3576, 
    7.1766, 14.9232), cameraUpVector=(-0.454643, 0.824654, -0.33652), 
    cameraTarget=(3.08527, 2.48637, 3.24125), viewOffsetX=2.94586, 
    viewOffsetY=-1.29422)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.5793, 
    farPlane=24.6237, width=16.045, height=8.16177, cameraPosition=(16.3874, 
    7.17884, 14.901), cameraUpVector=(-0.419659, 0.831428, -0.364161), 
    cameraTarget=(3.10677, 2.66627, 3.15862), viewOffsetX=2.94522, 
    viewOffsetY=-1.29394)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.5771, 
    farPlane=24.626, width=16.0424, height=8.16043, viewOffsetX=3.28381, 
    viewOffsetY=-1.10321)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.4504, 
    farPlane=24.5435, width=15.8927, height=8.08429, cameraPosition=(16.0215, 
    7.49421, 15.0155), cameraUpVector=(-0.408654, 0.824099, -0.392254), 
    cameraTarget=(3.02048, 2.75228, 3.05274), viewOffsetX=3.25317, 
    viewOffsetY=-1.09292)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.1876, 
    farPlane=24.4844, width=15.5822, height=7.92634, cameraPosition=(15.4346, 
    9.11552, 14.6202), cameraUpVector=(-0.467029, 0.767227, -0.439598), 
    cameraTarget=(2.88223, 2.73246, 2.94484), viewOffsetX=3.18961, 
    viewOffsetY=-1.07157)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0033, 
    farPlane=24.3992, width=15.3644, height=7.81554, cameraPosition=(14.8291, 
    10.7049, 14.0425), cameraUpVector=(-0.515734, 0.703431, -0.489085), 
    cameraTarget=(2.76887, 2.73652, 2.83206), viewOffsetX=3.14502, 
    viewOffsetY=-1.05659)
odb = session.odbs['C:/Users/maherlam.UMROOT/TestVSI-Tets2.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.8812, 
    farPlane=23.7042, width=13.4485, height=6.84097, viewOffsetX=-0.412547, 
    viewOffsetY=0.217373)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.8764, 
    farPlane=23.709, width=13.4435, height=6.83841, viewOffsetX=2.5267, 
    viewOffsetY=-0.793835)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.8709, 
    farPlane=23.7145, width=14.2954, height=7.2718, viewOffsetX=2.49495, 
    viewOffsetY=-0.747271)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0522, 
    farPlane=23.9942, width=14.4968, height=7.37424, cameraPosition=(14.1949, 
    12.5206, 12.7934), cameraUpVector=(-0.558065, 0.604193, -0.568783), 
    cameraTarget=(2.72262, 2.49008, 2.67414), viewOffsetX=2.5301, 
    viewOffsetY=-0.757798)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0852, 
    farPlane=23.9836, width=14.5335, height=7.39291, cameraPosition=(14.4295, 
    10.7446, 14.0431), cameraUpVector=(-0.468507, 0.69242, -0.548685), 
    cameraTarget=(2.71287, 2.57002, 2.61854), viewOffsetX=2.53651, 
    viewOffsetY=-0.759716)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0807, 
    farPlane=23.988, width=14.5286, height=7.39039, cameraPosition=(14.4499, 
    10.5264, 14.1783), cameraUpVector=(-0.514406, 0.694474, -0.503083), 
    cameraTarget=(2.73325, 2.35182, 2.75376), viewOffsetX=2.53565, 
    viewOffsetY=-0.759457)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0679, 
    farPlane=23.9711, width=14.5144, height=7.38318, cameraPosition=(14.3876, 
    10.4739, 14.2537), cameraUpVector=(-0.509054, 0.696936, -0.505118), 
    cameraTarget=(2.72104, 2.35551, 2.7382), viewOffsetX=2.53317, 
    viewOffsetY=-0.758715)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.Viewport(name='Viewport: 2', origin=(7.47499942779541, 
    -48.8444480895996), width=411.747894287109, height=232.400009155273)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].maximize()
session.viewports['Viewport: 1'].restore()
session.viewports['Viewport: 2'].restore()
session.viewports['Viewport: 1'].setValues(origin=(0.0, -48.8444366455078), 
    width=235.773941040039, height=239.866668701172)
session.viewports['Viewport: 2'].setValues(origin=(235.773941040039, 
    -48.8444366455078), width=235.773941040039, height=239.866668701172)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.0682, 
    farPlane=24.9709, width=12.5998, height=12.4678, viewOffsetX=2.44051, 
    viewOffsetY=-0.757351)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.1596, 
    farPlane=24.8794, width=12.6952, height=12.5623, viewOffsetX=-0.532115, 
    viewOffsetY=-0.746114)
odb = session.odbs['C:/Users/maherlam.UMROOT/TestVSI-Tets.odb']
session.viewports['Viewport: 2'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].makeCurrent()
odb = session.odbs['C:/Users/maherlam.UMROOT/TestVSI-Tets.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 2'].makeCurrent()
odb = session.odbs['C:/Users/maherlam.UMROOT/TestVSI-Tets2.odb']
session.viewports['Viewport: 2'].setValues(displayedObject=odb)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.386, 
    farPlane=24.6531, width=10.7407, height=10.6282, viewOffsetX=1.46436, 
    viewOffsetY=-0.68671)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.4691, 
    farPlane=24.5699, width=10.8129, height=10.6996, viewOffsetX=-0.00226569, 
    viewOffsetY=-1.21184)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.811, 
    farPlane=24.228, width=9.22729, height=9.13063, viewOffsetX=-0.135661, 
    viewOffsetY=-1.34478)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.3996, 
    farPlane=25.1531, width=12.169, height=12.0415, viewOffsetX=3.17678, 
    viewOffsetY=-0.826025)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.4958, 
    farPlane=25.0569, width=12.2634, height=12.1349, viewOffsetX=-1.03412, 
    viewOffsetY=-1.1932)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0673, 
    farPlane=24.4854, width=8.31625, height=8.22913, viewOffsetX=0.234811, 
    viewOffsetY=-0.856916)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.9776, 
    farPlane=24.0615, width=8.25923, height=8.17272, viewOffsetX=-0.0896844, 
    viewOffsetY=-1.37889)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.8985, 
    farPlane=24.1406, width=8.2089, height=8.12291, viewOffsetX=0.108667, 
    viewOffsetY=-0.942384)
session.viewports['Viewport: 2'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 2'].view.setValues(nearPlane=13.0964, 
    farPlane=24.1645, width=8.33485, height=8.24754, viewOffsetX=-0.849233, 
    viewOffsetY=-0.845389)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.0726, 
    farPlane=24.5001, width=8.31964, height=8.23249, viewOffsetX=-0.366512, 
    viewOffsetY=-1.04639)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 2'].view.setValues(nearPlane=13.0985, 
    farPlane=24.1624, width=8.33622, height=8.24889, viewOffsetX=-0.514584, 
    viewOffsetY=-0.800939)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
del session.viewports['Viewport: 2']
session.viewports['Viewport: 1'].maximize()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9461, 
    farPlane=23.6393, width=11.9429, height=6.07513, viewOffsetX=0.354975, 
    viewOffsetY=-0.212748)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9897, 
    farPlane=23.5957, width=12.7481, height=6.48468, viewOffsetX=0.288796, 
    viewOffsetY=-0.176757)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9371, 
    farPlane=23.6483, width=12.6964, height=6.4584, viewOffsetX=0.312784, 
    viewOffsetY=-0.234677)
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.137, 
    farPlane=26.8211, width=13.5626, height=6.899, cameraPosition=(16.33, 
    11.5019, 16.4024), cameraUpVector=(-0.508623, 0.71095, -0.485646), 
    cameraTarget=(3.48144, 3.74341, 5.94558), viewOffsetX=1.88324, 
    viewOffsetY=0.0448997)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.0549, 
    farPlane=26.8578, width=13.4936, height=6.8639, cameraPosition=(15.432, 
    13.7787, 15.5442), cameraUpVector=(-0.567425, 0.621507, -0.540146), 
    cameraTarget=(3.34266, 4.09486, 5.81296), viewOffsetX=1.87366, 
    viewOffsetY=0.0446712)
mdb.Model(name='HE-Test1', objectToCopy=mdb.models['Model-2'])
#: The model "HE-Test1" has been created.
a = mdb.models['HE-Test1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
del mdb.models['HE-Test1']
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.Model(name='HE-Unconfined', objectToCopy=mdb.models['Model-1'])
#: The model "HE-Unconfined" has been created.
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.Model(name='HE-Confined', objectToCopy=mdb.models['Model-2'])
#: The model "HE-Confined" has been created.
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
p1 = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['HE-Confined'].Material(name='Hyperelastic')
mdb.models['HE-Confined'].materials['Hyperelastic'].Hyperelastic(
    materialType=ISOTROPIC, testData=OFF, type=OGDEN, n=2, 
    volumetricResponse=VOLUMETRIC_DATA, table=((0.0090025, 45.005, -0.01297, 
    -38.78, 0.0, 0.0), ))
p1 = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['HE-Confined'].sections['Cube'].setValues(material='Hyperelastic', 
    thickness=None)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, 
    interactions=OFF, constraints=OFF, connectors=OFF, engineeringFeatures=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4H, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
p = mdb.models['HE-Confined'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
a1 = mdb.models['HE-Confined'].rootAssembly
a1.regenerate()
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF, 
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
mdb.models['HE-Confined'].steps['Compression'].setValues(nlgeom=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.Job(name='HE_Confined', model='HE-Confined', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
    multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
mdb.jobs['HE_Confined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Confined.inp" has been submitted for analysis.
#: Job HE_Confined: Analysis Input File Processor completed successfully.
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p1 = mdb.models['HE-Unconfined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
#: Error in job HE_Confined: Too many attempts made for this increment
#: Job HE_Confined: Abaqus/Standard aborted due to errors.
#: Error in job HE_Confined: Abaqus/Standard Analysis exited with an error - Please see the  message file for possible error messages if the file exists.
#: Job HE_Confined aborted due to errors.
mdb.models['HE-Unconfined'].Material(name='Hyperelastic')
mdb.models['HE-Unconfined'].materials['Hyperelastic'].Hyperelastic(
    materialType=ISOTROPIC, testData=OFF, type=OGDEN, n=2, 
    volumetricResponse=VOLUMETRIC_DATA, table=((0.0090025, 45.005, -0.01297, 
    -38.78, 0.0, 0.0), ))
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
p1 = mdb.models['HE-Unconfined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['HE-Unconfined'].materials['Hyperelastic'].hyperelastic.setValues(
    table=((0.5, 2.0, 0.1, 6.0, 0.01, 0.0), ))
p1 = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['HE-Confined'].materials['Hyperelastic'].hyperelastic.setValues(
    table=((0.5, 2.0, 0.1, 6.0, 0.01, 0.0), ))
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF, 
    constraints=OFF, connectors=OFF, engineeringFeatures=OFF, 
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, 
    adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4H, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
p = mdb.models['HE-Confined'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
p = mdb.models['HE-Unconfined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4H, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
p = mdb.models['HE-Unconfined'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
a1 = mdb.models['HE-Unconfined'].rootAssembly
a1.regenerate()
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF, 
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
mdb.models['HE-Unconfined'].steps['Compression'].setValues(nlgeom=ON)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a.regenerate()
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p1 = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['HE-Unconfined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['HE-Unconfined'].sections['Cube'].setValues(material='Hyperelastic', 
    thickness=None)
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.Job(name='HE_Unconfined', objectToCopy=mdb.jobs['HE_Confined'])
mdb.jobs['HE_Unconfined'].setValues(model='HE-Unconfined')
mdb.jobs['HE_Unconfined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Unconfined.inp" has been submitted for analysis.
#: Job HE_Unconfined: Analysis Input File Processor completed successfully.
#: Job HE_Unconfined: Abaqus/Standard completed successfully.
#: Job HE_Unconfined completed successfully. 
p1 = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['HE-Confined'].steps['Compression'].setValues(initialInc=0.0001, 
    maxInc=0.01)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.jobs['HE_Confined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Confined.inp" has been submitted for analysis.
#: Job HE_Confined: Analysis Input File Processor completed successfully.
#: Job HE_Confined: Abaqus/Standard completed successfully.
#: Job HE_Confined completed successfully. 
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.Model(name='HE-BiaxialStretch', objectToCopy=mdb.models['HE-Unconfined'])
#: The model "HE-BiaxialStretch" has been created.
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
del mdb.models['HE-BiaxialStretch']
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.Model(name='HE-BiaxialStretch', objectToCopy=mdb.models['HE-Confined'])
#: The model "HE-BiaxialStretch" has been created.
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
del mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-8']
del mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-7']
del mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-3']
del mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-4']
mdb.models['HE-BiaxialStretch'].rootAssembly.sets.changeKey(fromName='TopFace', 
    toName='YMAX')
mdb.models['HE-BiaxialStretch'].rootAssembly.sets.changeKey(
    fromName='BottomFace', toName='YMIN')
a = mdb.models['HE-Unconfined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.models['HE-Unconfined'].rootAssembly.sets.changeKey(fromName='BottomFace', 
    toName='YMIN')
mdb.models['HE-Unconfined'].rootAssembly.sets.changeKey(fromName='TopFace', 
    toName='YMAX')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
a = mdb.models['HE-Unconfined'].rootAssembly
region = a.sets['YMIN']
mdb.models['HE-Unconfined'].boundaryConditions['BC-1'].setValues(region=region)
a = mdb.models['HE-Unconfined'].rootAssembly
region = a.sets['YMAX']
mdb.models['HE-Unconfined'].boundaryConditions['BC-2'].setValues(region=region)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.models['HE-Confined'].rootAssembly.sets.changeKey(fromName='TopFace', 
    toName='YMAX')
mdb.models['HE-Confined'].rootAssembly.sets.changeKey(fromName='BottomFace', 
    toName='YMIN')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['YMIN']
mdb.models['HE-Confined'].boundaryConditions['BC-1'].setValues(region=region)
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['YMAX']
mdb.models['HE-Confined'].boundaryConditions['BC-2'].setValues(region=region)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['HE_Confined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Confined.inp" has been submitted for analysis.
mdb.jobs['HE_Unconfined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Unconfined.inp" has been submitted for analysis.
#: Job HE_Confined: Analysis Input File Processor completed successfully.
#: Job HE_Unconfined: Analysis Input File Processor completed successfully.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
#: Job HE_Unconfined: Abaqus/Standard completed successfully.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
#: Job HE_Unconfined completed successfully. 
mdb.models['HE-Confined'].boundaryConditions['BC-1'].setValues(u2=-0.25)
mdb.models['HE-Confined'].boundaryConditions['BC-2'].setValues(u2=0.25)
#: Job HE_Confined: Abaqus/Standard completed successfully.
#: Job HE_Confined completed successfully. 
del mdb.models['HE-Confined'].boundaryConditions['BC-3']
del mdb.models['HE-Confined'].boundaryConditions['BC-4']
mdb.models['HE-Confined'].boundaryConditions['BC-5'].setValues(u1=-0.25)
mdb.models['HE-Confined'].boundaryConditions['BC-6'].setValues(u1=0.25)
del mdb.models['HE-Confined'].boundaryConditions['BC-7']
del mdb.models['HE-Confined'].boundaryConditions['BC-8']
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['Corner1']
mdb.models['HE-Confined'].DisplacementBC(name='BC-7', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['Corner2']
mdb.models['HE-Confined'].DisplacementBC(name='BC-8', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF, adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.Job(name='HE_BiaxialStretch', objectToCopy=mdb.jobs['HE_Unconfined'])
mdb.jobs['HE_BiaxialStretch'].setValues(model='HE-BiaxialStretch')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['YMIN']
mdb.models['HE-Confined'].boundaryConditions['BC-1'].setValues(region=region)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['HE-Confined'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p = mdb.models['HE-Confined'].parts['Cube']
p.generateMesh()
elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4H, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
p = mdb.models['HE-Confined'].parts['Cube']
c = p.cells
cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(cells, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
a = mdb.models['HE-Confined'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
a = mdb.models['HE-BiaxialStretch'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#4 ]', ), )
a.Set(faces=faces1, name='YMIN')
#: The set 'YMIN' has been edited (1 face).
a = mdb.models['HE-BiaxialStretch'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#1 ]', ), )
a.Set(faces=faces1, name='YMAX')
#: The set 'YMAX' has been edited (1 face).
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p1 = mdb.models['HE-BiaxialStretch'].parts['Cube']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
del mdb.models['HE-BiaxialStretch'].rootAssembly.sets['YMIN']
del mdb.models['HE-BiaxialStretch'].rootAssembly.sets['YMAX']
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9461, 
    farPlane=23.6393, width=11.9429, height=6.07513, viewOffsetX=-0.149878, 
    viewOffsetY=-0.3467)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.9938, 
    farPlane=23.5916, width=11.9869, height=6.0975, cameraPosition=(12.861, 
    12.8476, 13.4753), cameraUpVector=(-0.945496, 0.322406, 0.0457397), 
    cameraTarget=(2.29974, 2.28629, 2.91397), viewOffsetX=-0.15043, 
    viewOffsetY=-0.347976)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.8761, 
    farPlane=24.2052, width=12.8009, height=6.51155, cameraPosition=(9.33813, 
    -8.83475, 16.1917), cameraUpVector=(-0.47752, 0.763831, 0.434209), 
    cameraTarget=(2.65054, 2.29297, 3.30479), viewOffsetX=-0.160645, 
    viewOffsetY=-0.371605)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#4 ]', ), )
a.Set(faces=faces1, name='YMIN')
#: The set 'YMIN' has been created (1 face).
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.599, 
    farPlane=24.0375, width=12.5453, height=6.38154, cameraPosition=(14.6663, 
    15.4285, 8.76505), cameraUpVector=(-0.764153, 0.478512, -0.432547), 
    cameraTarget=(2.68366, 3.20457, 2.31413), viewOffsetX=-0.157437, 
    viewOffsetY=-0.364185)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
f1 = a.instances['Cube-1'].faces
faces1 = f1.getSequenceFromMask(mask=('[#1 ]', ), )
a.Set(faces=faces1, name='YMAX')
#: The set 'YMAX' has been created (1 face).
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.6136, 
    farPlane=24.0229, width=12.5588, height=6.38839, viewOffsetX=-0.605542, 
    viewOffsetY=-1.45002)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.3102, 
    farPlane=24.3606, width=14.124, height=7.18458, cameraPosition=(10.2485, 
    5.81046, 20.4999), cameraUpVector=(-0.176874, 0.891639, -0.416769), 
    cameraTarget=(3.92734, 3.4479, 3.49742), viewOffsetX=-0.68101, 
    viewOffsetY=-1.63073)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.1932, 
    farPlane=24.4776, width=14.0161, height=7.12969, viewOffsetX=0.157382, 
    viewOffsetY=-1.36859)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.8293, 
    farPlane=25.1493, width=13.6804, height=6.95893, cameraPosition=(14.3467, 
    8.54023, 17.4774), cameraUpVector=(-0.291381, 0.817378, -0.496982), 
    cameraTarget=(4.18937, 3.62518, 3.07963), viewOffsetX=0.153612, 
    viewOffsetY=-1.33581)
a = mdb.models['HE-Confined'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.models['HE-Confined'].boundaryConditions['BC-1'].setValues(u2=0.0)
mdb.models['HE-Confined'].boundaryConditions['BC-2'].setValues(u2=-0.5)
mdb.models['HE-Confined'].boundaryConditions['BC-5'].setValues(u1=0.0)
mdb.models['HE-Confined'].boundaryConditions['BC-6'].setValues(u1=0.0)
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['ZMAX']
mdb.models['HE-Confined'].DisplacementBC(name='BC-9', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['HE-Confined'].rootAssembly
region = a.sets['ZMIN']
mdb.models['HE-Confined'].DisplacementBC(name='BC-10', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['HE_Confined'].submit(consistencyChecking=OFF)
#: The job input file "HE_Confined.inp" has been submitted for analysis.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['Model-2'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
#: Job HE_Confined: Analysis Input File Processor completed successfully.
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.8544, 
    farPlane=25.1243, width=13.7036, height=6.97072, viewOffsetX=-0.316792, 
    viewOffsetY=-0.470122)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.8152, 
    farPlane=25.1635, width=14.1535, height=7.1996, viewOffsetX=-0.315955, 
    viewOffsetY=-0.46888)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.818, 
    farPlane=25.1607, width=14.1562, height=7.20094, viewOffsetX=0.291747, 
    viewOffsetY=-1.03869)
session.viewports['Viewport: 1'].view.setValues(nearPlane=16.5927, 
    farPlane=25.5198, width=15.8516, height=8.06339, cameraPosition=(22.9572, 
    1.36331, 7.50987), cameraUpVector=(-0.174516, 0.944351, -0.278829), 
    cameraTarget=(5.46757, 3.08967, 2.43463), viewOffsetX=0.326688, 
    viewOffsetY=-1.16309)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.3312, 
    farPlane=25.8193, width=14.6465, height=7.45036, cameraPosition=(19.4261, 
    8.72687, 12.4527), cameraUpVector=(-0.500521, 0.822655, -0.269661), 
    cameraTarget=(4.4033, 3.95583, 3.16943), viewOffsetX=0.301851, 
    viewOffsetY=-1.07467)
#: Job HE_Confined: Abaqus/Standard completed successfully.
#: Job HE_Confined completed successfully. 
a = mdb.models['HE-BiaxialStretch'].rootAssembly
region = a.sets['YMIN']
mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-1'].setValues(
    region=region, u2=-0.25)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
region = a.sets['YMAX']
mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-2'].setValues(
    region=region, u2=0.25)
mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-5'].setValues(u1=-0.25)
mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-6'].setValues(u1=0.25)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
region = a.sets['Corner1']
mdb.models['HE-BiaxialStretch'].DisplacementBC(name='BC-7', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
region = a.sets['Corner2']
mdb.models['HE-BiaxialStretch'].DisplacementBC(name='BC-8', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
mdb.jobs['HE_BiaxialStretch'].submit(consistencyChecking=OFF)
#: The job input file "HE_BiaxialStretch.inp" has been submitted for analysis.
#: Job HE_BiaxialStretch: Analysis Input File Processor completed successfully.
#: Job HE_BiaxialStretch: Abaqus/Standard completed successfully.
#: Job HE_BiaxialStretch completed successfully. 
mdb.Model(name='HE-PlaneStrain', objectToCopy=mdb.models['HE-BiaxialStretch'])
#: The model "HE-PlaneStrain" has been created.
a = mdb.models['HE-PlaneStrain'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['HE-PlaneStrain'].rootAssembly
region = a.sets['ZMIN']
mdb.models['HE-PlaneStrain'].DisplacementBC(name='BC-9', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['HE-PlaneStrain'].rootAssembly
region = a.sets['ZMAX']
mdb.models['HE-PlaneStrain'].DisplacementBC(name='BC-10', 
    createStepName='Compression', region=region, u1=UNSET, u2=UNSET, u3=0.0, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
del mdb.models['HE-PlaneStrain'].boundaryConditions['BC-8']
mdb.models['HE-PlaneStrain'].boundaryConditions['BC-7'].setValues(u1=0.0, 
    u3=UNSET)
del mdb.models['HE-PlaneStrain'].boundaryConditions['BC-6']
del mdb.models['HE-PlaneStrain'].boundaryConditions['BC-5']
mdb.models['HE-PlaneStrain'].boundaryConditions['BC-2'].setValues(u2=0.5)
mdb.models['HE-PlaneStrain'].boundaryConditions['BC-1'].setValues(u2=0.0)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.Job(name='HE_PlaneStrain', objectToCopy=mdb.jobs['HE_BiaxialStretch'])
mdb.jobs['HE_PlaneStrain'].setValues(model='HE-PlaneStrain')
mdb.jobs['HE_PlaneStrain'].submit(consistencyChecking=OFF)
#: The job input file "HE_PlaneStrain.inp" has been submitted for analysis.
#: Job HE_PlaneStrain: Analysis Input File Processor completed successfully.
#: Job HE_PlaneStrain: Abaqus/Standard completed successfully.
#: Job HE_PlaneStrain completed successfully. 
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.Model(name='HE-SimpleShear', objectToCopy=mdb.models['HE-BiaxialStretch'])
#: The model "HE-SimpleShear" has been created.
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF, 
    constraints=OFF, connectors=OFF, engineeringFeatures=OFF, 
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Compression')
mdb.models['HE-SimpleShear'].boundaryConditions['BC-1'].setValues(u2=0.0)
mdb.models['HE-SimpleShear'].boundaryConditions['BC-7'].setValues(u1=0.0)
del mdb.models['HE-SimpleShear'].boundaryConditions['BC-6']
del mdb.models['HE-SimpleShear'].boundaryConditions['BC-5']
mdb.models['HE-SimpleShear'].boundaryConditions['BC-2'].setValues(u1=0.5, 
    u2=UNSET)
mdb.models['HE-SimpleShear'].boundaryConditions['BC-2'].setValues(u2=0.0)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.Job(name='HE_SimpleShear', objectToCopy=mdb.jobs['HE_BiaxialStretch'])
mdb.jobs['HE_SimpleShear'].setValues(model='HE-SimpleShear')
mdb.jobs['HE_SimpleShear'].submit(consistencyChecking=OFF)
#: The job input file "HE_SimpleShear.inp" has been submitted for analysis.
#: Job HE_SimpleShear: Analysis Input File Processor completed successfully.
#: Job HE_SimpleShear: Abaqus/Standard completed successfully.
#: Job HE_SimpleShear completed successfully. 
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_BiaxialStretch.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_BiaxialStretch.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=111 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.679, 
    farPlane=23.9064, width=11.6965, height=5.94976, viewOffsetX=1.13566, 
    viewOffsetY=-0.524752)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.092, 
    farPlane=23.9858, width=13.9225, height=7.08209, cameraPosition=(21.5105, 
    4.4986, 6.83579), cameraUpVector=(-0.0871593, 0.325355, -0.941567), 
    cameraTarget=(4.00573, 3.47104, 1.62513), viewOffsetX=1.35179, 
    viewOffsetY=-0.62462)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.2396, 
    farPlane=26.0996, width=14.0587, height=7.15137, cameraPosition=(20.1481, 
    -4.12257, -6.07912), cameraUpVector=(-0.394854, 0.661701, -0.637371), 
    cameraTarget=(4.75667, 2.75145, 1.02559), viewOffsetX=1.36501, 
    viewOffsetY=-0.630731)
session.viewports['Viewport: 1'].view.setValues(nearPlane=15.051, 
    farPlane=24.4509, width=13.8847, height=7.06288, cameraPosition=(1.6073, 
    10.0941, 20.8843), cameraUpVector=(-0.198758, 0.738743, -0.644014), 
    cameraTarget=(0.110297, 3.22349, 3.99708), viewOffsetX=1.34812, 
    viewOffsetY=-0.622927)
session.viewports['Viewport: 1'].view.setValues(nearPlane=13.9351, 
    farPlane=24.7747, width=12.8553, height=6.53923, cameraPosition=(14.3103, 
    8.09257, 16.9776), cameraUpVector=(-0.385669, 0.825049, -0.412982), 
    cameraTarget=(1.52722, 3.39459, 4.76507), viewOffsetX=1.24817, 
    viewOffsetY=-0.576743)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.0129, 
    farPlane=24.6969, width=12.9271, height=6.57575, cameraPosition=(14.3191, 
    8.14107, 16.9497), cameraUpVector=(-0.366127, 0.823726, -0.432928), 
    cameraTarget=(1.53603, 3.44309, 4.73719), viewOffsetX=1.25514, 
    viewOffsetY=-0.579964)
session.viewports['Viewport: 1'].view.setValues(nearPlane=14.2311, 
    farPlane=24.5827, width=13.1284, height=6.67817, cameraPosition=(13.8665, 
    6.85253, 17.792), cameraUpVector=(-0.182988, 0.84646, -0.500022), 
    cameraTarget=(1.5874, 3.62255, 4.62335), viewOffsetX=1.27469, 
    viewOffsetY=-0.588997)
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF, 
    constraints=OFF, connectors=OFF, engineeringFeatures=OFF, 
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
v1 = a.instances['Cube-1'].vertices
verts1 = v1.getSequenceFromMask(mask=('[#10 ]', ), )
region = a.Set(vertices=verts1, name='Corner3')
mdb.models['HE-BiaxialStretch'].boundaryConditions['BC-8'].setValues(
    region=region)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['HE_BiaxialStretch'].submit(consistencyChecking=OFF)
#: The job input file "HE_BiaxialStretch.inp" has been submitted for analysis.
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_Confined.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_Confined.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
#: Job HE_BiaxialStretch: Analysis Input File Processor completed successfully.
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/HE_Confined.odb'])
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_Unconfined.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_Unconfined.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          6
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
#: Job HE_BiaxialStretch: Abaqus/Standard completed successfully.
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
#: Job HE_BiaxialStretch completed successfully. 
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/HE_Unconfined.odb'])
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_SimpleShear.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_SimpleShear.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_PlaneStrain.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_PlaneStrain.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          10
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/HE_PlaneStrain.odb'])
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_BiaxialStretch.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_BiaxialStretch.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          11
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/HE_BiaxialStretch.odb'])
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.4655, 
    farPlane=23.6213, width=15.7332, height=8.00315, viewOffsetX=-1.1974, 
    viewOffsetY=-0.371051)
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.4091, 
    farPlane=23.6778, width=15.662, height=7.96691, viewOffsetX=-0.364393, 
    viewOffsetY=-0.596702)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
a = mdb.models['HE-BiaxialStretch'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['HE-SimpleShear'].rootAssembly
v1 = a.instances['Cube-1'].vertices
verts1 = v1.getSequenceFromMask(mask=('[#10 ]', ), )
region = a.Set(vertices=verts1, name='Corner3')
mdb.models['HE-SimpleShear'].boundaryConditions['BC-8'].setValues(
    region=region)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.jobs['HE_SimpleShear'].submit(consistencyChecking=OFF)
#: The job input file "HE_SimpleShear.inp" has been submitted for analysis.
#: Job HE_SimpleShear: Analysis Input File Processor completed successfully.
#: Job HE_SimpleShear: Abaqus/Standard completed successfully.
#: Job HE_SimpleShear completed successfully. 
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_SimpleShear.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_SimpleShear.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          11
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.6014, 
    farPlane=23.5489, width=11.625, height=5.91338, viewOffsetX=0.491413, 
    viewOffsetY=-0.736297)
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_SimpleShear.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
mdb.models['HE-SimpleShear'].boundaryConditions['BC-1'].setValues(u1=0.0, 
    u3=0.0)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.Job(name='HE_SimpleShearV2', objectToCopy=mdb.jobs['HE_SimpleShear'])
mdb.jobs['HE_SimpleShearV2'].submit(consistencyChecking=OFF)
#: The job input file "HE_SimpleShearV2.inp" has been submitted for analysis.
#: Job HE_SimpleShearV2: Analysis Input File Processor completed successfully.
#: Job HE_SimpleShearV2: Abaqus/Standard completed successfully.
#: Job HE_SimpleShearV2 completed successfully. 
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/maherlam.UMROOT/HE_SimpleShear.odb'])
o3 = session.openOdb(name='C:/Users/maherlam.UMROOT/HE_SimpleShearV2.odb')
#: Model: C:/Users/maherlam.UMROOT/HE_SimpleShearV2.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       8
#: Number of Node Sets:          11
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
a = mdb.models['HE-SimpleShear'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.jobs['HE_SimpleShear'].submit(consistencyChecking=OFF)
#: The job input file "HE_SimpleShear.inp" has been submitted for analysis.
#: Job HE_SimpleShear: Analysis Input File Processor completed successfully.
#: Job HE_SimpleShear: Abaqus/Standard completed successfully.
#: Job HE_SimpleShear completed successfully. 
mdb.save()
#: The model database has been saved to "C:\Users\maherlam.UMROOT\VSITest.cae".
