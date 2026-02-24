1

   Abaqus 2022                                  Date 23-Feb-2026   Time 12:11:43
   For use by UNIVERSITY OF MICHIGAN under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-04
          AND A TOTAL TIME PERIOD OF                            1.00    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-09
          THE MAXIMUM TIME INCREMENT ALLOWED IS                1.000E-02

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR PR. FLUX 
     CRITERION FOR RESIDUAL PR. FLUX  FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR VOL. LM. CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE PR. FLUX                            1.000E-02
     AVERAGE PR. FLUX  IS TIME AVERAGE PR. FLUX 
     ALTERNATE CRIT. FOR RESIDUAL PR. FLUX  FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO PR. FLUX  RELATIVE TO TIME AVRG. PR. FLUX      1.000E-05
     CRITERION FOR RESIDUAL PR. FLUX  WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR VOL. LM. CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL PR. FLUX  FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO PR. FLUX  REL. TO TIME AVRG. MAX. PR. FLUX     1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                   10000

          LARGE DISPLACEMENT THEORY WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     0.758    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.353E-08   TIME AVG. FORCE       1.000E-02
 LARGEST RESIDUAL FORCE            -2.051E-11   AT NODE        386   DOF  2
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-05   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        5.000E-05   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
          THERE IS ZERO FORCE     EVERYWHERE
 

 ***WARNING: THERE IS ZERO FORCE EVERYWHERE IN THE MODEL BASED ON THE DEFAULT 
             CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE FORCE DURING THE 
             CURRENT ITERATION TO VERIFY THAT THE FORCE IS SMALL ENOUGH TO BE 
             TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION CONTROLS TO RESET 
             THE CRITERION FOR ZERO FORCE.

 AVERAGE PR. FLUX                   3.394E-07   TIME AVG. PR. FLUX    3.394E-07
 LARGEST RESIDUAL PR. FLUX         -2.528E-09   AT NODE        986
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.172E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     4.172E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
          PR. FLUX  EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.353E-08   TIME AVG. FORCE       1.000E-02
 LARGEST RESIDUAL FORCE             7.920E-17   AT NODE        749   DOF  3
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-05   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        5.390E-11   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THERE IS ZERO FORCE     EVERYWHERE
 

 ***WARNING: THERE IS ZERO FORCE EVERYWHERE IN THE MODEL BASED ON THE DEFAULT 
             CRITERION. PLEASE CHECK THE VALUE OF THE AVERAGE FORCE DURING THE 
             CURRENT ITERATION TO VERIFY THAT THE FORCE IS SMALL ENOUGH TO BE 
             TREATED AS ZERO. IF NOT, PLEASE USE THE SOLUTION CONTROLS TO RESET 
             THE CRITERION FOR ZERO FORCE.

 AVERAGE PR. FLUX                   3.394E-07   TIME AVG. PR. FLUX    3.394E-07
 LARGEST RESIDUAL PR. FLUX         -3.739E-15   AT NODE        684
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.172E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     1.931E-12   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   2 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  2 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  1.000E-04
 STEP TIME COMPLETED       1.000E-04,  TOTAL TIME COMPLETED        1.000E-04


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.871E-07   TIME AVG. FORCE       1.871E-07
 LARGEST RESIDUAL FORCE            -8.792E-17   AT NODE        959   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-05   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-10   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   6.789E-07   TIME AVG. PR. FLUX    5.091E-07
 LARGEST RESIDUAL PR. FLUX          4.917E-15   AT NODE       1035
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.172E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.862E-12   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    1.500E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  2.000E-04
 STEP TIME COMPLETED       2.000E-04,  TOTAL TIME COMPLETED        2.000E-04


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.500E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.274E-07   TIME AVG. FORCE       2.572E-07
 LARGEST RESIDUAL FORCE             8.066E-17   AT NODE        650   DOF  3
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         7.500E-05   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        2.021E-10   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.188E-06   TIME AVG. PR. FLUX    7.354E-07
 LARGEST RESIDUAL PR. FLUX         -3.313E-15   AT NODE        735
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      6.259E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     7.242E-12   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    2.250E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.500E-04,  FRACTION OF STEP COMPLETED  3.500E-04
 STEP TIME COMPLETED       3.500E-04,  TOTAL TIME COMPLETED        3.500E-04


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.250E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.378E-07   TIME AVG. FORCE       3.508E-07
 LARGEST RESIDUAL FORCE            -6.848E-17   AT NODE        641   DOF  3
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         1.125E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        4.548E-10   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.952E-06   TIME AVG. PR. FLUX    1.040E-06
 LARGEST RESIDUAL PR. FLUX          4.702E-15   AT NODE        750
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      9.388E-07   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     1.629E-11   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    3.375E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.250E-04,  FRACTION OF STEP COMPLETED  5.750E-04
 STEP TIME COMPLETED       5.750E-04,  TOTAL TIME COMPLETED        5.750E-04


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.375E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.535E-07   TIME AVG. FORCE       4.765E-07
 LARGEST RESIDUAL FORCE            -7.727E-17   AT NODE        669   DOF  2
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         1.688E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.023E-09   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.097E-06   TIME AVG. PR. FLUX    1.451E-06
 LARGEST RESIDUAL PR. FLUX         -3.704E-15   AT NODE       1362
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      1.408E-06   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.666E-11   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    5.062E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.375E-04,  FRACTION OF STEP COMPLETED  9.125E-04
 STEP TIME COMPLETED       9.125E-04,  TOTAL TIME COMPLETED        9.125E-04


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.062E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.327E-06   TIME AVG. FORCE       6.466E-07
 LARGEST RESIDUAL FORCE             7.595E-17   AT NODE       1281   DOF  2
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         2.531E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        2.302E-09   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   4.816E-06   TIME AVG. PR. FLUX    2.012E-06
 LARGEST RESIDUAL PR. FLUX         -4.921E-15   AT NODE        682
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      2.112E-06   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     8.248E-11   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    7.594E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.062E-04,  FRACTION OF STEP COMPLETED  1.419E-03
 STEP TIME COMPLETED       1.419E-03,  TOTAL TIME COMPLETED        1.419E-03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-04
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.037E-06   TIME AVG. FORCE       8.784E-07
 LARGEST RESIDUAL FORCE             7.337E-17   AT NODE        620   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         3.797E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        5.180E-09   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   7.393E-06   TIME AVG. PR. FLUX    2.781E-06
 LARGEST RESIDUAL PR. FLUX         -3.775E-15   AT NODE       1048
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.168E-06   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     1.856E-10   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    1.139E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-04,  FRACTION OF STEP COMPLETED  2.178E-03
 STEP TIME COMPLETED       2.178E-03,  TOTAL TIME COMPLETED        2.178E-03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.139E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.103E-06   TIME AVG. FORCE       1.196E-06
 LARGEST RESIDUAL FORCE             8.659E-17   AT NODE        436   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.695E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.166E-08   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.126E-05   TIME AVG. PR. FLUX    3.840E-06
 LARGEST RESIDUAL PR. FLUX          5.161E-15   AT NODE        612
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.752E-06   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     4.175E-10   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    1.709E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.139E-03,  FRACTION OF STEP COMPLETED  3.317E-03
 STEP TIME COMPLETED       3.317E-03,  TOTAL TIME COMPLETED        3.317E-03


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.709E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.701E-06   TIME AVG. FORCE       1.634E-06
 LARGEST RESIDUAL FORCE            -3.216E-16   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         8.543E-04   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        2.623E-08   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.706E-05   TIME AVG. PR. FLUX    5.309E-06
 LARGEST RESIDUAL PR. FLUX         -1.161E-14   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      7.127E-06   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     9.392E-10   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    2.563E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.709E-03,  FRACTION OF STEP COMPLETED  5.026E-03
 STEP TIME COMPLETED       5.026E-03,  TOTAL TIME COMPLETED        5.026E-03


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.563E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.098E-06   TIME AVG. FORCE       2.241E-06
 LARGEST RESIDUAL FORCE            -1.648E-15   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         1.281E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        5.901E-08   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.576E-05   TIME AVG. PR. FLUX    7.354E-06
 LARGEST RESIDUAL PR. FLUX         -5.765E-14   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      1.069E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     2.113E-09   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT
          TIME INCREMENT MAY NOW INCREASE TO    3.844E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.563E-03,  FRACTION OF STEP COMPLETED  7.589E-03
 STEP TIME COMPLETED       7.589E-03,  TOTAL TIME COMPLETED        7.589E-03


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.844E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.069E-05   TIME AVG. FORCE       3.086E-06
 LARGEST RESIDUAL FORCE            -8.374E-15   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         1.922E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.328E-07   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.881E-05   TIME AVG. PR. FLUX    1.021E-05
 LARGEST RESIDUAL PR. FLUX         -2.922E-13   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      1.603E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     4.752E-09   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED
          TIME INCREMENT MAY NOW INCREASE TO    5.767E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.844E-03,  FRACTION OF STEP COMPLETED  1.143E-02
 STEP TIME COMPLETED       1.143E-02,  TOTAL TIME COMPLETED        1.143E-02


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.767E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.609E-05   TIME AVG. FORCE       4.268E-06
 LARGEST RESIDUAL FORCE            -4.241E-14   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         2.883E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        2.988E-07   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   5.838E-05   TIME AVG. PR. FLUX    1.423E-05
 LARGEST RESIDUAL PR. FLUX         -1.484E-12   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      2.404E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     1.069E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED
          TIME INCREMENT MAY NOW INCREASE TO    8.650E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.767E-03,  FRACTION OF STEP COMPLETED  1.720E-02
 STEP TIME COMPLETED       1.720E-02,  TOTAL TIME COMPLETED        1.720E-02


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.650E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.418E-05   TIME AVG. FORCE       5.927E-06
 LARGEST RESIDUAL FORCE            -2.147E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         4.325E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        6.722E-07   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   8.775E-05   TIME AVG. PR. FLUX    1.988E-05
 LARGEST RESIDUAL PR. FLUX         -7.512E-12   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.604E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     2.403E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED
          TIME INCREMENT MAY NOW INCREASE TO    1.000E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.650E-03,  FRACTION OF STEP COMPLETED  2.585E-02
 STEP TIME COMPLETED       2.585E-02,  TOTAL TIME COMPLETED        2.585E-02


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.353E-05   TIME AVG. FORCE       8.050E-06
 LARGEST RESIDUAL FORCE            -4.805E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.005E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   1.217E-04   TIME AVG. PR. FLUX    2.716E-05
 LARGEST RESIDUAL PR. FLUX         -1.680E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.164E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.590E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  3.585E-02
 STEP TIME COMPLETED       3.585E-02,  TOTAL TIME COMPLETED        3.585E-02


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.288E-05   TIME AVG. FORCE       1.054E-05
 LARGEST RESIDUAL FORCE            -5.530E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   1.556E-04   TIME AVG. PR. FLUX    3.572E-05
 LARGEST RESIDUAL PR. FLUX         -1.933E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.162E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.846E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  4.585E-02
 STEP TIME COMPLETED       4.585E-02,  TOTAL TIME COMPLETED        4.585E-02


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.223E-05   TIME AVG. FORCE       1.332E-05
 LARGEST RESIDUAL FORCE            -5.533E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   1.896E-04   TIME AVG. PR. FLUX    4.534E-05
 LARGEST RESIDUAL PR. FLUX         -1.934E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.159E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.841E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  5.585E-02
 STEP TIME COMPLETED       5.585E-02,  TOTAL TIME COMPLETED        5.585E-02


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.158E-05   TIME AVG. FORCE       1.633E-05
 LARGEST RESIDUAL FORCE            -5.536E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   2.236E-04   TIME AVG. PR. FLUX    5.582E-05
 LARGEST RESIDUAL PR. FLUX         -1.934E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.157E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.837E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  6.585E-02
 STEP TIME COMPLETED       6.585E-02,  TOTAL TIME COMPLETED        6.585E-02


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.093E-05   TIME AVG. FORCE       1.955E-05
 LARGEST RESIDUAL FORCE            -5.540E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   2.575E-04   TIME AVG. PR. FLUX    6.702E-05
 LARGEST RESIDUAL PR. FLUX         -1.935E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.154E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.832E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  7.585E-02
 STEP TIME COMPLETED       7.585E-02,  TOTAL TIME COMPLETED        7.585E-02


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.029E-05   TIME AVG. FORCE       2.292E-05
 LARGEST RESIDUAL FORCE            -5.543E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   2.914E-04   TIME AVG. PR. FLUX    7.883E-05
 LARGEST RESIDUAL PR. FLUX         -1.935E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.152E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.828E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  8.585E-02
 STEP TIME COMPLETED       8.585E-02,  TOTAL TIME COMPLETED        8.585E-02


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.964E-05   TIME AVG. FORCE       2.643E-05
 LARGEST RESIDUAL FORCE            -5.546E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   3.254E-04   TIME AVG. PR. FLUX    9.116E-05
 LARGEST RESIDUAL PR. FLUX         -1.936E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.149E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.823E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  9.585E-02
 STEP TIME COMPLETED       9.585E-02,  TOTAL TIME COMPLETED        9.585E-02


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.899E-05   TIME AVG. FORCE       3.006E-05
 LARGEST RESIDUAL FORCE            -5.549E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   3.593E-04   TIME AVG. PR. FLUX    1.039E-04
 LARGEST RESIDUAL PR. FLUX         -1.937E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.146E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.818E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.106    
 STEP TIME COMPLETED       0.106    ,  TOTAL TIME COMPLETED        0.106    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.083E-04   TIME AVG. FORCE       3.379E-05
 LARGEST RESIDUAL FORCE            -5.552E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   3.933E-04   TIME AVG. PR. FLUX    1.171E-04
 LARGEST RESIDUAL PR. FLUX         -1.937E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.144E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.814E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.116    
 STEP TIME COMPLETED       0.116    ,  TOTAL TIME COMPLETED        0.116    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.177E-04   TIME AVG. FORCE       3.760E-05
 LARGEST RESIDUAL FORCE            -5.555E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   4.272E-04   TIME AVG. PR. FLUX    1.306E-04
 LARGEST RESIDUAL PR. FLUX         -1.938E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.141E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.809E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.126    
 STEP TIME COMPLETED       0.126    ,  TOTAL TIME COMPLETED        0.126    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.270E-04   TIME AVG. FORCE       4.149E-05
 LARGEST RESIDUAL FORCE            -5.558E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   4.612E-04   TIME AVG. PR. FLUX    1.443E-04
 LARGEST RESIDUAL PR. FLUX         -1.938E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.139E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.804E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.136    
 STEP TIME COMPLETED       0.136    ,  TOTAL TIME COMPLETED        0.136    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.364E-04   TIME AVG. FORCE       4.545E-05
 LARGEST RESIDUAL FORCE            -5.561E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   4.951E-04   TIME AVG. PR. FLUX    1.584E-04
 LARGEST RESIDUAL PR. FLUX         -1.939E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.136E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.799E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.146    
 STEP TIME COMPLETED       0.146    ,  TOTAL TIME COMPLETED        0.146    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.457E-04   TIME AVG. FORCE       4.946E-05
 LARGEST RESIDUAL FORCE            -5.564E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   5.291E-04   TIME AVG. PR. FLUX    1.726E-04
 LARGEST RESIDUAL PR. FLUX         -1.939E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.134E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.795E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.156    
 STEP TIME COMPLETED       0.156    ,  TOTAL TIME COMPLETED        0.156    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.551E-04   TIME AVG. FORCE       5.352E-05
 LARGEST RESIDUAL FORCE            -5.567E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE PR. FLUX                   5.630E-04   TIME AVG. PR. FLUX    1.871E-04
 LARGEST RESIDUAL PR. FLUX         -1.940E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.131E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.790E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.166    
 STEP TIME COMPLETED       0.166    ,  TOTAL TIME COMPLETED        0.166    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.644E-04   TIME AVG. FORCE       5.763E-05
 LARGEST RESIDUAL FORCE            -5.569E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   5.970E-04   TIME AVG. PR. FLUX    2.017E-04
 LARGEST RESIDUAL PR. FLUX         -1.940E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.129E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.785E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.176    
 STEP TIME COMPLETED       0.176    ,  TOTAL TIME COMPLETED        0.176    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.738E-04   TIME AVG. FORCE       6.178E-05
 LARGEST RESIDUAL FORCE            -5.572E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   6.309E-04   TIME AVG. PR. FLUX    2.165E-04
 LARGEST RESIDUAL PR. FLUX         -1.941E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.126E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.780E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.186    
 STEP TIME COMPLETED       0.186    ,  TOTAL TIME COMPLETED        0.186    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.831E-04   TIME AVG. FORCE       6.596E-05
 LARGEST RESIDUAL FORCE            -5.575E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   6.649E-04   TIME AVG. PR. FLUX    2.315E-04
 LARGEST RESIDUAL PR. FLUX         -1.941E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.124E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.775E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.196    
 STEP TIME COMPLETED       0.196    ,  TOTAL TIME COMPLETED        0.196    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.925E-04   TIME AVG. FORCE       7.018E-05
 LARGEST RESIDUAL FORCE            -5.577E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   6.988E-04   TIME AVG. PR. FLUX    2.465E-04
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.121E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.771E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.206    
 STEP TIME COMPLETED       0.206    ,  TOTAL TIME COMPLETED        0.206    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.018E-04   TIME AVG. FORCE       7.443E-05
 LARGEST RESIDUAL FORCE            -5.580E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   7.328E-04   TIME AVG. PR. FLUX    2.617E-04
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.119E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.766E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.216    
 STEP TIME COMPLETED       0.216    ,  TOTAL TIME COMPLETED        0.216    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.112E-04   TIME AVG. FORCE       7.870E-05
 LARGEST RESIDUAL FORCE            -5.582E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   7.667E-04   TIME AVG. PR. FLUX    2.770E-04
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.116E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.761E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.226    
 STEP TIME COMPLETED       0.226    ,  TOTAL TIME COMPLETED        0.226    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.205E-04   TIME AVG. FORCE       8.300E-05
 LARGEST RESIDUAL FORCE            -5.584E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   8.007E-04   TIME AVG. PR. FLUX    2.924E-04
 LARGEST RESIDUAL PR. FLUX         -1.943E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.114E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.756E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.236    
 STEP TIME COMPLETED       0.236    ,  TOTAL TIME COMPLETED        0.236    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.299E-04   TIME AVG. FORCE       8.732E-05
 LARGEST RESIDUAL FORCE            -5.587E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   8.347E-04   TIME AVG. PR. FLUX    3.079E-04
 LARGEST RESIDUAL PR. FLUX         -1.943E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.111E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.751E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.246    
 STEP TIME COMPLETED       0.246    ,  TOTAL TIME COMPLETED        0.246    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.392E-04   TIME AVG. FORCE       9.166E-05
 LARGEST RESIDUAL FORCE            -5.589E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   8.686E-04   TIME AVG. PR. FLUX    3.235E-04
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.109E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.746E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.256    
 STEP TIME COMPLETED       0.256    ,  TOTAL TIME COMPLETED        0.256    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.486E-04   TIME AVG. FORCE       9.602E-05
 LARGEST RESIDUAL FORCE            -5.591E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   9.025E-04   TIME AVG. PR. FLUX    3.392E-04
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.106E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.741E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.266    
 STEP TIME COMPLETED       0.266    ,  TOTAL TIME COMPLETED        0.266    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.580E-04   TIME AVG. FORCE       1.004E-04
 LARGEST RESIDUAL FORCE            -5.593E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   9.365E-04   TIME AVG. PR. FLUX    3.549E-04
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.104E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.736E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.276    
 STEP TIME COMPLETED       0.276    ,  TOTAL TIME COMPLETED        0.276    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.673E-04   TIME AVG. FORCE       1.048E-04
 LARGEST RESIDUAL FORCE            -5.595E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   9.704E-04   TIME AVG. PR. FLUX    3.707E-04
 LARGEST RESIDUAL PR. FLUX         -1.945E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.101E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.731E-08   AT NODE        152
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.286    
 STEP TIME COMPLETED       0.286    ,  TOTAL TIME COMPLETED        0.286    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.766E-04   TIME AVG. FORCE       1.092E-04
 LARGEST RESIDUAL FORCE            -5.597E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.004E-03   TIME AVG. PR. FLUX    3.865E-04
 LARGEST RESIDUAL PR. FLUX         -1.945E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.099E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.726E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.296    
 STEP TIME COMPLETED       0.296    ,  TOTAL TIME COMPLETED        0.296    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.860E-04   TIME AVG. FORCE       1.136E-04
 LARGEST RESIDUAL FORCE            -5.599E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.038E-03   TIME AVG. PR. FLUX    4.024E-04
 LARGEST RESIDUAL PR. FLUX         -1.945E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.096E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.732E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.306    
 STEP TIME COMPLETED       0.306    ,  TOTAL TIME COMPLETED        0.306    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.954E-04   TIME AVG. FORCE       1.180E-04
 LARGEST RESIDUAL FORCE            -5.598E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.072E-03   TIME AVG. PR. FLUX    4.184E-04
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.094E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.737E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.316    
 STEP TIME COMPLETED       0.316    ,  TOTAL TIME COMPLETED        0.316    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.047E-04   TIME AVG. FORCE       1.225E-04
 LARGEST RESIDUAL FORCE            -5.606E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.106E-03   TIME AVG. PR. FLUX    4.344E-04
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.091E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.742E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.326    
 STEP TIME COMPLETED       0.326    ,  TOTAL TIME COMPLETED        0.326    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.140E-04   TIME AVG. FORCE       1.269E-04
 LARGEST RESIDUAL FORCE            -5.603E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.140E-03   TIME AVG. PR. FLUX    4.504E-04
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.089E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.747E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.336    
 STEP TIME COMPLETED       0.336    ,  TOTAL TIME COMPLETED        0.336    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.234E-04   TIME AVG. FORCE       1.314E-04
 LARGEST RESIDUAL FORCE            -5.614E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.174E-03   TIME AVG. PR. FLUX    4.665E-04
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.086E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.752E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.346    
 STEP TIME COMPLETED       0.346    ,  TOTAL TIME COMPLETED        0.346    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.327E-04   TIME AVG. FORCE       1.359E-04
 LARGEST RESIDUAL FORCE            -5.619E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.208E-03   TIME AVG. PR. FLUX    4.826E-04
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.084E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.757E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.356    
 STEP TIME COMPLETED       0.356    ,  TOTAL TIME COMPLETED        0.356    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.421E-04   TIME AVG. FORCE       1.404E-04
 LARGEST RESIDUAL FORCE            -5.620E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.242E-03   TIME AVG. PR. FLUX    4.988E-04
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.082E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.762E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.366    
 STEP TIME COMPLETED       0.366    ,  TOTAL TIME COMPLETED        0.366    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.515E-04   TIME AVG. FORCE       1.449E-04
 LARGEST RESIDUAL FORCE            -5.604E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.276E-03   TIME AVG. PR. FLUX    5.150E-04
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.079E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.767E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.376    
 STEP TIME COMPLETED       0.376    ,  TOTAL TIME COMPLETED        0.376    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.608E-04   TIME AVG. FORCE       1.494E-04
 LARGEST RESIDUAL FORCE            -5.594E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.310E-03   TIME AVG. PR. FLUX    5.312E-04
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.077E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.772E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.386    
 STEP TIME COMPLETED       0.386    ,  TOTAL TIME COMPLETED        0.386    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.702E-04   TIME AVG. FORCE       1.539E-04
 LARGEST RESIDUAL FORCE            -5.630E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.344E-03   TIME AVG. PR. FLUX    5.474E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.074E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.776E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.396    
 STEP TIME COMPLETED       0.396    ,  TOTAL TIME COMPLETED        0.396    


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.795E-04   TIME AVG. FORCE       1.584E-04
 LARGEST RESIDUAL FORCE            -5.632E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.378E-03   TIME AVG. PR. FLUX    5.637E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.072E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.781E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.406    
 STEP TIME COMPLETED       0.406    ,  TOTAL TIME COMPLETED        0.406    


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.889E-04   TIME AVG. FORCE       1.629E-04
 LARGEST RESIDUAL FORCE            -5.610E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.412E-03   TIME AVG. PR. FLUX    5.800E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.070E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.786E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.416    
 STEP TIME COMPLETED       0.416    ,  TOTAL TIME COMPLETED        0.416    


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.982E-04   TIME AVG. FORCE       1.674E-04
 LARGEST RESIDUAL FORCE            -5.619E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.080E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.446E-03   TIME AVG. PR. FLUX    5.964E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.067E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.791E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.426    
 STEP TIME COMPLETED       0.426    ,  TOTAL TIME COMPLETED        0.426    


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.076E-04   TIME AVG. FORCE       1.720E-04
 LARGEST RESIDUAL FORCE            -5.624E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.480E-03   TIME AVG. PR. FLUX    6.127E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.065E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.795E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.436    
 STEP TIME COMPLETED       0.436    ,  TOTAL TIME COMPLETED        0.436    


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.169E-04   TIME AVG. FORCE       1.765E-04
 LARGEST RESIDUAL FORCE            -5.622E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.514E-03   TIME AVG. PR. FLUX    6.291E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.062E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.800E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.446    
 STEP TIME COMPLETED       0.446    ,  TOTAL TIME COMPLETED        0.446    


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.263E-04   TIME AVG. FORCE       1.810E-04
 LARGEST RESIDUAL FORCE            -5.619E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.548E-03   TIME AVG. PR. FLUX    6.455E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.060E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.805E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.456    
 STEP TIME COMPLETED       0.456    ,  TOTAL TIME COMPLETED        0.456    


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.357E-04   TIME AVG. FORCE       1.856E-04
 LARGEST RESIDUAL FORCE            -5.623E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.582E-03   TIME AVG. PR. FLUX    6.619E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.058E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.809E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.466    
 STEP TIME COMPLETED       0.466    ,  TOTAL TIME COMPLETED        0.466    


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.450E-04   TIME AVG. FORCE       1.901E-04
 LARGEST RESIDUAL FORCE            -5.621E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.616E-03   TIME AVG. PR. FLUX    6.784E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.055E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.814E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.476    
 STEP TIME COMPLETED       0.476    ,  TOTAL TIME COMPLETED        0.476    


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.543E-04   TIME AVG. FORCE       1.947E-04
 LARGEST RESIDUAL FORCE            -5.625E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.650E-03   TIME AVG. PR. FLUX    6.948E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.053E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.819E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.486    
 STEP TIME COMPLETED       0.486    ,  TOTAL TIME COMPLETED        0.486    


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.637E-04   TIME AVG. FORCE       1.992E-04
 LARGEST RESIDUAL FORCE            -5.620E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.684E-03   TIME AVG. PR. FLUX    7.113E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.051E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.823E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.496    
 STEP TIME COMPLETED       0.496    ,  TOTAL TIME COMPLETED        0.496    


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.730E-04   TIME AVG. FORCE       2.038E-04
 LARGEST RESIDUAL FORCE            -5.627E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.717E-03   TIME AVG. PR. FLUX    7.278E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.048E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.828E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.506    
 STEP TIME COMPLETED       0.506    ,  TOTAL TIME COMPLETED        0.506    


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.824E-04   TIME AVG. FORCE       2.084E-04
 LARGEST RESIDUAL FORCE            -5.629E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.751E-03   TIME AVG. PR. FLUX    7.443E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.046E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.832E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.516    
 STEP TIME COMPLETED       0.516    ,  TOTAL TIME COMPLETED        0.516    


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.918E-04   TIME AVG. FORCE       2.129E-04
 LARGEST RESIDUAL FORCE            -5.622E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.785E-03   TIME AVG. PR. FLUX    7.608E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.044E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.836E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.526    
 STEP TIME COMPLETED       0.526    ,  TOTAL TIME COMPLETED        0.526    


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.012E-04   TIME AVG. FORCE       2.175E-04
 LARGEST RESIDUAL FORCE            -5.631E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.819E-03   TIME AVG. PR. FLUX    7.774E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.041E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.841E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.536    
 STEP TIME COMPLETED       0.536    ,  TOTAL TIME COMPLETED        0.536    


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.105E-04   TIME AVG. FORCE       2.221E-04
 LARGEST RESIDUAL FORCE            -5.623E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.853E-03   TIME AVG. PR. FLUX    7.939E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.039E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.845E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.546    
 STEP TIME COMPLETED       0.546    ,  TOTAL TIME COMPLETED        0.546    


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.199E-04   TIME AVG. FORCE       2.267E-04
 LARGEST RESIDUAL FORCE            -5.627E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.887E-03   TIME AVG. PR. FLUX    8.105E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.037E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.849E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.556    
 STEP TIME COMPLETED       0.556    ,  TOTAL TIME COMPLETED        0.556    


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.292E-04   TIME AVG. FORCE       2.313E-04
 LARGEST RESIDUAL FORCE            -5.626E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.921E-03   TIME AVG. PR. FLUX    8.271E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.034E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.854E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.566    
 STEP TIME COMPLETED       0.566    ,  TOTAL TIME COMPLETED        0.566    


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.386E-04   TIME AVG. FORCE       2.359E-04
 LARGEST RESIDUAL FORCE            -5.631E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.955E-03   TIME AVG. PR. FLUX    8.437E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.032E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.858E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.576    
 STEP TIME COMPLETED       0.576    ,  TOTAL TIME COMPLETED        0.576    


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.479E-04   TIME AVG. FORCE       2.404E-04
 LARGEST RESIDUAL FORCE            -5.630E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   1.989E-03   TIME AVG. PR. FLUX    8.603E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.030E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.862E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.586    
 STEP TIME COMPLETED       0.586    ,  TOTAL TIME COMPLETED        0.586    


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.573E-04   TIME AVG. FORCE       2.450E-04
 LARGEST RESIDUAL FORCE            -5.626E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.023E-03   TIME AVG. PR. FLUX    8.769E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.027E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.866E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.596    
 STEP TIME COMPLETED       0.596    ,  TOTAL TIME COMPLETED        0.596    


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.666E-04   TIME AVG. FORCE       2.496E-04
 LARGEST RESIDUAL FORCE            -5.628E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.057E-03   TIME AVG. PR. FLUX    8.935E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.025E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.870E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.606    
 STEP TIME COMPLETED       0.606    ,  TOTAL TIME COMPLETED        0.606    


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.760E-04   TIME AVG. FORCE       2.542E-04
 LARGEST RESIDUAL FORCE            -5.627E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.091E-03   TIME AVG. PR. FLUX    9.101E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.023E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.874E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.616    
 STEP TIME COMPLETED       0.616    ,  TOTAL TIME COMPLETED        0.616    


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.853E-04   TIME AVG. FORCE       2.588E-04
 LARGEST RESIDUAL FORCE            -5.631E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.079E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.125E-03   TIME AVG. PR. FLUX    9.268E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.020E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.879E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.626    
 STEP TIME COMPLETED       0.626    ,  TOTAL TIME COMPLETED        0.626    


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.947E-04   TIME AVG. FORCE       2.634E-04
 LARGEST RESIDUAL FORCE            -5.629E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.159E-03   TIME AVG. PR. FLUX    9.434E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.018E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.883E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.636    
 STEP TIME COMPLETED       0.636    ,  TOTAL TIME COMPLETED        0.636    


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.041E-04   TIME AVG. FORCE       2.680E-04
 LARGEST RESIDUAL FORCE            -5.627E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.193E-03   TIME AVG. PR. FLUX    9.601E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.016E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.887E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.646    
 STEP TIME COMPLETED       0.646    ,  TOTAL TIME COMPLETED        0.646    


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.135E-04   TIME AVG. FORCE       2.726E-04
 LARGEST RESIDUAL FORCE            -5.627E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.227E-03   TIME AVG. PR. FLUX    9.768E-04
 LARGEST RESIDUAL PR. FLUX         -1.949E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.014E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.891E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.656    
 STEP TIME COMPLETED       0.656    ,  TOTAL TIME COMPLETED        0.656    


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.228E-04   TIME AVG. FORCE       2.772E-04
 LARGEST RESIDUAL FORCE            -5.628E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.261E-03   TIME AVG. PR. FLUX    9.934E-04
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.011E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.894E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.666    
 STEP TIME COMPLETED       0.666    ,  TOTAL TIME COMPLETED        0.666    


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.322E-04   TIME AVG. FORCE       2.819E-04
 LARGEST RESIDUAL FORCE            -5.626E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.295E-03   TIME AVG. PR. FLUX    1.010E-03
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.009E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.898E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.676    
 STEP TIME COMPLETED       0.676    ,  TOTAL TIME COMPLETED        0.676    


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.416E-04   TIME AVG. FORCE       2.865E-04
 LARGEST RESIDUAL FORCE            -5.625E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.329E-03   TIME AVG. PR. FLUX    1.027E-03
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.007E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.902E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.686    
 STEP TIME COMPLETED       0.686    ,  TOTAL TIME COMPLETED        0.686    


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.509E-04   TIME AVG. FORCE       2.911E-04
 LARGEST RESIDUAL FORCE            -5.625E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.363E-03   TIME AVG. PR. FLUX    1.044E-03
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.005E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.906E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.696    
 STEP TIME COMPLETED       0.696    ,  TOTAL TIME COMPLETED        0.696    


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.603E-04   TIME AVG. FORCE       2.957E-04
 LARGEST RESIDUAL FORCE            -5.623E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.397E-03   TIME AVG. PR. FLUX    1.060E-03
 LARGEST RESIDUAL PR. FLUX         -1.948E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.002E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.910E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.706    
 STEP TIME COMPLETED       0.706    ,  TOTAL TIME COMPLETED        0.706    


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.697E-04   TIME AVG. FORCE       3.003E-04
 LARGEST RESIDUAL FORCE            -5.626E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.431E-03   TIME AVG. PR. FLUX    1.077E-03
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      4.000E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.914E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.716    
 STEP TIME COMPLETED       0.716    ,  TOTAL TIME COMPLETED        0.716    


  INCREMENT    83 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.791E-04   TIME AVG. FORCE       3.049E-04
 LARGEST RESIDUAL FORCE            -5.624E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.465E-03   TIME AVG. PR. FLUX    1.094E-03
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.998E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.917E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.726    
 STEP TIME COMPLETED       0.726    ,  TOTAL TIME COMPLETED        0.726    


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.884E-04   TIME AVG. FORCE       3.095E-04
 LARGEST RESIDUAL FORCE            -5.607E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.078E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.498E-03   TIME AVG. PR. FLUX    1.110E-03
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.996E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.921E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.736    
 STEP TIME COMPLETED       0.736    ,  TOTAL TIME COMPLETED        0.736    


  INCREMENT    85 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.977E-04   TIME AVG. FORCE       3.142E-04
 LARGEST RESIDUAL FORCE            -5.620E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.532E-03   TIME AVG. PR. FLUX    1.127E-03
 LARGEST RESIDUAL PR. FLUX         -1.947E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.994E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.925E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.746    
 STEP TIME COMPLETED       0.746    ,  TOTAL TIME COMPLETED        0.746    


  INCREMENT    86 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.071E-04   TIME AVG. FORCE       3.188E-04
 LARGEST RESIDUAL FORCE            -5.624E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.566E-03   TIME AVG. PR. FLUX    1.144E-03
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.991E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.928E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.756    
 STEP TIME COMPLETED       0.756    ,  TOTAL TIME COMPLETED        0.756    


  INCREMENT    87 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.164E-04   TIME AVG. FORCE       3.234E-04
 LARGEST RESIDUAL FORCE            -5.618E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.600E-03   TIME AVG. PR. FLUX    1.161E-03
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.989E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.932E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.766    
 STEP TIME COMPLETED       0.766    ,  TOTAL TIME COMPLETED        0.766    


  INCREMENT    88 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.259E-04   TIME AVG. FORCE       3.280E-04
 LARGEST RESIDUAL FORCE            -5.617E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.634E-03   TIME AVG. PR. FLUX    1.177E-03
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.987E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.935E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.776    
 STEP TIME COMPLETED       0.776    ,  TOTAL TIME COMPLETED        0.776    


  INCREMENT    89 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.352E-04   TIME AVG. FORCE       3.327E-04
 LARGEST RESIDUAL FORCE            -5.623E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.668E-03   TIME AVG. PR. FLUX    1.194E-03
 LARGEST RESIDUAL PR. FLUX         -1.946E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.985E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.939E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.786    
 STEP TIME COMPLETED       0.786    ,  TOTAL TIME COMPLETED        0.786    


  INCREMENT    90 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.446E-04   TIME AVG. FORCE       3.373E-04
 LARGEST RESIDUAL FORCE            -5.614E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.702E-03   TIME AVG. PR. FLUX    1.211E-03
 LARGEST RESIDUAL PR. FLUX         -1.945E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.983E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.942E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.796    
 STEP TIME COMPLETED       0.796    ,  TOTAL TIME COMPLETED        0.796    


  INCREMENT    91 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.540E-04   TIME AVG. FORCE       3.419E-04
 LARGEST RESIDUAL FORCE            -5.616E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.736E-03   TIME AVG. PR. FLUX    1.228E-03
 LARGEST RESIDUAL PR. FLUX         -1.945E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.981E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.946E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.806    
 STEP TIME COMPLETED       0.806    ,  TOTAL TIME COMPLETED        0.806    


  INCREMENT    92 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.633E-04   TIME AVG. FORCE       3.466E-04
 LARGEST RESIDUAL FORCE            -5.615E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.077E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.770E-03   TIME AVG. PR. FLUX    1.244E-03
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.978E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.949E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.816    
 STEP TIME COMPLETED       0.816    ,  TOTAL TIME COMPLETED        0.816    


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.727E-04   TIME AVG. FORCE       3.512E-04
 LARGEST RESIDUAL FORCE            -5.613E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.804E-03   TIME AVG. PR. FLUX    1.261E-03
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.976E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.952E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.826    
 STEP TIME COMPLETED       0.826    ,  TOTAL TIME COMPLETED        0.826    


  INCREMENT    94 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.821E-04   TIME AVG. FORCE       3.558E-04
 LARGEST RESIDUAL FORCE            -5.614E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.838E-03   TIME AVG. PR. FLUX    1.278E-03
 LARGEST RESIDUAL PR. FLUX         -1.944E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.974E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.956E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.836    
 STEP TIME COMPLETED       0.836    ,  TOTAL TIME COMPLETED        0.836    


  INCREMENT    95 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.915E-04   TIME AVG. FORCE       3.605E-04
 LARGEST RESIDUAL FORCE            -5.607E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.872E-03   TIME AVG. PR. FLUX    1.295E-03
 LARGEST RESIDUAL PR. FLUX         -1.943E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.972E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.959E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.846    
 STEP TIME COMPLETED       0.846    ,  TOTAL TIME COMPLETED        0.846    


  INCREMENT    96 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.008E-04   TIME AVG. FORCE       3.651E-04
 LARGEST RESIDUAL FORCE            -5.611E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.906E-03   TIME AVG. PR. FLUX    1.311E-03
 LARGEST RESIDUAL PR. FLUX         -1.943E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.970E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.962E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.856    
 STEP TIME COMPLETED       0.856    ,  TOTAL TIME COMPLETED        0.856    


  INCREMENT    97 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.102E-04   TIME AVG. FORCE       3.697E-04
 LARGEST RESIDUAL FORCE            -5.603E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.940E-03   TIME AVG. PR. FLUX    1.328E-03
 LARGEST RESIDUAL PR. FLUX         -1.943E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.968E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.965E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.866    
 STEP TIME COMPLETED       0.866    ,  TOTAL TIME COMPLETED        0.866    


  INCREMENT    98 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.196E-04   TIME AVG. FORCE       3.744E-04
 LARGEST RESIDUAL FORCE            -5.606E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   2.974E-03   TIME AVG. PR. FLUX    1.345E-03
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.966E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.968E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.876    
 STEP TIME COMPLETED       0.876    ,  TOTAL TIME COMPLETED        0.876    


  INCREMENT    99 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.290E-04   TIME AVG. FORCE       3.790E-04
 LARGEST RESIDUAL FORCE            -5.600E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.076E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.008E-03   TIME AVG. PR. FLUX    1.362E-03
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.964E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.972E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.886    
 STEP TIME COMPLETED       0.886    ,  TOTAL TIME COMPLETED        0.886    


  INCREMENT   100 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.384E-04   TIME AVG. FORCE       3.836E-04
 LARGEST RESIDUAL FORCE            -5.603E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.042E-03   TIME AVG. PR. FLUX    1.379E-03
 LARGEST RESIDUAL PR. FLUX         -1.942E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.961E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.975E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.896    
 STEP TIME COMPLETED       0.896    ,  TOTAL TIME COMPLETED        0.896    


  INCREMENT   101 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.477E-04   TIME AVG. FORCE       3.883E-04
 LARGEST RESIDUAL FORCE            -5.597E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.076E-03   TIME AVG. PR. FLUX    1.395E-03
 LARGEST RESIDUAL PR. FLUX         -1.941E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.959E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.978E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.906    
 STEP TIME COMPLETED       0.906    ,  TOTAL TIME COMPLETED        0.906    


  INCREMENT   102 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.571E-04   TIME AVG. FORCE       3.929E-04
 LARGEST RESIDUAL FORCE            -5.599E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.110E-03   TIME AVG. PR. FLUX    1.412E-03
 LARGEST RESIDUAL PR. FLUX         -1.940E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.957E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.981E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.916    
 STEP TIME COMPLETED       0.916    ,  TOTAL TIME COMPLETED        0.916    


  INCREMENT   103 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.665E-04   TIME AVG. FORCE       3.976E-04
 LARGEST RESIDUAL FORCE            -5.597E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.144E-03   TIME AVG. PR. FLUX    1.429E-03
 LARGEST RESIDUAL PR. FLUX         -1.940E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.955E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.984E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.926    
 STEP TIME COMPLETED       0.926    ,  TOTAL TIME COMPLETED        0.926    


  INCREMENT   104 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.758E-04   TIME AVG. FORCE       4.022E-04
 LARGEST RESIDUAL FORCE            -5.592E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.178E-03   TIME AVG. PR. FLUX    1.446E-03
 LARGEST RESIDUAL PR. FLUX         -1.940E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.953E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.987E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.936    
 STEP TIME COMPLETED       0.936    ,  TOTAL TIME COMPLETED        0.936    


  INCREMENT   105 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.852E-04   TIME AVG. FORCE       4.069E-04
 LARGEST RESIDUAL FORCE            -5.594E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.075E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.212E-03   TIME AVG. PR. FLUX    1.463E-03
 LARGEST RESIDUAL PR. FLUX         -1.939E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.951E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.990E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.946    
 STEP TIME COMPLETED       0.946    ,  TOTAL TIME COMPLETED        0.946    


  INCREMENT   106 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.946E-04   TIME AVG. FORCE       4.115E-04
 LARGEST RESIDUAL FORCE            -5.585E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.074E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.245E-03   TIME AVG. PR. FLUX    1.480E-03
 LARGEST RESIDUAL PR. FLUX         -1.939E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.949E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.992E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.956    
 STEP TIME COMPLETED       0.956    ,  TOTAL TIME COMPLETED        0.956    


  INCREMENT   107 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.040E-04   TIME AVG. FORCE       4.161E-04
 LARGEST RESIDUAL FORCE            -5.583E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.074E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.279E-03   TIME AVG. PR. FLUX    1.496E-03
 LARGEST RESIDUAL PR. FLUX         -1.938E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.947E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.995E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.966    
 STEP TIME COMPLETED       0.966    ,  TOTAL TIME COMPLETED        0.966    


  INCREMENT   108 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.134E-04   TIME AVG. FORCE       4.208E-04
 LARGEST RESIDUAL FORCE            -5.583E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.074E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.313E-03   TIME AVG. PR. FLUX    1.513E-03
 LARGEST RESIDUAL PR. FLUX         -1.938E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.945E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     3.998E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.976    
 STEP TIME COMPLETED       0.976    ,  TOTAL TIME COMPLETED        0.976    


  INCREMENT   109 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.228E-04   TIME AVG. FORCE       4.254E-04
 LARGEST RESIDUAL FORCE            -5.582E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.074E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.347E-03   TIME AVG. PR. FLUX    1.530E-03
 LARGEST RESIDUAL PR. FLUX         -1.937E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.943E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     4.001E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.986    
 STEP TIME COMPLETED       0.986    ,  TOTAL TIME COMPLETED        0.986    


  INCREMENT   110 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-02
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.321E-04   TIME AVG. FORCE       4.301E-04
 LARGEST RESIDUAL FORCE            -5.579E-13   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         5.000E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        1.074E-06   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.381E-03   TIME AVG. PR. FLUX    1.547E-03
 LARGEST RESIDUAL PR. FLUX         -1.936E-11   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      3.941E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     4.003E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM EQUATIONS HAVE CONVERGED

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-02,  FRACTION OF STEP COMPLETED  0.996    
 STEP TIME COMPLETED       0.996    ,  TOTAL TIME COMPLETED        0.996    


  INCREMENT   111 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.151E-03
 
	REGULARIZED SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     5820
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         5.704e+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.360E-04   TIME AVG. FORCE       4.347E-04
 LARGEST RESIDUAL FORCE            -4.827E-14   AT NODE        203   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF DISP.         2.075E-03   AT NODE          5   DOF  1
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO DISP.        3.153E-07   AT NODE        220   DOF  1
   INSTANCE: CUBE-1                                                                          
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE PR. FLUX                   3.395E-03   TIME AVG. PR. FLUX    1.563E-03
 LARGEST RESIDUAL PR. FLUX         -1.669E-12   AT NODE       1332
   INSTANCE: CUBE-1                                                                          
 LARGEST INCREMENT OF VOL. LM.      1.635E-05   AT NODE         24
   INSTANCE: CUBE-1                                                                          
 LARGEST CORRECTION TO VOL. LM.     1.176E-08   AT NODE        153
   INSTANCE: CUBE-1                                                                          
          THE PR. FLUX  EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.151E-03,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        1.00    ,  TOTAL TIME COMPLETED         1.00    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        111  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     112  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     112  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     112  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                       2  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   39.000    
       SYSTEM TIME (SEC)    =   2.3000    
       TOTAL CPU TIME (SEC) =   41.300    
       WALLCLOCK TIME (SEC) =         42
