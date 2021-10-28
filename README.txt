This code provides the tools to reproduce the results in the paper "When shared concept cells support associations: theory of overlapping memory engrams",
by Chiara Gastaldi, Tilo Schwalger, Emanuela De Falco, Rodrigo Quian Quiroga, and Wufram Gerstner

The code is written by Chiara Gastaldi.

The AttractorNetwork folder contains code to solve mean the field equations of an attractor neural network, 
in which correlated patterns are stored.
The code is written in Julialang. In the sub-folder "Plots", you can find some python files to plot the result of the main code. 
In order to reproduce the results of the paper and plot them follow the instruction below.

----------------------------------------------------------------------------------
PART 1 - INITIALISATION:
----------------------------------------------------------------------------------
The code is written in julialang (tested with version Julia-1.4) and Python (tested with version 3.7.0)

Download the AttractorNetwork folder from :  .... !!!!!!!!!!!
In the folder you will find 
    - the "Main.jl" file, 
    - the "Plots" folder which contains the plot scripts in Python  
    - the "src" folder which contains the "AttractorNetwork.jl" file.
    - the "Data_analysis" folder
    - the "Files" folder

The following empty folders should also be present in the AttractorNetwork folder.
a. Output folders:
    - Output_1pop
    - Output_2pop
    - Output_3pop
    - Output_4pop
    - Output_2pop_adapt
    - Output_4pop_adapt
    - Output_full_sim
    - Output_full_sim_adapt
    - Output_full_sim_distr_gains
    - Output_full_sim_adapt_distr_gains

b. Figures folders:
    - Figures_1pop
    - Figures_2pop
    - Figures_3pop
    - Figures_4pop
    - Figures_2pop_adapt
    - Figures_4pop_adapt
    - Figures_full_sim
    - Figures_full_sim_adapt
    - Figures_full_sim_distr_gains
    - Figures_full_sim_adapt_distr_gains
    - Figures_exp

IN SHORT: 
The output folders will host the .dat files that are outputted by running the Main.jl script. 
The same data files are read by the plot scripts contained in the "Plots" folder.
Running the plots scripts creates images files in the corresponding Figures folders.
Details are given below.

----------------------------------------------------------------------------------
PART 2 - USING THE CODE:
----------------------------------------------------------------------------------

Open the file "Main.jl" and uncomment the function you want to run.
Open Julia in your terminal and run the "Main.jl" file doing: import("Main.jl")
The following operations are executed: 
- activation of the package AttractorNetwork, 
- importation of the module AttractorNetwork.jl contained in the "src" sub-folder
- definition of the used model: 
    you can choose between TwoPopulations.MeanField() and TwoPopulations.MFnoSelfInteraction(), this will be relevant only if you run a function from the module TwoPopolations.

    The model TwoPopulations.MeanField() used a standard mean field for two correlated memory patters, while the model TwoPopulations.MFnoSelfInteraction() uses the mean field theory in which self interaction is excluded, this requires the an extra correction as described in the Methods session.

    WARNING: the model TwoPopulations.MFnoSelfInteraction() requires to solve the mean-field equation recursively. This means that solving the system with this model requires a long computational time and it is recommended to run the computation in parallel. For example, if you want to use 4 cores on your machine, you can run the command: import("Main.jl") -p 4

- running the uncommented function.


-- LIST OF THE 21 FUNCTIONS IN THE Main.jl SCRIPT AND WHAT THEY DO --
In the following we use a # character at the place of the parameters' values.

####################################################################
################# MEAN FIELD #######################################
####################################################################

-- LIST OF PARAMETERS -- 
    gamma = sparseness of the memory representation
    resolution = resolution of the grid search in Eq. 108, 110 of the paper
    factor = resolution factor named correction-constant in Eq. 107, 109
    size = how many values of C_hat (or alpha) do you want to consider in building the bifurcation plot
    rm = maximal firing rate, r_{max} in the paper
    A = connection strength in the definition of the weight matrix
    h0 = firing rate threshold 
    b = steepness of the gain function, see Eq. 9
    load = network load = alpha  = P/N
    I1 = intensity of the external current given to population 1
    I2 = intensity of the external current given to population 2
    bound_low = lower bound of the grid search, compare to Eq. 110
    bound_up = upper bound of the grid search, compare to Eq. 110
    bound_r = upper bound of the R variable, max(R) in Eq. 108
    C_hat = correlation between patterns, corresponding to the parameter C defined in Eq. 41. 
        NOTE: C_hat is the correlation coefficient C, not the fraction of shared neurons c. The relation between the two quantities is c = C(1 − γ) + γ
    mod = you can choose one of the following model, which correspond to mean field with or without self interaction (see section "Excluding self-interaction" in the paper): 
        TwoPopulations.MeanField() or TwoPopulations.MFnoSelfInteraction() 
    corr_noise = boolean variable, choose true if you want to have correlated background patterns of false if you want background patterns to be independent. Refer to section "Overlapping background patterns" in the paper.
    time_step = time step in the Euler integration of a dynamical equation

------------------ ONE POPULATION ------------------------------
The following scripts are need to solve the mean field theory for one homogeneous neural population only. 
No figure in the paper has been generated with this functions, but we used them to run sanity checks in the simple situation in which only one memory assembly has been stored in the network.

1. OnePopulation.PP_m1R(): it produces phase plane of the m1-nullcline and the manifold of solutions of R and the fixed point solutions at their intersections.
It outputs in the Output_1pop folder the following files 
    - the m^1-nullcline: nullclineM_1pop_noise_gamma#_resolution#_factor#_rm#_A#_h0#_load#.dat
    - the R-nullcline: nullclineR_pop_noise_gamma#_resolution#_factor#_rm#_A#_h0#_load#.dat
    - the posistion of the fixed points: fp3D_1pop_noise_gamma#_resolution#_factor#_rm#_A#_h0#_load#.dat
    - their stability: Stab_1pop_noise_gamma#_resolution#_factor#_rm#_A#_h0#_load#.dat
DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, load = 10, resolution, resolution_factor = 1, c_hat = 0
Parameters can be changed manually by specifying them into the brackets.  

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "OnePopulation_PP_m1R.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "OnePopulation_PP_m1R.py" (please uncomment the legend command if you want it). 
- the file "PP_m1R_b#_load#_gamma#_rm1_h0#_resolution#.png" will appear in the folder "Figures_1pop".

The figure shows the m1 nullcline in blue and the manifold of R solutions in orange. 
At their intersections, the fixed points appear color-coded: blue = stable, green = saddle, red = unstable.
Be carful that for b >= 1000, the unstable fixed point will not appear in the plot due to finite nature of the grid search.

2. OnePopulation.compute_critical_capacity(): it produces a bifurcation diagram in which we see the m^1 coordinate of the fixed points as a function of the network load alpha
It outputs in the Output_1pop folder the following files 
    - the position of the m1 component of the fixed points vs the load: bifurcation_1pop_fixed_points_vs_load_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_chat#.dat
DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, max_load = 10, resolution, resolution_factor = 1, size, c_hat = 0
Parameters can be changed manually by specifying them into the brackets. 

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "OnePopulation_compute_critical_capacity.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "OnePopulation_compute_critical_capacity.py" (please uncomment the legend command if you want it). 
- the file "bifurcation_1pop_vs_load_size#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_chat#.png" will appear in the folder "Figures_1pop".

The figure shows bifurcation plot vs the load alpha. 
The lines generated by the m1 projection of the fixed points appear color-coded: blue = stable, green = saddle, red = unstable.
Be carful that for b >= 1000, the unstable fixed point will not appear in the plot due to finite nature of the grid search.

-------------------------- TWO POPULATIONS ---------------------------------------

3. TwoPopulations.generate_bifurcation_diagram(): it produces a bifurcation diagram in which we see the m^1 coordinate of the fixed points as a function of the correlation C_hat between the first two patterns as in Fig. 9B-C.
It outputs in the Output_2pop folder the following files 
    - the position of the m1 component of the fixed points vs the correlation: Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A1_h0#_b#_load#_model#_corr_noise#.dat
    - a bunch of files for generating the quiver plots: Quiver_PP_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#
    DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, I1 = 0., I2 = 0., bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution_factor = 1,  corr_noise = false
    The values of the following parameters don't have a default value assigned and they must be specified: resolution, size, load, model.
    All parameters can be changed manually by specifying them into the brackets. 

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "TwoPopulation_generate_bifurcation_diagram.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "TwoPopulation_generate_bifurcation_diagram.py". Note that this scripts converts the values of correlation C into fraction of shared neruons c.
- the file "bifurcation_2pop_vs_load_size#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_chat#.pdf" will appear in the folder "Figures_2pop".

The figure shows bifurcation plot vs the fraction of shared neurons c. 
The lines generated by the m1 projection of the fixed points appear color-coded: blue = stable, green = saddle, red = unstable.
Be carful that for b >= 1000, the unstable fixed point will not appear in the plot due to finite nature of the grid search.

4. TwoPopulations.generate_critical_corr_vs_b_h0(): it produces Fig 2.A
It outputs in the Output_2pop folder the following files 
    - a bunch of bifurcation plots, one for each pixel of the graph: Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_model#_corr_noise#.dat
    DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, I1 = 0., I2 = 0., bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution_factor = 1, corr_noise = false
    The values of the following parameters don't have a default value assigned and they must be specified: resolution, size, load, model.
    All parameters can be changed manually by specifying them into the brackets. 
    
To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "PLOT_C_hat_crit_vs_h0_b.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "PLOT_C_hat_crit_vs_h0_b.py". Note that this scripts converts the values of correlation C into fraction of shared neruons c.
- the files "Crit_C_hat_vs_b_h0_gamma#_A#_rm#.png", "Crit_C_hat_vs_h0_gamma#_A#_rm#_b#.pdf" will appear in the folder "Figures_2pop".

The first figure reproduces Fig. 2A in the paper, while the second one shows the profile line for b = 100. 
The lines generated by the m1 projection of the fixed points appear color-coded: blue = stable, green = saddle, red = unstable.
Be carful that for very big values of b, the unstable fixed point will not appear in the plot due to finite nature of the grid search.

5. TwoPopulations.run_dynamics(): it is necessary to produce the mean field dynamics as in Fig. S1. 
The script outputs in the Output_2pop folder the following files
    - 2 files of the type fp_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - 2 files of the type nullcline1_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - 2 files of the type nullcline2_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - 2 files of the type Quiver_PP_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    one of the two files is for I1 = 0 and one for I1 = 1.
    - 1 file of the type MF_vs_time_gamma#_time_step#_resolution#_factor#_rm#_A#_h0#_b#_load#_Chat#.dat
    DEFAULT PARAMETERS:rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, I2_val = 0.1, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 15, t_onset = 0.5, t_offset = 5.5, h = 0.01, tau_m = 1, corr_noise = false
    The values of the following parameters don't have a default value assigned and they must be specified: model.
    All parameters can be changed manually by specifying them into the brackets. 

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Phase_plane_with_trajectory.py" and make sure that the parameters at the beginning of the file are specified correctly. 
- run the plotting script "Phase_plane_with_trajectory.py"
- the files "PP_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_Chat#.pdf" and "MF_vs_time_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_Chat#.pdf" will appear in the folder "Figures_2pop".   

6. TwoPopulations.generate_I_C_curve(): it is necessary to produce Fig. 1E
It outputs in the Output_2pop folder the following files
    - a bunch of files of the type fp_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - a file of the type generate_I_C_curve_gamma#_rm#_A#_h0#_b#_load#.dat
    - a bunch of files of the type MF_vs_time_gamma#_time_step#_resolution#_factor#_rm#_A#_h0#_b#_load#_Chat#.dat
    - a bunch of files of the type nullcline1_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - a bunch of files of the type nullcline2_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    - a bunch of files of the type Quiver_PP_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#_I1#.dat
    DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, tau_m = 1, h = 0.1,  bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1,  model = "MeanField", corr_noise = false
    The values of the following parameters don't have a default value assigned and they must be specified: size_I, size_C, load
    All parameters can be changed manually by specifying them into the brackets. 


To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "C_I_curve.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "C_I_curve.py"
- the file "C_I_curve.pdf" will appear in the folder "Figures_2pop".

-------------------------- THREE POPULATIONS ---------------------------------------

7. ThreePopulations.generate_bifurcation_diagram(): it is needed to generate one column in Fig. 2B.
It outputs in the "Output_3pop" folder the following files
    - the file Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#.dat
    DEFAULT PARAMETERS: rm = 50, A = 1, h0 = 10, γ = 0.001 , b = 0.8, bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution_factor = 1
    The values of the following parameters don't have a default value assigned and they must be specified: size, resolution, load
    All parameters can be changed manually by specifying them into the brackets. 


To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "ThreePopulation_generate_bifurcation_diagram.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "ThreePopulation_generate_bifurcation_diagram.py"
- the file "Bifurcation_vs_c_size#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_load#_model#.pdf" will appear in the folder "Figures_3pop".

-------------------------- FOUR POPULATIONS ---------------------------------------

8. FourPopulations.generate_bifurcation_diagram(): it is needed to generate one column in Fig. 2B.
It outputs in the "Output_4pop" folder the following files
    - Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#.dat
    DEFAULT PARAMETERS:

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "FourPopulation_generate_bifurcation_diagram.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "FourPopulation_generate_bifurcation_diagram.py"
- the file "Bifurcation_vs_c_size#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_load#_model#.pdf" will appear in the folder "Figures_4pop".

-------------- TWO POPULATIONS WITH ADAPTATION AND PERIODIC INHIBITION --------------

9. AdaptTwoPop.run_dynamics(): this function is needed to generate Fig. S3A,C,E,G and Fig. S4A. It also produce  the file needed to print all the phase-planes diagrams that appear in the paper.
It outputs in the "Output_2pop_adapt" folder the following files
    - check_J0.dat
    - check_thr.dat
    - fp.dat
    - MF_adapt_vs_time_p2_gamma#_time_step#_resolution#_factor#_rm#_A#_h0#_b#_load#.dat
    - nullcline1_gamma#_resolution#_factor#_rm#_A#_h0#_b#_load#_C_hat#.dat
    - quiver.dat
    DEFAULT PARAMETERS:rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 100, h = 1,  min_J0=0.7, max_J0=1.2, T=0.015, Tth=45, TJ0=25, tau_m = 1
    All parameters can be changed manually by specifying them into the brackets. 

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "PLOT_2pop_vs_time_adaptation.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "PLOT_2pop_vs_time_adaptation.py"
- the file "bifurcation_2pop_vs_t_time_step#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_load#_minJ0#_maxJ0#_Chat#.dat" will appear in the folder "Figures_2pop_adapt".

10. AdaptTwoPop.generate_bifurcation_diagram(): this function is needed to generate Fig. S2.
It outputs in the Output_2pop_adapt folder the following files
    - Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_J0#.dat
    DEFAULT PARAMETERS: h0 = 0., rm = 1, A = 1, b = 30, γ = 0.002, resolution = 300, resolution_factor = 1,  size = 50, load = 0 , bound_r = 0.03, J0 = 0.7

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "PLOT_bifurcation_2pop_vs_C_hat_adaptation.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "PLOT_bifurcation_2pop_vs_C_hat_adaptation.py"
- the file "bifurcation_2pop_vs_c_hat_size#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_load#_J0#.dat" will appear in the folder "Figures_2pop_adapt".

11. AdaptTwoPop.compute_critical_C_vs_gamma(): it generates Fig. 5C.
It outputs in the Output_2pop_adapt folder the following files
    - a bunch of files of the type Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_J0#.dat
    - a file of the type Cs_vs_gamma_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#.dat
    DEFAULT PARAMETERS:  rm = 1., A = 1., h0 = 0. , b = 100., bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 10

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Crit_C_vs_gamma_adapt.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Crit_C_vs_gamma_adapt.py"
- the files "Bifurcation_plot_vs_C_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#" and "HomeMadeBifurcation_plot_vs_C_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#.dat" will appear in the folder "Figures_2pop_adapt".

12. AdaptTwoPop.compute_critical_C_vs_b(): it generates Fig. 5D.
It outputs in the Output_2pop_adapt folder the following files
    - a bunch of files of the type Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_J0#.dat
    - a file of the type Cs_vs_b_size#_resolution#_factor#_rm#_A#_h0#_gamma#_load#_MinJ0#_MaxJ0#.dat
    DEFAULT PARAMETERS: γ = 0.002, rm = 1., A = 1., h0 = 0. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 50

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Crit_C_vs_b_adapt.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Crit_C_vs_b_adapt.py"
- the files "Bifurcation_plot_vs_b_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#" and "HomeMadeBifurcation_plot_vs_b_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#.dat" will appear in the folder "Figures_2pop_adapt".

13. AdaptTwoPop.compute_critical_C_vs_h0():  not included in the paper.  
It outputs in the Output_2pop_adapt folder the following files
    - a bunch of files of the type Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_J0#.dat
    - a file of the type Cs_vs_h0_size#_resolution#_factor#_rm#_A#_h0#_gamma#_load#_MinJ0#_MaxJ0#.dat
    DEFAULT PARAMETERS: γ = 0.002, rm = 1., A = 1., b = 100. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0, min_J0 = 0.7, max_J0 = 1.2, size_gamma = 50

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Crit_C_vs_h0_adapt.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Crit_C_vs_h0_adapt.py"
- the files "Bifurcation_plot_vs_h0_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#" and "HomeMadeBifurcation_plot_vs_b_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#.dat" will appear in the folder "Figures_2pop_adapt".

14. AdaptTwoPop.compute_critical_C_vs_J0(): not included in the paper.  
It outputs in the Output_2pop_adapt folder the following files
    - a bunch of files of the type Bifurcation_plot_vs_C_gamma#_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_J0#.dat
    - a file of the type Cs_vs_J0_size#_resolution#_factor#_rm#_A#_h0#_gamma#_load#_MinJ0#_MaxJ0#.dat
    DEFAULT PARAMETERS: γ = 0.002, rm = 1., A = 1., b = 500., h0 = 0. ,  bound_low = -0.05, bound_up = 1.05, bound_r = 10, resolution = 150, resolution_factor = 1.1, size = 50, load = 0,  max_J0 = 1.2, size_gamma = 50

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Crit_C_vs_J0_min_adapt.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Crit_C_vs_J0_min_adapt.py"
- the files "Bifurcation_plot_vs_J0_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#" and "HomeMadeBifurcation_plot_vs_b_size#_resolution#_factor#_rm#_A#_h0#_b#_load#_MinJ0#_MaxJ0#.dat" will appear in the folder "Figures_2pop_adapt".

-------------- FOUR POPULATIONS WITH ADAPTATION AND PERIODIC INHIBITION --------------

15. AdaptFourPop.run_dynamics(): needed to produce Fig. S4B.
It outputes in the Output_4pop_adapt folder the following files
    - J0_vs_time.dat
    - thresholds_vs_time.dat
    - MF_adapt_vs_time_p4_gamma0.002_time_step0.1_resolution200_factor1_rm1_A1_h00_b300_load0.dat
    DEFAULT PARAMETERS: rm = 1, A = 1, h0 = 0, γ = 0.1 , b = 1000, bound_low = -0.05, bound_up = 1.05, bound_r = 0.008, resolution = 100, resolution_factor = 1, load = 0, C_hat = 0. , final_time = 100, h = 1,  min_J0=0.7, max_J0=1.2, T=0.015, Tth=45, TJ0=25, tau_m = 1

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "PLOT_4pop_vs_time_adaptation.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "PLOT_4pop_vs_time_adaptation.py"
- the file "bifurcation_4pop_vs_t_time_step#_resolution_pp_#_resolution_factor_#_rm#_A#_h0#_b#_gamma#_load#_minJ0#_maxJ0#_Chat#.pdf" will appear in the folder "Figures_4pop_adapt".

---------------------------------------------------------------------------
IMPORTANT NOTES: 
1) be careful that parameters' definition matches in the simulation and in the plotting script, 
    that includes int vs float definition. For example, if C_hat = 0. in the simulation script, you shouldn't define C_hat = 0 in the plotting script.
2) The lists of default parameters includes all parametrs taken by the function: don't add paramters which are not in the list. 
3) For parameter b >= 1000, the gain function is approximated by an Heaviside function.

####################################################################
################# FULL NETWORK SIMULATIONS #########################
####################################################################

NEW PARAMETERS:
N = total number of simulated neurons
P = number of memory patterns encoded in the network

16. FullSim.evolve()
It outputs in the Output_full_sim folder the following files
    - overlap_time_neus#_gamma#_P#_tau#_dt#_chat#_A#_rm#_b#.dat
    DEFAULT PARAMETERS: N = 10000, γ = 0.002, P = 10000, n = 2, t_max = 15, t_onset = 0.5, t_offset = 5.5, dt = 0.1,  I1 = 1, tau_s = 1,  rm = 1., b = 100, h0 = 0.25, A = 1., C_hat = 0, dilution = 0.006

Run the function "evolve" 3 times with parameters C_hat = 0.002, C_hat = 0.15 and C_hat = 0.3 and then proceed to the plotting.
To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Full_sim.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Full_sim.py"
- the file "N#_gamma#_P#_A#_rm#_beta#_h0#_tau#_h#_ton#_toff#_C_hat#.pdf" will appear in the folder "Figures_full_sim".

17. FullSim.experiment_proposal_evolve1(): it is needed to reproduce Fig. 8B-C.
It outputs in the "Output_full_sim" folder the following files
    - Exp_proposed_overlap_time_neus#_gamma#_P#_tau#_dt#_chat#_A#_rm#_b#.dat
    DEFAULT PARAMETERS: N = 10000, γ = 0.002, P = 10000, n = 2, t_max = 15, dt = 0.1, tau_s = 1,  rm = 1., b = 100, h0 = 0.25, A = 1., C_hat = 0, dilution = 0.006

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Full_sim_predicted_experiment.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Full_sim_predicted_experiment.py"
- the file "Predicted_experiement_with weak_stim#_toP2P3_N#_gamma#_P#_A#_rm#_beta#_h0#_tau#_h#_C_hat#.pdf" will appear in the folder "Figures_full_sim".

18. FullSim_DitributedGains.evolve(): it is needed to check the heterogeneity of frequency-current  curves, Fig 7A.
It outputs in the "Output_full_sim_distr_gains" folder the following files
    - overlap_time_neus#_gamma#_P#_tau#_dt#_chat#_A#_b#.dat
    DEFAULT PARAMETERS: N = 10000, γ = 0.002, P = 1000, n = 2, t_max = 15, t_onset = 0.5, t_offset = 8, dt = 0.1,  I1 = 1, tau_s = 1,  b = 100.,  A = 1., C_hat = 0., h0 = 0.4, μ_min = 0.25, σ_min = 0.1, μ_max = 1., σ_max = 0.35

Run the function "evolve" 3 times with parameters C_hat = 0., C_hat = 0.15 and C_hat = 0.3 and then proceed to the plotting.
To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Figures_full_sim.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Figures_full_sim.py"
- the file "N#_gamma#_P#_A#_beta#_h0#_tau#_h#_ton#_toff#_C_hat#.pdf" will appear in the folder "Figures_full_sim_distr_gains".

19. AdaptFullSim.evolve(): it is needed to generate Fig. 4B-C.
It outputs in the "Output_full_sim_adapt" folder the following files
    - adapt_overlap_time_neus#_gamma#_P#_tau#_dt#_chat#_A#_rm#_b#.dat
    DEFAULT PARAMETERS: N = 3000, γ = 0.1, P = 16, n = 2, t_max = 1000, t_onset = 0, t_offset = 0, dt = 1,  I1 = 0, tau_s = 1,  rm = 1, b = 2000, h0 = 0, A = 1, C_hat = 0, T=0.015, Tth=45, TJ0=25, min_J0=0.7, max_J0 = 1.2

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Full_sim_adapat.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Full_sim_adapat.py"
- the file "full_sim_neus#_gamma#_P#_A#_rm#_h0#_b#_tau#_h#_C_hat#_tmax#.pdf" will appear in the folder "Figures_full_sim_adapt".

20. AdaptFullSimDistrGains.evolve(): needed for Fig. Fig 7B.
It outputs in the "Output_full_sim_adapt_distr_gains" folder the following files
    - adapt_overlap_time_neus#_gamma#_P#_tau#_dt#_chat#_A#_rm#_b#.dat
    DEFAULT PARAMETERS:  N = 3000, γ = 0., P = 16, n = 2, t_max = 1000, t_onset = 0, t_offset = 0, dt = 1,  I1 = 0, tau_s = 1,  μ_min = 0.1, σ_min = 0.1, μ_max = 1., σ_max = 0.1, b = 1000., h0 = 0, A = 1, C_hat = 0, T=0.015, Tth=45, TJ0=25, min_J0=0.7, max_J0 = 1.2

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "Full_sim_adapt_distr_gains.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "Full_sim_adapt_distr_gains.py"
- the file "N#_gamma#_P#_A#_beta#_h0#_tau#_h#_ton#_toff#_C_hat#.pdf" will appear in the folder "Figures_full_sim_adapt_distr_gains".

####################################################################
########## REPRODUCE COMPARISON WITH EXPERIMENTAL DATA #############
####################################################################

NEW PARAMETERS:
N = total number of simulated neurons
trials = number of repetitions, trial that we repeat the experiment to extrat a mean and standard deviation

To reproduce the experimental data, you will need to use the files in the "Data_analysis" first.
Inside the "Data_analysis" folder, you find a "Data" folder that containes the original experimental data files.

Inside this folder, you find the scripts:
    - Check_prob_distr_correlation_and_sum_equal_1.py: it is a numerical check that the probability distributions of Eq. 99 actually sum up to 1

    - Compare_theory_probs_and_DeFalco2016.py: it plots the figures "mean_of_sessions_gamma#_c#.pdf" and the previous ones in the folder "Data_analysis/Figures"
        it reproduces theoretical predictions in Fig. S6 and the distribution of experimental data. 
        The plots the theoretical predictions in Fig. S6 and "Data_number_of_concepts_a_neu_responds_to_all_pyramidal_and_interneus" will be found in the folder "Data_analysis/Figures"
        NOTE: the data vector defined into the script has been given by Emanuela De Falco and Rodrigo Q. Quiroga and contains the data points in fig. 6 and fig.S6. 
        There are 2 versions of such a vector in the code, corresponding to a smaller and a bigger database.

    - extract_gamma_and_c_from_DeFalco2016.py: it prints the extracted values of c and gamma to the screen
    

21. AttractorNetwork.Reproduce_experiment(): it reproduces fig 6. and S6.: it runs a number 'trials' of virtual experiments with the chose number of neurons, find mean and std.
It outputs in the "Files" folder the following files
    - strict_how_many_neurons_respond_to_how_many_concepts_N#_trials#.dat
    - random_how_many_neurons_respond_to_how_many_concepts_N#_trials#.dat
    - parent_how_many_neurons_respond_to_how_many_concepts_N#_trials#.dat
    - ind_how_many_neurons_respond_to_how_many_concepts_N#_trials#.dat
    DEFAULT PARAMETERS: γ = 0.002, c_hat = 0.04, N = 1000, n_repetitions = 8

To plot the results of this function 
- go to the "Plots" folder 
- open the plotting script "PLOT_experiment_mean_and_std_groups_recalled.py" and make sure that the parameters at the beginning of the file are specified correctly.
- run the plotting script "PLOT_experiment_mean_and_std_groups_recalled.py"
- the file "Final_fig_N#_trials#.pdf" will appear in the folder "Figures_exp".

FINAL NOTES:
--- TwoPopAdapt ---
The maximal firing rate is assumed to be the same for all homogeneous populations {11}, {10}, {01}, {00} and g11.rm is the parameter used as a reference in the code.

--- Full simulation modules ---
There are two functions to generate correlated binary patterns: "generate_binary_patterns_fix_sparseness(p)" and "generate_binary_patterns(p)". 
The figures in the paper are generated with fixed sparsity, and in this the function that is used in "evolve(...)", however it can substituted manually. The alternative function "generate_binary_patterns" is already in place and commented at the moment.
The two modules of full simulation with distributed gains (with and without adaptation) allow a distribution of the gain function parameters across neurons and they don't have a corresponding mean-field module.

--- All modules with adaptation ---
The periodic global inhibition is defined as a Sinusoidal wave (as in the paper). An alternative square wave version can be chosen manually by uncommenting it and comment the sinusoidal version.


