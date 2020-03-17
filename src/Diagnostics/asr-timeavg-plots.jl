using DataFrames
using FileIO
using PGFPlots
using Printf 
using LaTeXStrings

include("diagnostic_vars.jl")

export run_plot_routine


# User specifies a list of strings corresponding to the variable names 
# (for which they need plots) in the Diagnostics module
output_vars = [
               "z"
               # Velocities
               "u"
               "v"
               "w"
               # Total moisture
               "q_tot"
               "q_liq"
               # Potential temperature grouping
               "thd"                # θ
               "thl"                # θ_liq
               "thv"                # θ_v
               # Energy grouping 
               "e_tot"
               "e_int"
               "h_m"
               "h_t"
               # vertical fluxes
               "qt_sgs"
               "ht_sgs"
               "vert_eddy_mass_flux" # <w′ρ′>
               "vert_eddy_qt_flux"   # <w'q_tot'>
               "vert_qt_flux"        # <w q_tot>
               "vert_eddy_ql_flux"   # <w′q_liq′>
               "vert_eddy_qv_flux"   # <w′q_vap′>
               # Turbulence terms
               "uvariance"          # u′u′
               "vvariance"          # v′v′
               "vert_eddy_u_flux"    # <w′u′>
               "vert_eddy_v_flux"    # <w′v′>
               "wvariance"          # w′w′
               # Potential temperature fluxes
               "vert_eddy_thd_flux"  # <w′θ′>
               "vert_eddy_thv_flux"  # <w′θ_v′>
               "vert_eddy_thl_flux"  # <w′θ_liq′>
               # skewness
               "wskew"              # w′w′w′
               # TKE
               "TKE"
              ]


# Allow user to call the plot function with a String argument for the filename
function run_plot_routine(filename::String; debug::Bool=false)
    # Load data from JLD2 file
    data = load(filename)
    

    @printf("Please enter the directory name for your simulation dataset \n")
    @printf("We want this to be a unique identifier for the sim-set \n")
    @printf("Eventually we automate this and store all plots sequentially \n")
    dirname = readline()

    # Query which timestep the user wants based on whats available
    @show(keys(data))
    # Allow user input here
    @printf("Please enter the starting time-stamp in seconds for which")
    @printf("you wish you create time-averaged plots. Time-avg is currently")
    @printf("taken from your specified time till the end of the simulation. \n")
    @printf("Options for the timestamps are shown above \n")
    @printf("Ignore the quotes when making your choice \n")
    
    user_specified_timestart = readline()
    # Proceed with plot based on user input
    # Give the user some information on how many timesteps are available
    if parse.(Float64,user_specified_timestart) < 0
        @error "Illegal timestart specified. Please select a time from the available timestamp keys \n"
    else
        timestart = user_specified_timestart
    end

    timeend = maximum(parse.(Float64,keys(data)))
    @printf("Taking time-avg from %s to %s \n", user_specified_timestart, timeend)
    # Using the first data key we allocate 
    # Allocate arrays
    Nqk = size(data[user_specified_timestart], 1)
    nvertelem = size(data[user_specified_timestart], 2)
    ntimes = 0
    for key in keys(data)
        # Initialise DataFrame so we can push! sequentially through the list of vars
        if  parse.(Float64,key) > parse.(Float64,timestart)
          ntimes += 1 
        end
    end
    # In ntimes we've recorded the number of timestamp instances available
    df = DataFrame(null=zeros(Nqk*nvertelem))
    for vars in output_vars
        # For each variable, intialise an array of zeros
        V = zeros(Nqk * nvertelem)
        varsymbol = Symbol(vars)
        for key in keys(data)
           if parse.(Float64,key) > parse.(Float64,timestart)
                for ev in 1:nvertelem
                    for k in 1:Nqk
                        dv = diagnostic_vars(data[key][k,ev])
                        V[k+(ev-1)*Nqk] += getproperty(dv, varsymbol)
                    end
                end
            end
        end
        V ./= ntimes
        insertcols!(df, size(df)[2]+1, y=V;makeunique=true)
    end
    select!(df,Not(1)) 
    @assert size(df)[2] == length(output_vars)
    rename!(df,output_vars)

    # Make all the plots :D 
    
    timestring = "time-average"
    plotsdir = "./plots-bomex/"*dirname*timestring
    mkpath(plotsdir) 


    if !debug 

        p1 = Axis([
                   Plots.Linear(df.u,df.z, legendentry=L"$\langle u \rangle$", markSize=0.4, style = "solid"), 
                   Plots.Linear(df.v,df.z, legendentry=L"$\langle v \rangle$", markSize=0.4, style = "solid"), 
                   Plots.Linear(df.w,df.z, legendentry=L"$\langle w \rangle$", markSize=0.4, style = "solid"),
                  ], xlabel = L"$\langle u_{i} \rangle$", ylabel=L"$z$", style="smooth")
        p1.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("Velocity").pdf", p1)

        p2 = Axis([
                Plots.Linear(df.thd,df.z, legendentry=L"$\langle \theta_{dry} \rangle$", markSize=0.4, style="solid"),
                Plots.Linear(df.thl,df.z, legendentry=L"$\langle \theta_{liq} \rangle$", markSize=0.4, style="solid"),
                Plots.Linear(df.thv,df.z, legendentry=L"$\langle \theta_{vap} \rangle$", markSize=0.4, style="solid"),
               ], xlabel = L"$\langle θ \rangle$", ylabel=L"$z$", style="smooth")
        p2.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("PotTemp").pdf", p2)
        
        p3 = Axis([
                   Plots.Linear(df.e_tot, df.z, legendentry=L"$\langle e_{tot} \rangle$", markSize=0.4, style="solid"),
                   Plots.Linear(df.e_int, df.z, legendentry=L"$\langle e_{int} \rangle$", markSize=0.4, style="dashed"),
                   Plots.Linear(df.h_t, df.z, legendentry=L"$\langle h_{tot} \rangle$", markSize=0.4, style="solid"),
                   Plots.Linear(df.h_m, df.z, legendentry=L"$\langle h_{moist} \rangle$", markSize=0.4, style="dashed"),
                  ],xlabel=L"$\langle e \rangle ,\langle h \rangle$", ylabel=L"$z$", style="smooth")
        p3.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("EnergyEnthalpy").pdf", p3)
        
        p4 = Axis([
                   Plots.Linear(df.uvariance, df.z,legendentry=L"$\langle u^{\prime} u^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vvariance, df.z,legendentry=L"$\langle v^{\prime} v^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.wvariance, df.z,legendentry=L"$\langle w^{\prime} w^{\prime} \rangle$", markSize=0.4),
                  ], xlabel=L"$\langle u^{\prime 2}  \rangle$", ylabel=L"$z$", style="smooth")
        p4.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("VelVariance").pdf", p4)
        
        p5 = Axis([
                   Plots.Linear(df.vert_eddy_thd_flux, df.z,legendentry=L"$\langle w^{\prime} \theta_{dry}^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vert_eddy_thl_flux, df.z,legendentry=L"$\langle w^{\prime} \theta_{liq}^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vert_eddy_thv_flux, df.z,legendentry=L"$\langle w^{\prime} \theta_{vap}^{\prime} \rangle$", markSize=0.4),
                  ],xlabel=L"$\langle w^{\prime} \theta^{\prime} \rangle$", ylabel=L"$z$", style="smooth")
        p5.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("PotTempFluxes").pdf", p5)

        p6 = Axis([
                   Plots.Linear(df.wskew, df.z, legendentry=L"$\langle w^{\prime} w^{\prime} w^{\prime} \rangle$", markSize=0.4),
                  ], xlabel=L"$\langle w^{\prime} \rangle$", ylabel=L"$z$", style="smooth")
        p6.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("VertVelSkewness").pdf", p6)
        
        p7 = Axis([
                   Plots.Linear(df.q_liq, df.z, legendentry=L"$\langle q_{liq} \rangle$", markSize=0.4),
                  ], xlabel=L"$\langle q_{liq} \rangle$", ylabel=L"$z$", style="smooth")
        p7.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("MoistureLiquid").pdf", p7)
        
        p8 = Axis([
                   Plots.Linear(df.q_tot, df.z, legendentry=L"$\langle q_{tot} \rangle$", markSize=0.4),
                  ], xlabel = L"$\langle q_{tot} \rangle$",ylabel=L"$z$", style="smooth")
        p8.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("MoistureTotal").pdf", p8)
       
        p9 = Axis([
                   Plots.Linear(df.vert_eddy_mass_flux, df.z, legendentry=L"$\langle w^{\prime} \rho^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vert_eddy_qt_flux, df.z, legendentry=L"$\langle w^{\prime} q_{tot}^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vert_eddy_ql_flux, df.z, legendentry=L"$\langle w^{\prime} q_{liq}^{\prime} \rangle$", markSize=0.4),
                  ], xlabel=L"$\langle w^{\prime}\chi^{\prime} \rangle$", ylabel=L"$z$", style="smooth")
        p9.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("MoistureFluxes").pdf", p9)
        
        p10 = Axis([
                   Plots.Linear(df.vert_eddy_u_flux,df.z, legendentry=L"$\langle w^{\prime} u^{\prime} \rangle$", markSize=0.4),
                   Plots.Linear(df.vert_eddy_v_flux,df.z, legendentry=L"$\langle w^{\prime} v^{\prime} \rangle$", markSize=0.4),
                  ], xlabel=L"$\langle w^{\prime} u_{j}^{\prime} \rangle$", ylabel=L"$z$", style="smooth")
        p10.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("TurbulentFlux").pdf", p10)
        
        p11 = Axis([
                    Plots.Linear((df.uvariance + df.vvariance + df.wvariance)/ 2, df.z, 
                                 legendentry=L"$\langle tke \rangle$", markSize=0.4),
                   ], xlabel = L"$\langle \frac{u_{i}^{\prime} u_{i}^{\prime}}{2} \rangle$", ylabel=L"$z$", style="smooth")
        p11.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("TurbulentKineticEnergy").pdf", p11)
        
        p12 = Axis([
                    Plots.Linear(df.qt_sgs,df.z,  
                                 legendentry=L"$\langle SGS(q_{tot}) \rangle$", markSize=0.4),
                   ], xlabel=L"$\langle D \nabla q_{tot} \rangle$", ylabel=L"$z$", style="smooth")
        p12.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("SGSMoisture").pdf", p12)

        p13 = Axis([
                    Plots.Linear(df.ht_sgs, df.z, 
                                 legendentry=L"$\langle SGS(h_{tot}) \rangle$", markSize=0.4),
                   ], xlabel = L"$\langle D \nabla h_{tot} \rangle$", ylabel=L"$z$", style="smooth")
        p13.legendStyle = "{at={(1.00,1.00)},anchor=north west}"
        PGFPlots.save(plotsdir*"/plot-"*timestring*"-$("SGSEnthalpy").pdf", p13)
    else

      @show("Debug mode...")
      return df
    end
end
  # use graphing framework of choice to plot data in `df`

