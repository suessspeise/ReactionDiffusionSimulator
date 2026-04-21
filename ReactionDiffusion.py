#!/usr/bin/env python

"""
Jens Luebeck
UC San Diego, Bioinformatics & Systems Biology
jluebeck@ucsd.edu

Simulates Gray-Scott reaction diffusion model and can produce images for use in animations

Example usage:

    python ReactionDiffusion.py -o my_simulation --moviemode -n 10000

    This will output files with the prefix "OutputPrefix" into a directory of the same name, and as
the moviemode flag is set, it will store 250 images for animation. User will be prompted for model type

Example usage 2:

    python ReactionDiffusion.py -o my_simulation2 -m GM -n 5000

    Instead uses the Gierer-Meinhardt activator-inhibitor model (-gm).

"""

#Other notes:

#Du: Diffusivity of U
#Dv: Diffusivity of V
#F: "feed rate"
#k: dimensionless rate constant for V
#myCmap: colormap object compatible with matplotlib
#edgeMax: Set a constant for heatmap scaling purposes (True | False)
#dt: timestep
#dx: spatial stepsize (same as dy)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

try:
    # python2
    input_function = raw_input
except NameError:
    input_function = input

def makeImg(M,fname, myCmap, colorbar=False, bg='black', setEdge=True):
    """
    Create and save a heatmap image for a 2D array.

    Parameters
    ----------
    M : ndarray of shape (H, W)
        Data matrix to render. May be modified in-place if `setEdge` is True.
    fname : str
        Basename for the saved file. Path is built from globals `savPath` and `runName`.
    myCmap : matplotlib.colors.Colormap
        Colormap to apply.
    colorbar : bool, default False
        If True, draw a colorbar.
    bg : str, default 'black'
        Figure and canvas background color.
    setEdge : bool, default True
        If True, temporarily sets M[-1, -1]=1 and M[1, 1]=0 to enforce a fixed
        color scale across images.

    Returns
    -------
    None

    Notes
    -----
    - Saves to "{savPath}{runName}_{fname}.png" at DPI `myDPI` (globals).
    - Only M[-1, -1] is restored (to 0); M[1, 1] remains 0, permanently altering `M`.
    - Axes are hidden; image extent is [-1, 1] × [-1, 1].
    """
    plt.figure()
    plt.rcParams['axes.facecolor'] = bg
    plt.rcParams['savefig.facecolor'] = bg
    plt.axis('off')
    #Hackish way to ensure constant color scale across images
    if setEdge:
        M[-1,-1] = 1
        M[1, 1] = 0

    plt.imshow(M, cmap=myCmap, extent=[-1,1,-1,1]);
    if colorbar:
        plt.colorbar()
    #reset value
    if setEdge:
        M[-1,-1] = 0

    plt.savefig(savPath + runName + "_" + fname + ".png",dpi=myDPI)
    plt.close()

def laplacian_operator(U,V,dx):
    """
    Compute 2D Laplacians of U and V on the interior using a five-point stencil.

    Parameters
    ----------
    U, V : ndarray of shape (H, W)
        Input fields.
    dx : float
        Grid spacing (assumed equal in x and y).

    Returns
    -------
    Lu, Lv : ndarray of shape (H-2, W-2)
        Laplacians evaluated at interior points.

    Notes
    -----
    Uses (N + S + E + W - 4C) / dx**2 and excludes boundaries.
    """
    Lu = (U[0:-2,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] + U[2:,1:-1] - 4*U[1:-1,1:-1])/dx**2
    Lv = (V[0:-2,1:-1] + V[1:-1,0:-2] + V[1:-1,2:] + V[2:,1:-1] - 4*V[1:-1,1:-1])/dx**2
    return Lu,Lv

def GS(params, initial_matrices):
    """
    Integrate the Gray–Scott reaction–diffusion system with explicit Euler.

    Parameters
    ----------
    params : dict
        Keys: {'Du', 'Dv', 'F', 'k', 'dt', 'dx'}. May also include {'myCmap', 'edgeMax'}.
    initial_matrices : tuple of ndarray
        (U, V) including boundaries; interior is updated in-place.

    Returns
    -------
    u, v : ndarray
        Interior views U[1:-1, 1:-1], V[1:-1, 1:-1] after `n` steps.

    Notes
    -----
    - Evolves for `n` steps (global).
    - Boundaries remain fixed (not updated).
    - If `movieOutput` (global) is True, periodically saves frames via `makeImg`.
    - Equations: du/dt = Du∇²u − u v² + F(1 − u); dv/dt = Dv∇²v + u v² − (F + k)v.
    """
    Du,Dv,k,F,dt,dx = params['Du'],params['Dv'],params['k'],params['F'],params['dt'],params['dx']
    U,V = initial_matrices
    u,v = U[1:-1,1:-1], V[1:-1,1:-1]
    for i in range(n):
        Lu,Lv = laplacian_operator(U,V,dx)
        uvv = u*v*v
        su = Du*Lu - uvv + F *(1-u)
        sv = Dv*Lv + uvv - (F+k)*v
        u += dt*su
        v += dt*sv

        if movieOutput:
            #Some manually set initial frames to grab so we can see how the system evolves early on
            if i % frameMod == 0 or i in [1,2,3,4,5,10,15,20,30,40,50,60,70,80,90,100,110,120,150]:
                makeImg(v,"v_" + str(i),params["myCmap"],setEdge=params["edgeMax"])
                if i % 1000*frameMod == 0:
                    print(str(i))

    return u,v


def _simulate_gray_scott(params, initial_fields):
    """
    Integrate the Gray–Scott reaction–diffusion system (explicit Euler) with
    human-readable parameter names.

    Parameters
    ----------
    params : dict
        Required keys
            diffusion_u : float
                Diffusion coefficient for the activator (u).
            diffusion_v : float
                Diffusion coefficient for the inhibitor (v).
            feed_rate : float
                Feed rate F.
            kill_rate : float
                Kill/decay rate k for v.
            time_step : float
                Time step Δt.
            grid_spacing : float
                Spatial step Δx (= Δy).
        Optional keys
            colormap : matplotlib.colors.Colormap, optional
                Colormap used when saving frames.
            fix_color_scale : bool, optional
                If True, pass setEdge=True to makeImg for fixed color scaling.

    initial_fields : tuple of ndarray
        (U, V) full arrays (including boundary rows/cols).

    Returns
    -------
    activator, inhibitor : ndarray
        Interior views U[1:-1, 1:-1], V[1:-1, 1:-1] after n steps.

    Notes
    -----
    - Uses globals: n, movieOutput, frameMod, makeImg.
    - Boundaries are not updated (fixed at initial values).
    """
    diffusion_u = params["diffusion_u"]
    diffusion_v = params["diffusion_v"]
    feed_rate = params["feed_rate"]
    kill_rate = params["kill_rate"]
    time_step = params["time_step"]
    grid_spacing = params["grid_spacing"]
    colormap = params.get("colormap")
    fix_color_scale = params.get("fix_color_scale", False)

    U, V = initial_fields
    activator = U[1:-1, 1:-1]
    inhibitor = V[1:-1, 1:-1]

    for step in range(n):
        Lu, Lv = laplacian_operator(U, V, grid_spacing)

        reaction = activator * inhibitor * inhibitor  # u*v^2
        activator_rate = diffusion_u * Lu - reaction + feed_rate * (1.0 - activator)
        inhibitor_rate = diffusion_v * Lv + reaction - (feed_rate + kill_rate) * inhibitor

        activator += time_step * activator_rate
        inhibitor += time_step * inhibitor_rate

        if movieOutput:
            if step % frameMod == 0 or step in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150]:
                makeImg(inhibitor, "v_" + str(step), colormap, setEdge=fix_color_scale)
                if step % 1000 == 0:
                    print(str(step))

    return activator, inhibitor

def GS(params, initial_matrices):
    """
    Backward-compatible wrapper for Gray–Scott using legacy parameter names.

    Parameters
    ----------
    params : dict
        Expected legacy keys: {'Du', 'Dv', 'F', 'k', 'dt', 'dx'}.
        Optional: {'myCmap', 'edgeMax'}.
    initial_matrices : tuple of ndarray
        (U, V) full arrays.

    Returns
    -------
    u, v : ndarray
        Interior views after n steps.
    """
    mapped = {
        "diffusion_u": params["Du"],
        "diffusion_v": params["Dv"],
        "feed_rate": params["F"],
        "kill_rate": params["k"],
        "time_step": params["dt"],
        "grid_spacing": params["dx"],
        "colormap": params.get("myCmap"),
        "fix_color_scale": params.get("edgeMax", False),
    }
    return _simulate_gray_scott(mapped, initial_matrices)



def GM(params, initial_matrices):
    """
    Integrate the Gierer–Meinhardt activator–inhibitor system (explicit Euler).

    Parameters
    ----------
    params : dict
        Keys: {'Du','Dv','rho','kappa','mu','ku','kv','sv','dt','dx'}.
        May also include {'myCmap', 'edgeMax'}.
    initial_matrices : tuple of ndarray
        (U, V) including boundaries; interior is updated in-place.

    Returns
    -------
    u, v : ndarray
        Interior views U[1:-1, 1:-1], V[1:-1, 1:-1] after `n` steps.

    Notes
    -----
    - Evolves for `n` steps (global); boundaries remain fixed.
    - If `movieOutput` (global) is True, periodically saves frames via `makeImg`.
    - Equations: dv/dt = ρ (v² / (u (1+κ v²)) − μ v) + Dv∇²v;
                 du/dt = ρ (v² − k_u u) + Du∇²u.
    """
    Du,Dv,rho,kappa,mu,ku,kv,sv,dt,dx = params['Du'],params['Dv'],params['rho'],params['kappa'],params['mu'],\
        params['ku'],params['kv'],params['sv'],params['dt'],params['dx']
    U,V = initial_matrices
    u,v = U[1:-1,1:-1], V[1:-1,1:-1]
    for i in range(n):
        Lu,Lv = laplacian_operator(U,V,dx)
        vv = v*v
        sv = rho*(vv/(u*(1+kappa*vv)) - mu*v) + Dv*Lv
        su = rho*(vv - ku*u) + Du*Lu
        u += dt*su
        v += dt*sv

        if movieOutput:
            #Some manually set initial frames to grab so we can see how the system evolves early on
            if i % frameMod == 0 or i in [1,2,3,4,5,10,40,80,150]:
                makeImg(v,"v_" + str(i),params["myCmap"],setEdge=params["edgeMax"])
                if i % 1000*frameMod == 0:
                    print(str(i))

    return u,v


def _simulate_gierer_meinhardt(params, initial_fields):
    """
    Integrate the Gierer–Meinhardt activator–inhibitor system (explicit Euler)
    with human-readable parameter names.

    Parameters
    ----------
    params : dict
        Required keys
            diffusion_u : float
                Diffusion coefficient for the activator (u).
            diffusion_v : float
                Diffusion coefficient for the inhibitor (v).
            reaction_rate : float
                Global kinetic scaling (rho).
            saturation_coeff : float
                Saturation coefficient for inhibitor term (kappa).
            inhibitor_decay_rate : float
                Linear decay of inhibitor (mu).
            activator_decay_rate : float
                Linear decay of activator (k_u).
            time_step : float
                Time step Δt.
            grid_spacing : float
                Spatial step Δx (= Δy).
        Optional keys
            colormap : matplotlib.colors.Colormap
            fix_color_scale : bool

    initial_fields : tuple of ndarray
        (U, V) full arrays (including boundary rows/cols).

    Returns
    -------
    activator, inhibitor : ndarray
        Interior views after n steps.

    Notes
    -----
    - Uses globals: n, movieOutput, frameMod, makeImg.
    - Boundaries remain fixed (not updated).
    """
    diffusion_u = params["diffusion_u"]
    diffusion_v = params["diffusion_v"]
    reaction_rate = params["reaction_rate"]          # rho
    saturation_coeff = params["saturation_coeff"]    # kappa
    inhibitor_decay = params["inhibitor_decay_rate"] # mu
    activator_decay = params["activator_decay_rate"] # k_u
    time_step = params["time_step"]
    grid_spacing = params["grid_spacing"]
    colormap = params.get("colormap")
    fix_color_scale = params.get("fix_color_scale", False)

    U, V = initial_fields
    activator = U[1:-1, 1:-1]
    inhibitor = V[1:-1, 1:-1]

    for step in range(n):
        lap_u, lap_v = laplacian_operator(U, V, grid_spacing)

        inh_sq = inhibitor * inhibitor
        inhibitor_rate = (
            reaction_rate * (inh_sq / (activator * (1.0 + saturation_coeff * inh_sq)) - inhibitor_decay * inhibitor)
            + diffusion_v * lap_v
        )
        activator_rate = (
            reaction_rate * (inh_sq - activator_decay * activator)
            + diffusion_u * lap_u
        )

        activator += time_step * activator_rate
        inhibitor += time_step * inhibitor_rate

        if movieOutput:
            if step % frameMod == 0 or step in [1, 2, 3, 4, 5, 10, 40, 80, 150]:
                makeImg(inhibitor, f"v_{step}", colormap, setEdge=fix_color_scale)
                if step % (1000 * frameMod) == 0:
                    print(step)

    return activator, inhibitor

def GM(params, initial_matrices):
    """
    Backward-compatible wrapper for Gierer–Meinhardt using legacy parameter names.

    Parameters
    ----------
    params : dict
        Legacy keys: {'Du','Dv','rho','kappa','mu','ku','kv','sv','dt','dx'}.
        Optional: {'myCmap','edgeMax'}.
        Note: 'kv' and 'sv' are unused by this formulation and are ignored.
    initial_matrices : tuple of ndarray
        (U, V) full arrays.

    Returns
    -------
    u, v : ndarray
        Interior views after n steps.
    """
    # Inform about unused legacy keys (present in defaults, but not used here).
    if "kv" in params or "sv" in params:
        warnings.warn(
            "GM: legacy parameters 'kv' and 'sv' are ignored in this formulation.",
            RuntimeWarning,
            stacklevel=2,
        )

    mapped = {
        "diffusion_u":        params["Du"],
        "diffusion_v":        params["Dv"],
        "reaction_rate":      params["rho"],
        "saturation_coeff":   params["kappa"],
        "inhibitor_decay_rate": params["mu"],
        "activator_decay_rate": params["ku"],
        "time_step":          params["dt"],
        "grid_spacing":       params["dx"],
        "colormap":           params.get("myCmap"),
        "fix_color_scale":    params.get("edgeMax", False),
    }
    return _simulate_gierer_meinhardt(mapped, initial_matrices)

def FN(params, initial_matrices):
    """
    Integrate a FitzHugh–Nagumo-type reaction–diffusion system (explicit Euler).

    Parameters
    ----------
    params : dict
        Keys: {'Du','Dv','k','tau','dt','dx'}. May also include {'myCmap','edgeMax'}.
    initial_matrices : tuple of ndarray
        (U, V) including boundaries; interior is updated in-place.

    Returns
    -------
    u, v : ndarray
        Interior views U[1:-1, 1:-1], V[1:-1, 1:-1] after `n` steps.

    Notes
    -----
    - Evolves for `n` steps (global); boundaries remain fixed.
    - If `movieOutput` (global) is True, periodically saves frames via `makeImg`.
    - Equations: dv/dt = Dv∇²v + v − v³ − u + k;
                 du/dt = (Du∇²u + v − u)/τ.
    """
    U,V = initial_matrices
    u,v = U[1:-1,1:-1], V[1:-1,1:-1]
    Du,Dv,k,tau,dt,dx = params['Du'],params['Dv'],params['k'],params['tau'],params['dt'],params['dx']
    for i in range(n):
        Lu,Lv = laplacian_operator(U,V,dx)
        sv = Dv*Lv + v - v*v*v - u + k
        su = (Du*Lu + v - u)/tau
        u += dt*su
        v += dt*sv

        if movieOutput:
            #Some manually set initial frames to grab so we can see how the system evolves early on
            if i % frameMod == 0 or i in [1,2,3,4,5,10,40,80,150]:
                makeImg(v,"v_" + str(i),params["myCmap"],setEdge=params["edgeMax"])
                if i % 1000*frameMod == 0:
                    print(str(i))

    return u,v


def _simulate_fitzhugh_nagumo(params, initial_fields):
    """
    Integrate the FitzHugh–Nagumo reaction–diffusion system (explicit Euler)
    with human-readable parameter names.

    Parameters
    ----------
    params : dict
        Required keys
            diffusion_recovery : float
                Diffusion coefficient for the recovery variable (u).
            diffusion_excitation : float
                Diffusion coefficient for the excitation variable (v).
            drive : float
                Constant drive/bias term (k) in the v-equation.
            recovery_time_scale : float
                Time scale τ for the recovery variable.
            time_step : float
                Time step Δt.
            grid_spacing : float
                Spatial step Δx (= Δy).
        Optional keys
            colormap : matplotlib.colors.Colormap
            fix_color_scale : bool

    initial_fields : tuple of ndarray
        (U, V) full arrays (including boundary rows/cols).

    Returns
    -------
    recovery, excitation : ndarray
        Interior views after n steps.

    Notes
    -----
    - Uses globals: n, movieOutput, frameMod, makeImg.
    - Boundaries remain fixed (not updated).
    """
    diffusion_recovery = params["diffusion_recovery"]     # Du
    diffusion_excitation = params["diffusion_excitation"] # Dv
    drive = params["drive"]                               # k
    recovery_time_scale = params["recovery_time_scale"]   # tau
    time_step = params["time_step"]                       # dt
    grid_spacing = params["grid_spacing"]                 # dx
    colormap = params.get("colormap")
    fix_color_scale = params.get("fix_color_scale", False)

    U, V = initial_fields
    recovery = U[1:-1, 1:-1]    # u
    excitation = V[1:-1, 1:-1]  # v

    for step in range(n):
        Lu, Lv = laplacian_operator(U, V, grid_spacing)

        excitation_rate = diffusion_excitation * Lv + excitation - excitation**3 - recovery + drive
        recovery_rate = (diffusion_recovery * Lu + excitation - recovery) / recovery_time_scale

        excitation += time_step * excitation_rate
        recovery   += time_step * recovery_rate

        if movieOutput:
            if step % frameMod == 0 or step in [1, 2, 3, 4, 5, 10, 40, 80, 150]:
                makeImg(excitation, f"v_{step}", colormap, setEdge=fix_color_scale)
                if step % (1000 * frameMod) == 0:
                    print(step)

    return recovery, excitation


def FN(params, initial_matrices):
    """
    Backward-compatible wrapper for FitzHugh–Nagumo using legacy parameter names.

    Parameters
    ----------
    params : dict
        Legacy keys: {'Du','Dv','k','tau','dt','dx'}; optional {'myCmap','edgeMax'}.
    initial_matrices : tuple of ndarray
        (U, V) full arrays.

    Returns
    -------
    u, v : ndarray
        Interior views after n steps.
    """
    mapped = {
        "diffusion_recovery":   params["Du"],
        "diffusion_excitation": params["Dv"],
        "drive":                params["k"],
        "recovery_time_scale":  params["tau"],
        "time_step":            params["dt"],
        "grid_spacing":         params["dx"],
        "colormap":             params.get("myCmap"),
        "fix_color_scale":      params.get("edgeMax", False),
    }
    return _simulate_fitzhugh_nagumo(mapped, initial_matrices)

def setModelParams(model, verbose=True):
    """
    Select a model and assemble its default parameter dictionary.

    Parameters
    ----------
    model : {'FN', 'GM', 'GS'}
        Model identifier.
    verbose : bool, default False
        If True, print selection details and warnings.

    Returns
    -------
    params : dict
        Parameter dictionary for the chosen model (includes plotting keys).
    modelFunc : callable
        The corresponding simulator function (FN, GM, or GS).

    Notes
    -----
    - Sets global `size` (FN case).
    - For 'FN', computes dx=2/size and dt=0.9*dx**2/2 for stability.
    - For 'GS', prompts for a parameter preset and an initial seed via stdin.
    """
    global size
    if model == "FN":
        print("FitzHugh-Nagumo model selected")
        modelFunc = FN
        print("SETTING GRID SIZE TO 120 - STABILITY REASONS")
        print("WARNING: IF TIMESTEPS LESS THAN 80000, NO PATTERN MAY APPEAR")
        #FitzHugh-Nagumo requires fine-timestepping
        size = 120
        dx = 2./size
        dt = 0.9 * dx**2/2
        params = {"Du":5e-3, "Dv":2.8e-4, "tau":0.1, "k":-0.005,"myCmap":plt.cm.PRGn,"edgeMax":False,"dt":dt,"dx":dx,"seed":"noise"}

    elif model == "GM":
        print("Gierer-Meinhardt model selected")
        modelFunc = GM
        params = {"Du":2, "Dv":0.1, "rho":0.5, "kappa":0.238, "mu":1, "ku":0.9, "kv":1.0, "sv":0.3,
                  "myCmap":plt.cm.copper,"edgeMax":False,"dt":0.1,"dx":1,"seed":"noise"}

    elif model == "GS":
        print("Gray-Scott model selected")
        modelFunc = GS
        pnames = ["solitons","coral","maze","waves","flicker","worms"]
        pvals = [{"Du":0.14, "Dv":0.06, "F":0.035, "k":0.065,"myCmap":plt.cm.copper,"edgeMax":False,"dt":1,"dx":1},
                {"Du":0.16, "Dv":0.08, "F":0.06, "k":0.062,"myCmap":plt.cm.cubehelix,"edgeMax":False,"dt":1,"dx":1},
                {"Du":0.19, "Dv":0.05, "F":0.06, "k":0.062,"myCmap":plt.cm.cubehelix,"edgeMax":False,"dt":1,"dx":1},
                {"Du":0.12, "Dv":0.08, "F":0.02, "k":0.05,"myCmap":plt.cm.cubehelix,"edgeMax":True,"dt":1,"dx":1},
                {"Du":0.16, "Dv":0.08, "F":0.02, "k":0.055,"myCmap":plt.cm.cubehelix,"edgeMax":True,"dt":1,"dx":1},
                {"Du":0.16, "Dv":0.08, "F":0.054, "k":0.064,"myCmap":plt.cm.cubehelix,"edgeMax":False,"dt":1,"dx":1}]

        pchoices = dict(zip(pnames,pvals))
        pattern = ""
        while not pattern:
            print("--- Pattern choices ---")
            print("pick 1: " + ", ".join(pnames))
            pattern = input_function('Select reaction-diffusion model: ').rstrip().lower()
            if not pattern in pchoices:
                print("please select pattern name from list")
                pattern = ""

        params = pchoices[pattern]

        #Set initial seed pattern for GS
        seeds = ["single","dual","noise"]
        seed = ""
        while not seed:
            print("Initial seeding choices: " + ", ".join(seeds))
            seed = input_function("initial seeding choice: ").rstrip().lower()
            if not seed in seeds:
                print("please select seed condition from list")
                seed = ""

        params["seed"] = seed

    return params, modelFunc


if __name__ == "__main__":
    #Parses the command line arguments
    parser = argparse.ArgumentParser(description="Gray-Scott simulation")
    parser.add_argument("-o", "--outname", help="simulation run output name",default="simulation_output")
    parser.add_argument("-mov", "--moviemode", action='store_true',help="Run the script in \"movie mode\"\
        to store sequential images (default off)",default=False)
    parser.add_argument("-n", "--timesteps", type=int, help="Number of timesteps to simulate (default 8000)",default=8000)
    parser.add_argument("-m","--model", choices=['FN','GM','GS'], help="Model simulation choice FN [FitzHugh-Nagumo]\
        \nGM [Gierer-Meinhardt]\nGS [Gray-Scott].")
    args = parser.parse_args()

    #Set simulation grid size
    size = 200

    model = args.model

    #Get input interactively if no command line args set
    while not model:
        print("--- Model choices --- \nFN [FitzHugh-Nagumo]\nGM [Gierer-Meinhardt]\nGS [Gray-Scott]")
        model = input_function("(choose one): ").rstrip().upper()
        if model not in ["FN","GM","GS"]:
            print("please enter two-letter model name from list\n")
            model = ""


    #get params for model
    params,modelFunc = setModelParams(model)

    #create output dir
    runName = args.outname
    savPath = args.outname + "_images/"
    if not os.path.exists(savPath):
        os.makedirs(savPath)

    #set up image saving
    totFrames = 250
    movieOutput = False
    n = args.timesteps
    print("Running simulation with " + str(n) + " timesteps.")
    frameMod = n//totFrames

    myDPI = 300 #Image resolution DPI
    if args.moviemode:
        print("Movie mode set to ON")
        myDPI = 200 #reduce image DPI if movie mode
        movieOutput = True
        print(str(totFrames) + " movie images will be produced")
    else:
        print("Movie mode set to OFF")


    #set initial conditions
    U = np.zeros((size, size))
    V = np.zeros((size, size))
    u,v = U[1:-1,1:-1], V[1:-1,1:-1]
    u+=1.0

    #sets initialization of single or double squares, or completely random
    if params["seed"] == "single":
        r = 20
        U[size//2-r:size//2+r,size//2-r:size//2+r] = 0.50
        V[size//2-r:size//2+r,size//2-r:size//2+r] = 0.25
    elif params["seed"] == "dual":
        r = 15
        U[size//4-r:size//4+r,size//4-r:size//4+r] = 0.50
        V[size//4-r:size//4+r,size//4-r:size//4+r] = 0.25
        U[3*size//4-r:3*size//4+r,3*size//4-r:3*size//4+r] = 0.50
        V[3*size//4-r:3*size//4+r,3*size//4-r:3*size//4+r] = 0.25

    else: #seed with random noise
        # if not model == 'GM':
        u-=1
        u+=np.random.rand(len(u),len(u))
        v+=np.random.rand(len(u),len(u))

        # else:
        #     #add small amount of noise everywhere
        #     u += (0.01 + 0.01*(np.random.random((size-2,size-2))*2-1))
        #     v += (0.01 + 0.01*(np.random.random((size-2,size-2))*2-1))

    initial_matrices = (U,V)

    #RUN SIM
    makeImg(v,"initial_v",params["myCmap"],setEdge=params["edgeMax"])
    u,v = modelFunc(params,initial_matrices)
    makeImg(v,"final_v",params["myCmap"],setEdge=params["edgeMax"])
