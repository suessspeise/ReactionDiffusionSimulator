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


class SimulationGrid:
    """
    Owns the spatial domain and field arrays for a 2D reaction-diffusion simulation.

    The full arrays U and V include a one-cell boundary ring that remains fixed
    during simulation, allowing the five-point Laplacian stencil to operate on
    interior cells without edge-case logic.

    Parameters
    ----------
    grid_size : int
        Number of cells along each dimension (including boundary).
    grid_spacing : float
        Spatial step size dx (assumed equal in x and y).
    """

    def __init__(self, grid_size: int, grid_spacing: float):
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing

        self.U = np.zeros((grid_size, grid_size))
        self.V = np.zeros((grid_size, grid_size))

        # initialise interior of U to 1.0 (standard baseline)
        self.u[:] = 1.0

    @property
    def u(self):
        """Interior view of U (excludes boundary ring)."""
        return self.U[1:-1, 1:-1]

    @property
    def v(self):
        """Interior view of V (excludes boundary ring)."""
        return self.V[1:-1, 1:-1]

    def seed(self, method: str):
        """
        Set initial conditions on the field arrays.

        Parameters
        ----------
        method : {'single', 'dual', 'noise'}
            single — one central square perturbation
            dual   — two off-centre square perturbations
            noise  — fully random initialisation
        """
        s = self.grid_size

        if method == "single":
            r = 20
            self.U[s//2-r : s//2+r, s//2-r : s//2+r] = 0.50
            self.V[s//2-r : s//2+r, s//2-r : s//2+r] = 0.25

        elif method == "dual":
            r = 15
            self.U[s//4-r   : s//4+r,   s//4-r   : s//4+r]   = 0.50
            self.V[s//4-r   : s//4+r,   s//4-r   : s//4+r]   = 0.25
            self.U[3*s//4-r : 3*s//4+r, 3*s//4-r : 3*s//4+r] = 0.50
            self.V[3*s//4-r : 3*s//4+r, 3*s//4-r : 3*s//4+r] = 0.25

        elif method == "noise":
            interior_size = self.grid_size - 2
            self.u[:] = np.random.rand(interior_size, interior_size)
            self.v[:] = np.random.rand(interior_size, interior_size)

        else:
            raise ValueError(f"Unknown seed method '{method}'. Choose from: single, dual, noise.")

    def laplacian(self):
        """
        Compute the 2D Laplacian of U and V on the interior using a five-point stencil.

        Returns
        -------
        Lu, Lv : ndarray of shape (grid_size-2, grid_size-2)
            Laplacians evaluated at interior points.
        """
        dx2 = self.grid_spacing ** 2
        U, V = self.U, self.V

        Lu = (U[0:-2,1:-1] + U[1:-1,0:-2] + U[1:-1,2:] + U[2:,1:-1] - 4*U[1:-1,1:-1]) / dx2
        Lv = (V[0:-2,1:-1] + V[1:-1,0:-2] + V[1:-1,2:] + V[2:,1:-1] - 4*V[1:-1,1:-1]) / dx2

        return Lu, Lv


def makeImg(M,fname, params, colorbar=False, bg='black'):
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
    if params['fix_color_scale']:
        saved_corner = M[-1,-1]
        saved_inner  = M[1, 1]
        M[-1,-1] = 1
        M[1, 1]  = 0

    plt.imshow(M, cmap=params["colormap"], extent=[-1,1,-1,1]);
    if colorbar:
        plt.colorbar()
    #reset value
    if params['fix_color_scale']:
        M[-1,-1] = saved_corner
        M[1, 1]  = saved_inner

    plt.savefig(params["output_directory"] + params["simulation_name"] + "_" + fname + ".png", dpi=params["output_dpi"])
    plt.close()

from abc import ABC, abstractmethod

class ReactionDiffusionModel(ABC):
    """
    Abstract base class for reaction-diffusion models.

    Subclasses implement the specific reaction kinetics in step(),
    and inherit the run loop from this class. frame saving is handled 
    via an optional callback supplied by the runner.

    Parameters
    ----------
    params : dict
        Model-specific physical parameters.
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def step(self, grid: SimulationGrid):
        """
        Advance the simulation by one time step.

        Modifies grid.U and grid.V in place.

        Parameters
        ----------
        grid : SimulationGrid
        """
        pass

    def run(self, grid: SimulationGrid, n_steps: int, callback=None):
        """
        Run the simulation for n_steps iterations.

        Parameters
        ----------
        grid : SimulationGrid
        n_steps : int
        callback : callable, optional
            Called as callback(step, grid) at intervals determined
            by the runner. The model does not decide when to call it.
        """
        for step in range(n_steps):
            self.step(grid)
            if callback is not None:
                callback(step, grid)

class FitzHughNagumo(ReactionDiffusionModel):
    """
    FitzHugh-Nagumo reaction-diffusion model.

    Expected params keys:
        diffusion_recovery : float
        diffusion_excitation : float
        drive : float
        recovery_time_scale : float
        time_step : float
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.diffusion_recovery   = params["diffusion_recovery"]
        self.diffusion_excitation = params["diffusion_excitation"]
        self.drive                = params["drive"]
        self.recovery_time_scale  = params["recovery_time_scale"]
        self.time_step            = params["time_step"]

    def step(self, grid: SimulationGrid):
        Lu, Lv = grid.laplacian()

        excitation_rate = (self.diffusion_excitation * Lv
                           + grid.v - grid.v**3 - grid.u + self.drive)
        recovery_rate   = (self.diffusion_recovery * Lu
                           + grid.v - grid.u) / self.recovery_time_scale

        grid.u[:] += self.time_step * recovery_rate
        grid.v[:] += self.time_step * excitation_rate

class GrayScott(ReactionDiffusionModel):
    """
    Gray–Scott reaction-diffusion model.

    Expected params keys:
        diffusion_u : float
        diffusion_v : float
        feed_rate : float
        kill_rate : float
        time_step : float
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.diffusion_u = params["diffusion_u"]
        self.diffusion_v = params["diffusion_v"]
        self.feed_rate   = params["feed_rate"]
        self.kill_rate   = params["kill_rate"]
        self.time_step   = params["time_step"]

    def step(self, grid: SimulationGrid):
        Lu, Lv = grid.laplacian()
        activator = grid.u
        inhibitor = grid.v

        reaction = activator * inhibitor * inhibitor # u*v^2
        activator_rate = self.diffusion_u * Lu - reaction + self.feed_rate * (1.0 - activator)
        inhibitor_rate = self.diffusion_v * Lv + reaction - (self.feed_rate + self.kill_rate) * inhibitor

        grid.u[:] += time_step * activator_rate
        grid.v[:] += time_step * inhibitor_rate


class GiererMeinhardt(ReactionDiffusionModel):
    """
    Gierer–Meinhardt activator–inhibitor system reaction-diffusion model.

    Expected params keys:
        diffusion_u : float
        diffusion_v : float
        feed_rate : float
        kill_rate : float
        time_step : float
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.diffusion_u      = params["diffusion_u"]
        self.diffusion_v      = params["diffusion_v"]
        self.reaction_rate    = params["reaction_rate"]          # rho
        self.saturation_coeff = params["saturation_coeff"]       # kappa
        self.inhibitor_decay  = params["inhibitor_decay_rate"]   # mu
        self.activator_decay  = params["activator_decay_rate"]   # k_u
        self.time_step        = params["time_step"]

    def step(self, grid: SimulationGrid):
        Lu, Lv = grid.laplacian()
        activator = grid.u
        inhibitor = grid.v

        inh_sq = inhibitor * inhibitor
        inhibitor_rate = (
            self.reaction_rate * (inh_sq / (activator * (1.0 + self.saturation_coeff * inh_sq)) - self.inhibitor_decay * inhibitor)
            + self.diffusion_v * Lv
        )
        activator_rate = (
            self.reaction_rate * (inh_sq - self.activator_decay * activator)
            + self.diffusion_u * Lu
        )

        grid.u[:] += time_step * activator_rate
        grid.v[:] += time_step * inhibitor_rate


##########################

def simulate_gray_scott(params, grid):
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

    # U, V = initial_fields
    # activator = U[1:-1, 1:-1]
    # inhibitor = V[1:-1, 1:-1]
    activator = grid.u
    inhibitor = grid.v

    for step in range(params["n_steps"]):
        # Lu, Lv = laplacian_operator(U, V, grid_spacing)
        Lu, Lv = grid.laplacian()

        reaction = activator * inhibitor * inhibitor  # u*v^2
        activator_rate = diffusion_u * Lu - reaction + feed_rate * (1.0 - activator)
        inhibitor_rate = diffusion_v * Lv + reaction - (feed_rate + kill_rate) * inhibitor

        activator += time_step * activator_rate
        inhibitor += time_step * inhibitor_rate

        if params["save_frames"]:
            if step % params["frame_interval"] == 0 or step in [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150]:
                makeImg(inhibitor, "v_" + str(step), params)
                if step % 1000 == 0:
                    print(str(step))

    # return activator, inhibitor

def simulate_gierer_meinhardt(params, grid):
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

    # U, V = initial_fields
    # activator = U[1:-1, 1:-1]
    # inhibitor = V[1:-1, 1:-1]
    activator = grid.u
    inhibitor = grid.v

    for step in range(params["n_steps"]):
        # lap_u, lap_v = laplacian_operator(U, V, grid_spacing)
        Lu, Lv = grid.laplacian()

        inh_sq = inhibitor * inhibitor
        inhibitor_rate = (
            reaction_rate * (inh_sq / (activator * (1.0 + saturation_coeff * inh_sq)) - inhibitor_decay * inhibitor)
            + diffusion_v * Lv
        )
        activator_rate = (
            reaction_rate * (inh_sq - activator_decay * activator)
            + diffusion_u * Lu
        )

        activator += time_step * activator_rate
        inhibitor += time_step * inhibitor_rate

        if params["save_frames"]:
            if step % params["frame_interval"] == 0 or step in [1, 2, 3, 4, 5, 10, 40, 80, 150]:
                makeImg(inhibitor, f"v_{step}", params)
                if step % (1000 * params["frame_interval"]) == 0:
                    print(step)

    return activator, inhibitor

def simulate_fitzhugh_nagumo(params, grid):
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

    # U, V = initial_fields
    # recovery = U[1:-1, 1:-1]    # u
    # excitation = V[1:-1, 1:-1]  # v
    recovery = grid.u
    excitation = grid.v

    for step in range(params["n_steps"]):
        # Lu, Lv = laplacian_operator(U, V, grid_spacing)
        Lu, Lv = grid.laplacian() 

        excitation_rate = diffusion_excitation * Lv + excitation - excitation**3 - recovery + drive
        recovery_rate = (diffusion_recovery * Lu + excitation - recovery) / recovery_time_scale

        excitation += time_step * excitation_rate
        recovery   += time_step * recovery_rate

        if params["save_frames"]:
            if step % params["frame_interval"] == 0 or step in [1, 2, 3, 4, 5, 10, 40, 80, 150]:
                makeImg(excitation, f"v_{step}", params)
                if step % (1000 * params["frame_interval"]) == 0:
                    print(step)

    return recovery, excitation


#################



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
    grid_size = 200

    if model == "FN":
        print("FitzHugh-Nagumo model selected")
        modelFunc = simulate_fitzhugh_nagumo
        print("SETTING GRID SIZE TO 120 - STABILITY REASONS")
        print("WARNING: IF TIMESTEPS LESS THAN 80000, NO PATTERN MAY APPEAR")
        #FitzHugh-Nagumo requires fine-timestepping
        grid_size = 120
        dx = 2./grid_size
        dt = 0.9 * dx**2/2
        params = {"Du":5e-3, "Dv":2.8e-4, "tau":0.1, "k":-0.005,"myCmap":plt.cm.PRGn,"edgeMax":False,"dt":dt,"dx":dx,"seed":"noise"}
        params = {
            "diffusion_recovery":   params["Du"],
            "diffusion_excitation": params["Dv"],
            "drive":                params["k"],
            "recovery_time_scale":  params["tau"],
            "time_step":            params["dt"],
            "grid_spacing":         params["dx"],
            "colormap":             params.get("myCmap"),
            "fix_color_scale":      params.get("edgeMax", False),
            "seed" : params["seed"],
        }


    elif model == "GM":
        print("Gierer-Meinhardt model selected")
        modelFunc = simulate_gierer_meinhardt
        params = {"Du":2, "Dv":0.1, "rho":0.5, "kappa":0.238, "mu":1, "ku":0.9, "kv":1.0, "sv":0.3,
                  "myCmap":plt.cm.copper,"edgeMax":False,"dt":0.1,"dx":1,"seed":"noise"}
        params = {
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
            "seed" : params["seed"],
        }

    elif model == "GS":
        print("Gray-Scott model selected")
        modelFunc = simulate_gray_scott
        pnames = ["solitons","coral","maze","waves","flicker","worms"]
        pvals = [{"Du":0.14, "Dv":0.06, "F":0.035, "k":0.065, "myCmap":plt.cm.copper,    "edgeMax":False, "dt":1, "dx":1},
                 {"Du":0.16, "Dv":0.08, "F":0.060, "k":0.062, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1},
                 {"Du":0.19, "Dv":0.05, "F":0.060, "k":0.062, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1},
                 {"Du":0.12, "Dv":0.08, "F":0.020, "k":0.050, "myCmap":plt.cm.cubehelix, "edgeMax":True,  "dt":1, "dx":1},
                 {"Du":0.16, "Dv":0.08, "F":0.020, "k":0.055, "myCmap":plt.cm.cubehelix, "edgeMax":True,  "dt":1, "dx":1},
                 {"Du":0.16, "Dv":0.08, "F":0.054, "k":0.064, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1}]

        pchoices = dict(zip(pnames,pvals))
        pattern = ""
        while not pattern:
            print("--- Pattern choices ---")
            print("pick 1: " + ", ".join(pnames))
            pattern = input('Select reaction-diffusion model: ').rstrip().lower()
            if not pattern in pchoices:
                print("please select pattern name from list")
                pattern = ""

        params = pchoices[pattern]

        #Set initial seed pattern for GS
        seeds = ["single","dual","noise"]
        seed = ""
        while not seed:
            print("Initial seeding choices: " + ", ".join(seeds))
            seed = input("initial seeding choice: ").rstrip().lower()
            if not seed in seeds:
                print("please select seed condition from list")
                seed = ""

        params["seed"] = seed
        params = {
            "diffusion_u": params["Du"],
            "diffusion_v": params["Dv"],
            "feed_rate": params["F"],
            "kill_rate": params["k"],
            
            "time_step": params["dt"],
            "grid_spacing": params["dx"],
            "seed" : params["seed"],

            "colormap": params.get("myCmap"),
            "fix_color_scale": params.get("edgeMax", False),
            
        }

    # this serves no purpose yet, other than to show how 
    # parameters are going to be separated in the future
    model_params = { 
        # "n_steps" : n,
        "grid_size" : grid_size,
        "time_step": params["time_step"],
        "grid_spacing": params["grid_spacing"],
        "seed" : params["seed"],
        }
    for k,v in model_params.items():
        if not k in params.keys(): params[k] = v

    model_classes = {
        "FN": FitzHughNagumo,
        "GM": GiererMeinhardt,
        "GS": GrayScott,
    }
    modelClass = model_classes[model]
    return params, modelFunc, modelClass


def arg_parse():
    #Parses the command line arguments
    parser = argparse.ArgumentParser(description="Gray-Scott simulation")
    parser.add_argument("-o", "--outname", help="simulation run output name",default="simulation_output")
    parser.add_argument("-mov", "--moviemode", action='store_true',help="Run the script in \"movie mode\"\
        to store sequential images (default off)",default=False)
    parser.add_argument("-n", "--timesteps", type=int, help="Number of timesteps to simulate (default 8000)",default=8000)
    parser.add_argument("-m","--model", choices=['FN','GM','GS'], help="Model simulation choice FN [FitzHugh-Nagumo]\
        \nGM [Gierer-Meinhardt]\nGS [Gray-Scott].")
    args = parser.parse_args()
    
    model = args.model

    #Get input interactively if no command line args set
    while not model:
        print("--- Model choices --- \nFN [FitzHugh-Nagumo]\nGM [Gierer-Meinhardt]\nGS [Gray-Scott]")
        model = input("(choose one): ").rstrip().upper()
        if model not in ["FN","GM","GS"]:
            print("please enter two-letter model name from list\n")
            model = ""

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



    #get params for model
    params, modelFunc, modelClass = setModelParams(model)

    # this serves no purpose yet, other than to show how 
    # parameters are going to be separated in the future
    run_config = {
        "save_frames" : movieOutput,
        "frame_interval" : frameMod,
        "output_dpi" : myDPI,
        "output_directory" : savPath,
        "simulation_name" : runName,
        "max_frames" : totFrames, #(implicit in frameMod currently)
    }
    for k,v in run_config.items():
        if not k in params.keys(): params[k] = v
    params["n_steps"] = n


    return params, modelFunc, modelClass

if __name__ == "__main__":
    # params, modelFunc = arg_parse()
    
    # grid = SimulationGrid(params["grid_size"], params["grid_spacing"])
    # grid.seed(params["seed"])

    # makeImg(grid.v, "initial_v", params)
    # modelFunc(params, grid)
    # makeImg(grid.v, "final_v", params)


    params, modelFunc, modelClass = arg_parse()

    grid = SimulationGrid(params["grid_size"], params["grid_spacing"])
    grid.seed(params["seed"])

    model = modelClass(params)

    makeImg(grid.v, "initial_v", params)
    model.run(grid, params["n_steps"])
    makeImg(grid.v, "final_v", params)