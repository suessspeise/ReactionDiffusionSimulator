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

import os
import argparse
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

# these are default time steps that can be written out regar
DEFAULT_EARLY_STEPS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150]

def make_frame_selector(n_steps, max_frames=250, early_steps=None):
    early = set(early_steps or DEFAULT_EARLY_STEPS)
    interval = max(1, n_steps // max_frames)
    def should_save(step):
        return step in early or step % interval == 0
    return should_save


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

class Renderer:
    """
    Handles rendering and saving of simulation field snapshots.

    Owns all rendering configuration. The simulation and runner
    are entirely agnostic of how or whether frames are saved.

    Parameters
    ----------
    params : dict
        Expected keys:
            colormap        : matplotlib.colors.Colormap
            fix_color_scale : bool
            simulation_name : str
            output_dpi      : int
        Optional keys:
            bg              : str  (default 'black')
    """

    def __init__(self, params: dict):
        self.simulation_name  = params["simulation_name"]
        self.output_directory = self.simulation_name + '_images'
        self.colormap         = params.get("colormap", plt.cm.viridis)
        self.fix_color_scale  = params.get("fix_color_scale", False)
        self.output_dpi       = params.get("output_dpi", 300)
        self.bg               = params.get("bg", "black")
        self.output_directory_created = False

    def _build_path(self, label: str) -> str:
        return os.path.join(self.output_directory, f"{self.simulation_name}_{label}.png")

    def _create_output_directory(self):
        self.output_directory = self.simulation_name + '_images'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.output_directory_created = True

    def _render(self, M: np.ndarray, label: str, colorbar: bool = False):
        """
        Internal: render a single field snapshot and save to disk.

        Uses explicit vmin/vmax instead of sentinel value mutation
        to enforce a fixed color scale.
        """
        if not self.output_directory_created: self._create_output_directory()

        vmin, vmax = (0.0, 1.0) if self.fix_color_scale else (None, None)

        plt.figure()
        plt.rcParams['axes.facecolor']    = self.bg
        plt.rcParams['savefig.facecolor'] = self.bg
        plt.axis('off')

        plt.imshow(M, cmap=self.colormap, extent=[-1, 1, -1, 1],
                   vmin=vmin, vmax=vmax)

        if colorbar:
            plt.colorbar()

        plt.savefig(self._build_path(label), dpi=self.output_dpi)
        plt.close()

    def save_frame(self, step: int, grid: SimulationGrid, colorbar: bool = False):
        """
        Save the current inhibitor field as a numbered frame.

        This is the method intended for use as a runner callback.

        Parameters
        ----------
        step : int
            Current simulation step, used to label the file.
        grid : SimulationGrid
        """
        self._render(grid.v, f"v_{step}", colorbar)

    def save_image(self, label: str, grid: SimulationGrid, colorbar: bool = False):
        """
        Save the current inhibitor field with an arbitrary label.

        Intended for one-off snapshots such as initial and final state.

        Parameters
        ----------
        label : str
            Descriptive label used in the filename.
        grid : SimulationGrid
        """
        self._render(grid.v, label, colorbar)


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

        grid.u[:] += self.time_step * activator_rate
        grid.v[:] += self.time_step * inhibitor_rate

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

        grid.u[:] += self.time_step * activator_rate
        grid.v[:] += self.time_step * inhibitor_rate

class ExperimentLibrary:
    """
    A curated collection of ready-to-run reaction-diffusion experiments.

    Each experiment bundles a model, grid, and renderer with
    sensible defaults.

    Usage
    -----
    library = ExperimentLibrary()
    library.list()
    model, grid, renderer = library.fetch("gray_scott_coral")
    """

    _EXPERIMENTS = {

        # --- Gray-Scott ---

        "gray_scott_solitons": {
            "description": "Stable localised spots that persist and slowly move.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.14,
                "diffusion_v": 0.06,
                "feed_rate":   0.035,
                "kill_rate":   0.065,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.copper, "fix_color_scale": True,
            },
        },

        "gray_scott_coral": {
            "description": "Coral-like branching structures with slow, organic growth.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.16,
                "diffusion_v": 0.08,
                "feed_rate":   0.060,
                "kill_rate":   0.062,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.cubehelix, "fix_color_scale": False,
            },
        },

        "gray_scott_maze": {
            "description": "Dense labyrinthine stripes filling the domain.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.19,
                "diffusion_v": 0.05,
                "feed_rate":   0.060,
                "kill_rate":   0.062,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.cubehelix, "fix_color_scale": False,
            },
        },

        "gray_scott_waves": {
            "description": "Travelling wave fronts that sweep across the domain.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.12,
                "diffusion_v": 0.08,
                "feed_rate":   0.020,
                "kill_rate":   0.050,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.cubehelix, "fix_color_scale": True,
            },
        },

        "gray_scott_flicker": {
            "description": "Unstable spots that appear, flicker, and annihilate.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.16,
                "diffusion_v": 0.08,
                "feed_rate":   0.020,
                "kill_rate":   0.055,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.cubehelix, "fix_color_scale": True,
            },
        },

        "gray_scott_worms": {
            "description": "Worm-like moving filaments with occasional branching.",
            "model_class": GrayScott,
            "model_params": {
                "diffusion_u": 0.16,
                "diffusion_v": 0.08,
                "feed_rate":   0.054,
                "kill_rate":   0.064,
                "time_step":   1.0,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.cubehelix, "fix_color_scale": False,
            },
        },

        # --- Gierer-Meinhardt ---

        "gierer_meinhardt": {
            "description": "Activator-inhibitor system producing spot and stripe Turing patterns.",
            "model_class": GiererMeinhardt,
            "model_params": {
                "diffusion_u":          2.0,
                "diffusion_v":          0.1,
                "reaction_rate":        0.5,
                "saturation_coeff":     0.238,
                "inhibitor_decay_rate": 1.0,
                "activator_decay_rate": 0.9,
                "time_step":            0.1,
            },
            "grid": {
                "grid_size": 200, "grid_spacing": 1.0, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.copper, "fix_color_scale": False,
            },
        },

        # --- FitzHugh-Nagumo ---

        "fitzhugh_nagumo": {
            "description": (
                "Excitable medium producing spiral waves. "
                "Requires fine timestepping — allow at least 80000 steps for patterns to emerge."
            ),
            "model_class": FitzHughNagumo,
            "model_params": {
                "diffusion_recovery":   5e-3,
                "diffusion_excitation": 2.8e-4,
                "drive":                -0.005,
                "recovery_time_scale":  0.1,
                "time_step":            0.9 * (2./120)**2 / 2,  # stability-derived
            },
            "grid": {
                "grid_size": 120, "grid_spacing": 2./120, "seed": "noise",
            },
            "renderer": {
                "colormap": plt.cm.PRGn, "fix_color_scale": False,
            },
        },
    }

    def list(self):
        """Print available experiments with descriptions."""
        for name, spec in self._EXPERIMENTS.items():
            print(f"{name:30s} {spec['description']}")

    def fetch(self, name: str, simulation_name: str = None):
        """
        Assemble and return a (model, grid, renderer) bundle.

        Parameters
        ----------
        name : str
            Experiment name as returned by list().
        simulation_name : str, optional
            Override the output directory name. Defaults to the
            experiment name.

        Returns
        -------
        model : ReactionDiffusionModel
        grid  : SimulationGrid
        renderer : Renderer
        """
        if name not in self._EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{name}'. "
                f"Call list() to see available experiments."
            )

        spec = self._EXPERIMENTS[name]

        grid = SimulationGrid(
            spec["grid"]["grid_size"],
            spec["grid"]["grid_spacing"],
        )
        grid.seed(spec["grid"]["seed"])

        model = spec["model_class"](spec["model_params"])

        render_params = dict(spec["renderer"])
        render_params["simulation_name"] = simulation_name or name
        renderer = Renderer(render_params)

        return model, grid, renderer

# def setModelParams(args, verbose=True):
#     model = args.model
#     grid_size = 200

#     if model == "FN":
#         if verbose: print("FitzHugh-Nagumo model selected")
#         # modelFunc = simulate_fitzhugh_nagumo
#         if verbose: print("SETTING GRID SIZE TO 120 - STABILITY REASONS")
#         if verbose: print("WARNING: IF TIMESTEPS LESS THAN 80000, NO PATTERN MAY APPEAR")
#         #FitzHugh-Nagumo requires fine-timestepping
#         grid_size = 120
#         dx = 2./grid_size
#         dt = 0.9 * dx**2/2
#         params = {"Du":5e-3, "Dv":2.8e-4, "tau":0.1, "k":-0.005,"myCmap":plt.cm.PRGn,"edgeMax":False,"dt":dt,"dx":dx,"seed":"noise"}
#         params = {
#             "diffusion_recovery":   params["Du"],
#             "diffusion_excitation": params["Dv"],
#             "drive":                params["k"],
#             "recovery_time_scale":  params["tau"],
#             "time_step":            params["dt"],
#             "grid_spacing":         params["dx"],
#             "colormap":             params.get("myCmap"),
#             "fix_color_scale":      params.get("edgeMax", False),
#             "seed" : params["seed"],
#         }


#     elif model == "GM":
#         if verbose: print("Gierer-Meinhardt model selected")
#         # modelFunc = simulate_gierer_meinhardt
#         params = {"Du":2, "Dv":0.1, "rho":0.5, "kappa":0.238, "mu":1, "ku":0.9, "kv":1.0, "sv":0.3,
#                   "myCmap":plt.cm.copper,"edgeMax":False,"dt":0.1,"dx":1,"seed":"noise"}
#         params = {
#             "diffusion_u":        params["Du"],
#             "diffusion_v":        params["Dv"],
#             "reaction_rate":      params["rho"],
#             "saturation_coeff":   params["kappa"],
#             "inhibitor_decay_rate": params["mu"],
#             "activator_decay_rate": params["ku"],
#             "time_step":          params["dt"],
#             "grid_spacing":       params["dx"],
#             "colormap":           params.get("myCmap"),
#             "fix_color_scale":    params.get("edgeMax", False),
#             "seed" : params["seed"],
#         }

#     elif model == "GS":
#         if verbose: print("Gray-Scott model selected")
#         # modelFunc = simulate_gray_scott
#         pnames = ["solitons","coral","maze","waves","flicker","worms"]
#         pvals = [{"Du":0.14, "Dv":0.06, "F":0.035, "k":0.065, "myCmap":plt.cm.copper,    "edgeMax":False, "dt":1, "dx":1},
#                  {"Du":0.16, "Dv":0.08, "F":0.060, "k":0.062, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1},
#                  {"Du":0.19, "Dv":0.05, "F":0.060, "k":0.062, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1},
#                  {"Du":0.12, "Dv":0.08, "F":0.020, "k":0.050, "myCmap":plt.cm.cubehelix, "edgeMax":True,  "dt":1, "dx":1},
#                  {"Du":0.16, "Dv":0.08, "F":0.020, "k":0.055, "myCmap":plt.cm.cubehelix, "edgeMax":True,  "dt":1, "dx":1},
#                  {"Du":0.16, "Dv":0.08, "F":0.054, "k":0.064, "myCmap":plt.cm.cubehelix, "edgeMax":False, "dt":1, "dx":1}]

#         pchoices = dict(zip(pnames,pvals))
#         params = pchoices[args.pattern]
#         params["seed"] = args.seed




#         params = {
#             "diffusion_u": params["Du"],
#             "diffusion_v": params["Dv"],
#             "feed_rate": params["F"],
#             "kill_rate": params["k"],
            
#             "time_step": params["dt"],
#             "grid_spacing": params["dx"],
#             "seed" : params["seed"],

#             "colormap": params.get("myCmap"),
#             "fix_color_scale": params.get("edgeMax", False),
            
#         }

#     # this serves no purpose yet, other than to show how 
#     # parameters are going to be separated in the future

#     model_params = { 
#         "grid_size" : grid_size,
#         "time_step": params["time_step"],
#         "grid_spacing": params["grid_spacing"],
#         "seed" : params["seed"],
#         }
#     for k,v in model_params.items():
#         if not k in params.keys(): params[k] = v

#     model_classes = {
#         "FN": FitzHughNagumo,
#         "GM": GiererMeinhardt,
#         "GS": GrayScott,
#     }
#     return params, model_classes[model]

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

    #Get input interactively if no command line args set
    while not args.model:
        print("--- Model choices --- \nFN [FitzHugh-Nagumo]\nGM [Gierer-Meinhardt]\nGS [Gray-Scott]")
        args.model = input("(choose one): ").rstrip().upper()
        if args.model not in ["FN","GM","GS"]:
            print("please enter two-letter model name from list\n")
            args.model = ""

    if args.model == "GS":
        pnames = ["solitons","coral","maze","waves","flicker","worms"]
        pattern = ""
        while not pattern:
            print("--- Pattern choices ---")
            print("pick 1: " + ", ".join(pnames))
            pattern = input('Select reaction-diffusion model: ').rstrip().lower()
            if not pattern in pnames:
                print("please select pattern name from list")
                pattern = ""
        args.pattern = pattern

        seeds = ["single","dual","noise"]
        seed = ""
        while not seed:
            print("Initial seeding choices: " + ", ".join(seeds))
            seed = input("initial seeding choice: ").rstrip().lower()
            if not seed in seeds:
                print("please select seed condition from list")
                seed = ""
        args.seed = seed

    model_classes = {
        "FN": 'fitzhugh_nagumo',
        "GM": 'gierer_meinhardt',
        "GS": 'gray_scott',
    }
    args.experiment_name = f'{model_classes[args.model]}{"_" + args.pattern if hasattr(args, 'pattern') else ""}'
    
    return args


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    
    # read arguments, create basic setup
    args = arg_parse()
    library = ExperimentLibrary()
    model, grid, renderer = library.fetch(args.experiment_name)

    # configure output
    if hasattr(args, 'outname'): renderer.simulation_name = args.outname
    n_steps = args.timesteps
    max_frames=250
    frame_interval = n_steps//max_frames
    print("Running simulation with " + str(n_steps) + " timesteps.")

    save_frames=False
    if args.moviemode:
        print("Movie mode set to ON")
        output_dpi = 200 #reduce image DPI if movie mode
        save_frames = True
        print(str(max_frames) + " movie images will be produced")

    should_save = make_frame_selector(n_steps)
    def callback(step, grid):
        if save_frames and should_save(step):
            renderer.save_frame(step, grid)

    # core: snapshot, running, snapshot
    renderer.save_image("initial_v", grid)
    model.run(grid, n_steps, callback=callback)
    renderer.save_image("final_v", grid)