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
import warnings
try:
    from PIL import Image
    _HAS_PILLOW = True
except ModuleNotFoundError:
    _HAS_PILLOW = False
    warnings.warn("Pillow not installed; PillowRenderer will be unavailable.", UserWarning, stacklevel=2)

__version__  = "2.0.0"
__author__ = "Jens Luebeck; Hernan Campos"
__url__ = "https://github.com/suessspeise/ReactionDiffusionSimulator/"
__all__      = [
    "SimulationGrid",
    "ReactionDiffusionModel",
    "GrayScott",
    "GiererMeinhardt",
    "FitzHughNagumo",
    "MatplotlibRenderer",
    "ExperimentLibrary",
    "make_frame_selector",
    "DEFAULT_EARLY_STEPS",
] + (["PillowRenderer"] if _HAS_PILLOW else [])


# these are default time steps that can be written out regardle
DEFAULT_EARLY_STEPS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150]

def make_frame_selector(n_steps, max_frames=250, write_out_steps=None):
    """
    Return a function that decides whether a given step should be saved.

    Combines a regular interval (derived from n_steps and max_frames) with
    an explicit list of early steps to capture pattern emergence at the
    start of the simulation.

    Parameters
    ----------
    n_steps : int
        Total number of simulation steps. Used to compute the interval.
    max_frames : int, optional
        Target maximum number of frames to save over the full run.
        Actual count may be slightly higher due to early_steps. Default 250.
    write_out_steps : list of int, optional
        Specific step indices to always save, regardless of interval.
        Defaults to DEFAULT_EARLY_STEPS if not provided.

    Returns
    -------
    should_save : callable
        A function should_save(step: int) -> bool.
    """
    early = set(write_out_steps or DEFAULT_EARLY_STEPS)
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

class MatplotlibRenderer:
    """
    Matplotlib-based renderer for simulation field snapshots.

    Owns all rendering configuration. 

    Parameters
    ----------
    params : dict
        Expected keys:
            simulation_name : str
        Optional keys:
            colormap        : matplotlib.colors.Colormap  (default plt.cm.viridis)
            fix_color_scale : bool                        (default False)
            output_dpi      : int                         (default 300)
            bg              : str                         (default 'black')
    """

    def __init__(self, params: dict):
        self.simulation_name         = params["simulation_name"]
        self.output_directory        = self.simulation_name + "_images"
        self.colormap                = params.get("colormap", plt.cm.viridis)
        self.fix_color_scale         = params.get("fix_color_scale", False)
        self.output_dpi              = params.get("output_dpi", 300)
        self.bg                      = params.get("bg", "black")
        self.output_directory_created = False

    def __iter__(self):
        yield 'simulation_name', self.simulation_name
        yield 'output_directory', self.output_directory
        yield 'fix_color_scale', self.fix_color_scale
        yield 'output_dpi', self.output_dpi
        yield 'bg', self.bg
        yield 'output_directory_created', self.output_directory_created

    def params(self):
        return dict(self)

    def _build_path(self, label: str) -> str:
        return os.path.join(self.output_directory,
                            f"{self.simulation_name}_{label}.png")

    def _create_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.output_directory_created = True

    def _to_image(self, M: np.ndarray) -> tuple:
        """
        Render a 2D float array to a matplotlib (fig, ax) tuple.

        The figure is not saved or displayed — the caller decides
        what to do with it. This is the single override point for
        subclasses that change rendering behaviour.

        Parameters
        ----------
        M : ndarray of shape (H, W)

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes
        """
        vmin, vmax = (0.0, 1.0) if self.fix_color_scale else (None, None)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor(self.bg)
        ax.set_facecolor(self.bg)
        ax.axis("off")
        ax.imshow(M, cmap=self.colormap, extent=[-1, 1, -1, 1],
                  vmin=vmin, vmax=vmax)
        return fig, ax

    def _render(self, M: np.ndarray, label: str):
        """Produce a figure and save it to disk."""
        if not self.output_directory_created:
            self._create_output_directory()
        fig, _ = self._to_image(M)
        fig.savefig(self._build_path(label), dpi=self.output_dpi)
        plt.close(fig)

    def to_image(self, grid: SimulationGrid) -> tuple:
        """
        Return the current inhibitor field as a (fig, ax) tuple without saving.

        Intended for interactive use in Jupyter notebooks. The caller
        has full control over the figure — it can be displayed, annotated,
        or saved to a custom path.

        Parameters
        ----------
        grid : SimulationGrid

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes

        Example
        -------
        fig, ax = renderer.to_image(grid)
        ax.set_title("My simulation")
        fig.savefig("custom_path.png", dpi=150)
        """
        return self._to_image(grid.v)

    def save_frame(self, step: int, grid: SimulationGrid):
        """
        Save the current inhibitor field as a numbered frame.

        Intended for use as a callback during model.run().

        Parameters
        ----------
        step : int
            Current simulation step, used to label the file.
        grid : SimulationGrid
        """
        self._render(grid.v, f"v_{step}")

    def save_image(self, label: str, grid: SimulationGrid):
        """
        Save the current inhibitor field with an arbitrary label.

        Intended for one-off snapshots such as initial and final state.

        Parameters
        ----------
        label : str
            Descriptive label used in the filename.
        grid : SimulationGrid
        """
        self._render(grid.v, label)

if _HAS_PILLOW:
    class PillowRenderer(MatplotlibRenderer):
        """
        Grayscale Pillow-based renderer for simulation field snapshots.

        Produces grayscale PNG images without matplotlib figure overhead,
        making frame saving significantly faster for large movie runs.
        This class can also return PIL.Image objects directly, which display 
        inline in Jupyter notebooks.

        Inherits output directory management, path building, and the
        save_frame / save_image interface from MatplotlibRenderer.

        Parameters
        ----------
        params : dict
            Expected keys:
                simulation_name : str
            Optional keys:
                fix_color_scale    : bool  (default False)
                output_dpi         : int   (default 300)
                output_size_inches : float (default 5.0)
                    # MINOR CHOICE: square output assumed.
                    # Could be a (width, height) tuple later.
        """

        DEFAULT_SIZE_INCHES = 5.0

        def __init__(self, params: dict):
            # Inject neutral defaults so the base class is satisfied
            # without requiring the caller to supply colormap or bg.
            render_params = dict(params)
            render_params.setdefault("colormap", None)
            render_params.setdefault("bg", "black")
            super().__init__(render_params)

            self.size_inches = params.get("output_size_inches", self.DEFAULT_SIZE_INCHES)

        def _to_image(self, M: np.ndarray) -> Image.Image:
            """
            Convert a 2D float array to a grayscale PIL Image.

            Normalisation behaviour mirrors the base Renderer:
            - fix_color_scale=True  : clamp to [0, 1]
            - fix_color_scale=False : normalise to per-frame min/max

            Parameters
            ----------
            M : ndarray of shape (H, W)

            Returns
            -------
            PIL.Image in mode 'L' (8-bit grayscale)
            """
            if self.fix_color_scale:
                normalised = np.clip(M, 0.0, 1.0)
            else:
                lo, hi = M.min(), M.max()
                if hi > lo:
                    normalised = (M - lo) / (hi - lo)
                else:
                    normalised = np.zeros_like(M)

            grey = (normalised * 255).astype(np.uint8)

            pixel_size = int(self.size_inches * self.output_dpi)
            img = Image.fromarray(grey, mode="L")
            # MINOR CHOICE: LANCZOS resampling for downscaling quality.
            # Could be NEAREST for speed, or made configurable.
            img = img.resize((pixel_size, pixel_size), Image.LANCZOS)
            return img

        def _render(self, M: np.ndarray, label: str):
            """Override base _render — produce and save a grayscale image."""
            if not self.output_directory_created:
                self._create_output_directory()
            self._to_image(M).save(self._build_path(label))

        def to_image(self, grid: SimulationGrid) -> Image.Image:
            """
            Return the current inhibitor field as a PIL Image without saving.

            Intended for interactive use in Jupyter notebooks, where the
            returned Image renders inline. Can also be used to save to a
            custom path or pass to further image processing.

            Parameters
            ----------
            grid : SimulationGrid

            Returns
            -------
            PIL.Image in mode 'L' (8-bit grayscale)

            Example
            -------
            img = renderer.to_image(grid)
            img  # displays inline in Jupyter
            """
            return self._to_image(grid.v)
else:
    class PillowRenderer: # stub
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PillowRenderer requires Pillow. Install with: pip install pillow")

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
        diffusion_recovery   : float
        diffusion_excitation : float
        drive                : float
        recovery_time_scale  : float
        time_step            : float
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
        feed_rate  : float
        kill_rate  : float
        time_step  : float
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
        diffusion_u          : float
        diffusion_v          : float
        reaction_rate        : float
        saturation_coeff     : float
        inhibitor_decay_rate : float
        activator_decay_rate : float
        time_step            : float
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

    def __iter__(self):
        return iter(self._EXPERIMENTS)
    
    def __contains__(self, name):
        return name in self._EXPERIMENTS

    def __getitem__(self, name):
        return self._EXPERIMENTS[name]

    def __len__(self):
        return len(self._EXPERIMENTS)

    def __repr__(self):
        return "ExperimentLibrary, contains: " + str(list(self._EXPERIMENTS.keys()))

    def print(self):
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
        renderer : MatplotlibRenderer
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
        renderer = MatplotlibRenderer(render_params)

        return model, grid, renderer

def arg_parse():
    """
    Parse command-line arguments and collect any required interactive input.

    Handles model selection interactively if not supplied via the command
    line. For the Gray-Scott model, also prompts for pattern and seed
    choices, since these have no sensible single default.

    Constructs args.experiment_name as a key into ExperimentLibrary,
    combining model and pattern where applicable.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments. Always contains:
            model          : str  — 'FN', 'GM', or 'GS'
            outname        : str  — output name prefix
            moviemode      : bool
            timesteps      : int
            experiment_name: str  — key for ExperimentLibrary.fetch()
        For GS only:
            pattern        : str  — e.g. 'coral', 'maze'
            seed           : str  — 'single', 'dual', or 'noise'
    """
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
    args.experiment_name = f'{model_classes[args.model]}{"_" + args.pattern if hasattr(args, "pattern") else ""}'
    
    return args

if __name__ == "__main__":
    # make this consistent
    np.random.seed(187)

    # use 'Agg' backend in CLI mode
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
        renderer.output_dpi = 200 #reduce image DPI if movie mode
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