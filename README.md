# ReactionDiffusionSimulator

## Refactor roadmap

- **Step 1.0 — produce reference output for verification**
- **✅ Step 1.1 — Eliminate the duplicate functions and legacy wrappers** 
- **Step 1.2 — Kill the globals**
Replace `n`, `movieOutput`, `frameMod`, `size`, `myDPI`, `savPath`, `runName` with explicit parameters passed into functions. The `__main__` block assembles them and passes them down.
*Verify: same reference output.*
- **Step 1.3 — Fix the `makeImg` mutation bug**
The `M[1,1]` permanent alteration. Small fix, but do it now before the renderer gets wrapped in a class.
*Verify: same reference output (the bug likely has negligible effect on most runs, but fix it cleanly).*
- **Step 2.1 — Create the class, move data into it**
`SimulationGrid` holds `U`, `V`, `size`, `dx`. Constructor takes `size` and `dx`, initialises the arrays. Nothing else yet.
- **Step 2.2 — Move seeding logic in**
The `single` / `dual` / `noise` initialisation block moves in as a `seed(method)` method.
- **Step 2.3 — Move `laplacian_operator` in**
Becomes `grid.laplacian()`, operates on `self.U` and `self.V`.
*Verify: call `grid.laplacian()` and compare output to the old standalone function on the same arrays.*
- **Step 3.1 — Abstract base class**
A simple `ReactionDiffusionModel` base with an interface: `__init__(params)`, `step(grid)`, `run(grid, n_steps, callback=None)`. The callback is how the runner will hook in frame saving — the model calls `callback(step, grid)` if one is provided, and otherwise does nothing. This keeps the model rendering-agnostic.
- **Step 3.2 — Implement `GrayScott` subclass**
Port `_simulate_gray_scott` into a class. `step(grid)` does one Euler step. `run()` is the loop. 
*Verify: run `GrayScott` end-to-end against reference output.*
- **Step 3.3 — Implement `GiererMeinhardt` and `FitzHughNagumo`**
Same pattern. One at a time, each verified against its own reference output.
- **Step 4.1 — Wrap `makeImg` in a `Renderer` class**
At this stage it still uses matplotlib internally — no Pillow yet. The class just encapsulates the current logic cleanly. Interface: `renderer.save_frame(array, label)`.
- **Step 4.2 — Create `SimulationRunner`**
Takes a model, a grid, a renderer, and run config. Owns the frame cadence logic and the early-frame list. Wires everything together. The `__main__` block becomes: parse args → construct objects → `runner.run()`.
*Verify: full end-to-end run matches reference output.*
- **Step 4.3 — Move parameter presets into model classes**
The `setModelParams` function's preset dictionaries (`solitons`, `coral`, etc.) become class methods or a factory: `GrayScott.from_preset("coral")`. `setModelParams` can then be retired.


#### Summary table

| Phase | What changes | How to verify |
|---|---|---|
| 1.1 | Remove duplicate functions | Reference array comparison |
| 1.2 | Eliminate globals | Reference array comparison |
| 1.3 | Fix mutation bug | Code review |
| 2.x | `SimulationGrid` | Unit test `laplacian()`, seed shapes |
| 3.x | Model classes | Per-model reference array comparison |
| 4.1–4.2 | `Renderer`, `Runner` | Full end-to-end run |
| 4.3 | Preset factory methods | Interactive test, same params selected |
| 5 | Pillow swap | Visual inspection + array values |


### [Watch an example simulation on YouTube.](https://youtu.be/jFM8qlKXyp0)
Simulates Reaction Diffusion models
This tool simulates a number of reaction-diffusion systems and produces [Turing patterns](https://en.wikipedia.org/wiki/Turing_pattern). Optionally, the images may be saved sequentially and can output images to string together in an animation. 

## Requirements
Python with `numpy` and `matplotlib` packages installed.

## Examples 

#### Example usage 1:

    python ReactionDiffusion.py -o my_simulation --moviemode -n 10000

This will output files with the prefix "OutputPrefix" into a directory of the same name, and as
the moviemode flag is set, it will store 250 images for animation. User will be prompted for model type

#### Example usage 2:

    python ReactionDiffusion.py -o my_simulation2 -m GM -n 5000

Instead uses the Gierer-Meinhardt activator-inhibitor model (-gm).

## Supported models
Currently supports the following reaction-diffusion systems (output results of ReactionDiffusion.py shown below).

### Gray-Scott
["Complex Patterns in a Simple System", John Pearson, *Science* 1993](https://www.ljll.math.upmc.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf)

#### Solitons
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/solitons.png" width="640">

#### Coral
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/coral.png" width="640">

#### Maze
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/maze.png" width="640">

#### Waves
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/waves.png" width="640">

#### Flicker
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/flicker.png" width="640">

#### Worms
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/worms.png" width="640">

### FitzHugh-Nagumo
["FitzHugh–Nagumo model", *Wikipedia*](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model)
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/fitzhugh_nagumo.png" width="640">

### Gierer-Meinhardt
["Gierer-Meinhardt model", *Scholarpedia*](http://www.scholarpedia.org/article/Gierer-Meinhardt_model)
<img src="https://github.com/jluebeck/ReactionDiffusionSimulator/blob/master/images/gierer_meinhardt.png" width="640">


