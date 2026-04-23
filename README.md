# ReactionDiffusionSimulator

### [Watch an example simulation on YouTube.](https://youtu.be/jFM8qlKXyp0)

Simulates Reaction Diffusion models
This tool simulates a number of reaction-diffusion systems and produces [Turing patterns](https://en.wikipedia.org/wiki/Turing_pattern). Optionally, the images may be saved sequentially and can output images to string together in an animation. 

## Requirements
Python with `numpy` and `matplotlib` packages installed. `Pillow` is optional.

## Examples 

### CLI usage

```py
python ReactionDiffusion.py -o my_simulation --moviemode -n 10000
```

This will output files with the prefix "OutputPrefix" into a directory of the same name, and as
the moviemode flag is set, it will store 250 images for animation. User will be prompted for model type

```py
python ReactionDiffusion.py -o my_simulation2 -m GM -n 5000
```

Instead uses the Gierer-Meinhardt activator-inhibitor model.

### As a module

```py
import ReactionDiffusion

library = ReactionDiffusion.ExperimentLibrary()
model, grid, renderer = library.fetch('gierer_meinhardt')
model.run(grid, 10000)             # run 10000 time steps
renderer.save_image('final', grid) # save state after run
```

See the example notebook.


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

## Notes on the refactor

Version 2.0.0 is a structural refactor of the original procedural script.
The simulation physics and all known experiment configurations are preserved
exactly. The goals were to make the code usable as a library in Jupyter
notebooks.

All parameter names have been updated from terse abbreviations (`Du`, `Dv`,
`k`, `F`, `dt`, `dx`) to descriptive names (`diffusion_u`, `feed_rate`,
`time_step`, `grid_spacing`, etc.). The legacy names no longer appear in the
codebase.
