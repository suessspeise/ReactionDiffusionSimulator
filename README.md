# ReactionDiffusionSimulator

## Refactor roadmap

- **✅ 1.0 — produce reference output for verification**
- **✅ 1.1 — Eliminate the duplicate functions and legacy wrappers** 
- **✅ 1.2 — Kill the globals**
- **✅ 1.3 — Fix the `makeImg` mutation bug**
- **✅ 2.1 — Create the class, move data into it**
- **✅ 2.2 — Move seeding logic in**
- **✅ 2.3 — Move `laplacian_operator` in**
- **✅ 3.1 — Abstract base class**
- **✅ 3.2 — Implement `GrayScott` subclass**
- **✅ 3.3 — Implement `GiererMeinhardt` and `FitzHughNagumo`**
- **✅ 3.4 — figure out how to actually use the OOP models, without skipping more steps.**
- **✅ 4.1 — Wrap `makeImg` in a `Renderer` class**
- **✅ 4.2 - Make a function that selects frams for saving**
- **✅ 4.3 — seperate functionality of `arg_parse`**
- **✅ 4.4 — clean up main and encapsulate run config creation**
- **✅ 4.5 - Create ExperimentLibrary class**
- **✅ 4.6 - replace setModelParams with use of Experiment class in main**
- **4.7 - augment functionality of ExperimentLibrary class**
`__repr__`, `__iter__`, `__contains__`, `__getitem__`, `__len__`, name property

- **5.0 implement random seeds**
- **6.0 Pillow Renderer**
- **7.0 augment CLI interface capabilities**
- **8.0 more comprehensive class level documentation**


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


