# QDotPolaronKMC

Code to evolve the dynamics of excitons on quantum dot lattices with KMC based on the Polaron-transformed Redfield equation (PTRE).

### Installation as Package

The code is written as a python package, called `qdotkmc`. Please do the following in order to install it.
If not having done so already, clone this repository locally with `git clone git@github.com:<ORG_OR_USER>/<REPO>.git`. Navigate into the repo's home directory and install into you python environment in *editable* mode by running

```
pip install -e .
```

Note that desired python environment needs to be activate for this. The `-e` (editable) flag tells pip to link your Python environment to your local repo instead of copying files. Now, if you change Python files in your repo, those changes will immediately affect your installed package. If the package has dependencies (check `pyproject.toml`), you can install those too with `pip install -e ".[dev]"`

To see if it worked, type

```
python -c "import qdotkmc; print(qdotkmc.__file__)"
```
This should point into your local repo directory.


### Usage

All functionalities of the code can be loaded at the top of a python script by simply using `import qdotkmc`. Make sure the package is properly installed according to the installtion instructions above. 

A sample script that runs KMC simulations with specified input parameters is given in `test/test_main.py` and sample SLURM submit scripts are given by `cpu_sample.sh` (for CPU execution) and `gpu_sample.sh` (for GPU execution) in the `jobs` directories as well. The submit script is written to execute `test/test_main.py`. 

Here is what to do: Copy the `.sh` script into the `cwd` (current working directory) where you want to execute a job. This can be anywhere on your local set-up, so using it outside of the code base is encouraged to keep things clean and separated. Also produce a local copy of `test/test_main.py` and put it into the same directory where you have the submit `.sh`script. Modify `test_main.py` desired to have the simulation parameters of choice. Change the path to the python script we want to execute in `.sh` to link the submit script to your version of `test_main.py`. Make sure your python environment is activated so that running `python ...` is not throwing an error. Then you should be all set to just run the job via

```
sbatch <SUBMIT_NAME>.sh
```

This should get the job done and export all output files to `cwd`. All print command are directed to `.log` file in `cwd` specified in `output=test.log` of `.sh` script.


### Functionality

The current version of the code has two different ways of computing rates implemented (switched by `rates_by`). The underlying theory for computing rates of course remains unchanged, *i.e.*, the polaron-transformed Redfield theory, but what changes is the way we perform **truncation** in high-dimensional QD lattices due to increased computational cost:

* `rates_by = "radius"` is the implementation with $r_\mathrm{hop}$ and $r_\mathrm{ove}$, inspired from the Kassal paper - and in alignment with previous implementations of the code. As $r_\mathrm{hop} \to \infty$ and $r_\mathrm{ove} \to \infty$, we approach the full Redfield rates.

* `rates_by = "weight"` is a novel implementation based on $\theta_\mathrm{site}$ and $\theta_\mathrm{pol}$ where we use these two parameters to steer how many destination polaron states $\nu'$ and site state we want to consider for the rate computation at each polaron site $\nu$. As $\theta_\mathrm{site} \to 0$ and $\theta_\mathrm{pol} \to 0$, we recover the full Redfield rates. 

There are currently two types if test scripts in `tests` directory that can serve as templates:

* `test_main.py`: very base-line code to peform a single KMC simulation, setting the rate hyperparameters $r_\mathrm{hop}$ / $r_\mathrm{ove}$ (if `rates_by = "radius"`) or $\theta_\mathrm{site}$ / $\theta_\mathrm{pol}$ (if `rates_by = "weight"`) as fixed and obtaining diffusivities.

* `test_conv.py`: includes an automatic refinement of $\theta_\mathrm{site}$ / $\theta_\mathrm{pol}$ (if `rates_by = "weight"`) to some optimal parameters  $\theta_\mathrm{site}^\ast$ and $\theta_\mathrm{pol}^\ast$ and then uses those parameters in a KMC simulation to obtain diffusivities eventually. This more realistically mimics the workflow as we need to optimize these hyperparameters in order to have a good balance of accuracy and efficiency of our KMC simulations. **Note**: If the simulations take too long to run with $\theta_\mathrm{site}^\ast$ / $\theta_\mathrm{pol}^\ast$, then this could mean that we take into account too many polarons/sites for the rate computation to be reasonably cheap. In this case, you might want to increase the parameter `delta` in `auto_tune_thetas` to $\approx. 10 \%$ for example. 