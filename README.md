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

A sample script that runs KMC simulations with specified input parameters is given in `test/box_main.py` and a sample SLURM submit script is given by `qdot.sh` as well. The submit script is written to execute `test/box_main.py`. 

Here is what to do: Copy the `.sh` script into the `cwd` (current working directory) where you want to execute a job. This can be anywhere on your local set-up, so using it outside of the code base is encouraged to keep things clean and separated. Also produce a local copy of `test/box_main.py` and put it into the same directory where you have the submit `.sh`script. Modify `box_main.py` desired to have the simulation parameters of choice. Change the path to the python script we want to execute in `.sh` to link the submit script to your version of `box_main.py`. Make sure your python environment is activated so that running `python ...` is not throwing an error. Then you should be all set to just run the job via

```
sbatch <SUBMIT_NAME>.sh
```

This should get the job done and export all output files to `cwd`. All print command are directed to `.log` file in `cwd` specified in `output=test.log` of `.sh` script.


### Functionality

The current version of the code does not (yet) include a module to handle proper convergence of the KMC parameters that we use to make the computation of rates computationally more efficient (`r_hop/r_ove`). We assume these parameters as given, future versions will include convergence tests and some auto-tune functionality to obtain these parameters before running the KMC simulation and obtaining credible results. 