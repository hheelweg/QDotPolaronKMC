# QDotPolaronKMC

Code to evolve the dynamics of excitons on quantum dot lattices with KMC based on the Polaron-transformed Redfield equation (PTRE).

### Usage
The `main.py` functions are in folder `scripts`. In order to execute them (e.g. `[main_name].py`) locally on your computer, naviagte to the root dorectory (`/path/to/QDotPolaronKMC/`) and execute via

```
python -m scripts.[main_name]
```

Execution of `main.py` functions via SLURM should work "as always", just make sure to add the correct path to `main.py`, i.e. `/path/to/QDotPolaronKMC/scripts/[main_name].py` in the submit `.sh` script.

I have also added a `test.py` test script in the root directory that is checking the box construction etc. 