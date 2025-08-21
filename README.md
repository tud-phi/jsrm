# JAX Soft Robot Modeling

This repository contains symbolic derivations of the kinematics and dynamics of various soft robots using Sympy.
The symbolic expressions are then implemented in JAX and can be used for fast, parallelizable, and differentiable simulations.
So far, we have focused on planar settings and implemented the following soft robots:

- [N-link pendulum](examples/simulate_pendulum.py)
- [Planar Piecewise Constant Strain (PCS) continuum soft robot](examples/simulate_planar_pcs.py)
- [Planar Handed Shearing Auxetics (HSA) robot](examples/simulate_planar_hsa.py)

We are happy to receive contributions for other soft robots and/or other settings (e.g., 3D).

## Citation

This simulator is part of the publication **An Experimental Study of Model-based Control
for Planar Handed Shearing Auxetics Robots** presented at the _18th International Symposium on Experimental Robotics_. 
You can find the publication online in the Springer Proceedings on Advanced Robotics (SPAR): https://doi.org/10.1007/978-3-031-63596-0_14

Please use the following citation if you use our software in your (scientific) work:

```bibtex
@inproceedings{stolzle2023experimental,
  title={An experimental study of model-based control for planar handed shearing auxetics robots},
  author={St{\"o}lzle, Maximilian and Rus, Daniela and Della Santina, Cosimo},
  booktitle={International Symposium on Experimental Robotics},
  pages={153--167},
  year={2023},
  organization={Springer}
}
```

## Installation

The plugin can be installed from PyPI:

```bash
pip install jsrm
```

or locally from the source code:

```bash
pip install -e .
```

If you want to run the examples, you will also need to install the following dependencies:

```bash
pip install -e ".[examples]"
```

## Usage

Always first source all necessary environment variables when opening a new terminal:

```bash
source 01-configure-env-vars.sh
```

Then, we can symbolically derive the pendulum kinematics and dynamics:

```bash
python examples/derive_pendulum.py
```

Finally, we can simulate the pendulum
```bash
python examples/simulate_pendulum.py
```

## See also

You might also be interested in the following repositories:
 - The [`jax-spcs-kinematics`](https://github.com/tud-phi/jax-spcs-kinematics) repository contains an implementation
 of the Selective Piecewise Constant Strain (SPCS) kinematics in JAX. We have shown in our paper that this kinematic 
model is suitable for representing the shape of HSA rods.
 - The [`HSA-PyElastica`](https://github.com/tud-phi/HSA-PyElastica) repository contains a plugin for PyElastica
for the simulation of HSA robots.
 - The [`hsa-planar-control`](https://github.com/tud-phi/hsa-planar-control) repository contains JAX and ROS2 implementations
 of model-based control algorithms for planar HSA robots.
