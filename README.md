# Autonomous Surface Ship Navigation with Chart-Based MPC

A Python re-implementation of an autonomous ship navigation framework originally developed in MATLAB and validated in a peer-reviewed journal (Journal of Marine Science and Engineering, 2025). This system enables autonomous ships to navigate complex coastal environments while avoiding land and other vessels in compliance with the International Regulations for Preventing Collisions at Sea (COLREGs).

## Overview

This project implements a two-level autonomous ship navigation system:

1. **Global Planning**: A grid-based Theta* planner computes collision-free waypoint sequences from map data
2. **Local MPC**: A Model Predictive Control (MPC) controller ensures real-time collision avoidance and COLREG compliance

The system uses a simplified MPC approach that evaluates a finite set of turning rates over a prediction horizon, trading optimality for real-time feasibility while maintaining safety and rule compliance.

## System Model

**State**: The own ship state is $s_t = (x_t, y_t, \psi_t, v_t)$, where $(x, y)$ is the planar position, $\psi$ is the heading, and $v$ is the speed.

**Control**: The control input is the heading change rate $u_t = \Delta \psi_t$, bounded by a maximum turning rate.

**Dynamics**: A discrete-time kinematic ship model:
- $x_{t+1} = x_t + v_t \cos(\psi_t) \Delta t$
- $y_{t+1} = y_t + v_t \sin(\psi_t) \Delta t$
- $\psi_{t+1} = \psi_t + \Delta \psi_t$

**Safety Constraints**: 
- **Collision zone** ($d_{collision} = 0.5$ nm): Minimum safe separation distance
- **COLREG zone** ($d_{COLREG} = 3.0$ nm): Zone where COLREG-compliant maneuvers are required

## How It Works

### MPC Controller

At each time step, the MPC controller:

1. **Generates candidate trajectories**: Creates $M$ candidate trajectories by applying constant turning rates $u \in [-u_{max}, u_{max}]$ over a prediction horizon $H$
2. **Filters infeasible trajectories**: Discards trajectories that violate the minimum-distance constraint with other vessels
3. **Selects best trajectory**: Among feasible trajectories, selects the one that minimizes deviation from the next waypoint while applying COLREG-compliant bias (preferring starboard turns for head-on, crossing, and overtaking scenarios)
4. **Receding horizon**: Applies only the first control input, then re-solves at the next time step with updated observations

### COLREG Compliance

The controller implements COLREG Rules 13, 14, and 15:
- **Rule 13 (Overtaking)**: Overtaking vessel must keep clear, preferably passing on starboard side
- **Rule 14 (Head-on)**: Both vessels turn starboard
- **Rule 15 (Crossing)**: Vessel with traffic on starboard side must give way by turning starboard

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install the Package
Install the package in development mode (in root):
```bash
pip install -e .
```

## Running Scenarios

### Basic Usage

Navigate to the examples directory:
```bash
cd examples/sim
```

### COLREG Test Scenarios

Run individual COLREG test scenarios:

**Head-on encounter (Rule 14)**:
```bash
python test_colreg_headon.py
```

**Crossing from starboard (Rule 15)** - Own ship gives way:
```bash
python test_colreg_crossing_starboard.py
```

**Crossing from port (Rule 15)** - Traffic ship gives way:
```bash
python test_colreg_crossing_port.py
```

**Overtaking (Rule 13)** - Own ship overtakes:
```bash
python test_colreg_overtaking.py
```

**Overtaken (Rule 13)** - Own ship is overtaken:
```bash
python test_colreg_overtaken.py
```

### Command-Line Options

All test scenarios support the following options:

```bash
python test_colreg_headon.py [OPTIONS]
```

Options:
- `--t-final FLOAT`: Simulation duration in seconds (default: 100000.0)
- `--dt FLOAT`: Time step in seconds (default: 1.0)
- `--log-interval INT`: Logging interval in steps (default: 5)
- `--animate`: Save animation as GIF file
- `--no-plot`: Skip matplotlib plot display
- `--no-traffic-colreg`: Disable COLREG compliance for traffic ships

### Example: Run with Animation

```bash
python test_colreg_crossing_starboard.py --animate
```

This will:
1. Run the simulation
2. Display a plot of the trajectories
3. Save an animated GIF to `test_colreg_crossing_starboard.gif`

### Other Scenarios

**Multi-point scenario**:
```bash
python run_multipoint_scenario.py
```

**COLREG benchmark** (runs all COLREG scenarios and aggregates results):
```bash
python run_colreg_benchmark_scenario.py
```

## Project Structure

```
ift6162-final-project/
├── mpc_ship_nav/          # Main package
│   ├── charts/            # Chart environment and global planner
│   ├── dynamics/          # Ship dynamics, COLREG logic, traffic
│   ├── mpc/               # MPC controller implementation
│   └── sim/               # Simulation engine and visualization
├── examples/
│   ├── global_planner/    # Global planning examples
│   └── sim/               # Simulation scenarios
│       ├── test_colreg_*.py    # COLREG test scenarios
│       ├── scenario_utils.py   # Utility functions for scenarios
│       └── run_*.py            # Other scenario scripts
├── data/                  # Geographic data files
└── setup.py              # Package installation script
```

## Limitations

- **Simplified environment**: The simulation does not account for complex factors such as weather conditions, partial observability, or vessel size and maneuverability differences
- **Limited control authority**: The controller acts only on steering (heading changes). In dense traffic scenarios, speed control would be necessary
- **Discretized control**: The simplified MPC evaluates a finite set of constant turning rates, which trades optimality for real-time feasibility

## What to do next
- **Baseline Comparision:** Add a naive controller which uses the global planner with a control selection system that aims to minimize the distance between the ship and the waypoints.
- **Benchamark and metrics** Log the total time of the trajectory and number of collision.


## Reference

Primul Potočnik. "Model Predictive Control for Autonomous Ship Navigation with COLREG Compliance and Chart-Based Path Planning". In: *Journal of Marine Science and Engineering* 13.7 (2025). ISSN: 2077-1312. DOI: 10.3390/jmse13071246. URL: https://www.mdpi.com/2077-1312/13/7/1246

## Authors

- Thierry Bédard-Cortey
- William Chidiac
- Tom Stanic
