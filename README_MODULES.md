# FEM Modular System Documentation

## 📋 Overview

This FEM system has been restructured into three independent modules following standard FEM workflow:

1. **Pre-Processor** (`preprocessor.py`) - Geometry and boundary conditions
2. **Solver** (`solver.py`) - FEM system assembly and solution
3. **Post-Processor** (`postprocessor.py`) - Results visualization

## 📁 Project Structure

```
Fem_Assigment/
├── preprocessor.py       # Module 1: Geometry definition and export
├── solver.py             # Module 2: FEM solver
├── postprocessor.py      # Module 3: Results visualization
├── geometry_utils.py     # Utility functions (rotation)
├── transformation.py     # Transformation matrices
├── FEM_Main.py          # Original monolithic code (unchanged)
├── data/                 # Input/output data files
│   ├── structure.dat     # Input for solver (from preprocessor)
│   └── structure.res     # Output from solver (for postprocessor)
└── plots/                # Visualization outputs
```

## 🚀 Usage

### Complete Workflow

Run the three modules in sequence:

```bash
# Step 1: Create geometry and boundary conditions
python preprocessor.py

# Step 2: Solve FEM system
python solver.py

# Step 3: Visualize results
python postprocessor.py
```

### Individual Module Usage

#### 1. Pre-Processor

```python
from preprocessor import FEMPreProcessor

# Create and configure
preprocessor = FEMPreProcessor()
preprocessor.create_geometry()
preprocessor.create_elements()
preprocessor.set_default_bcs_and_loads()

# Visualize geometry
preprocessor.visualize_geometry(show_plot=True)

# Export to file
preprocessor.export_to_file('data/structure.dat')
```

#### 2. Solver

```python
from solver import FEMSolver

# Create solver and load data
solver = FEMSolver()
solver.read_input_file('data/structure.dat')

# Solve system
solver.assemble_stiffness_matrix()
solver.assemble_force_vector()
solver.apply_boundary_conditions()

# Calculate results
solver.calculate_reactions()
solver.calculate_element_forces()

# Export results
solver.export_results('data/structure.res')
```

#### 3. Post-Processor

```python
from postprocessor import FEMPostProcessor

# Create post-processor and load data
postprocessor = FEMPostProcessor()
postprocessor.read_input_file('data/structure.dat')
postprocessor.read_results_file('data/structure.res')

# Create visualizations
postprocessor.plot_deformed_structure(magnification=1.0)
postprocessor.plot_magnified_deformation(magnification=1000)
postprocessor.plot_stress_distribution()
```

## 📝 File Formats

### Input File Format (structure.dat)

```
PARAMETERS
L 1.695000
A 2.028000
phi 69.100000
A_0 3.780000

MATERIALS
STEEL 2.100000e+11 0.300000

NODES
# Node_ID X Y Z
1 2.028000 -0.847500 0.000000
2 2.028000 0.847500 0.000000
...

ELEMENTS
# Elem_ID Node1 Node2 CrossSection E nu
1 1 2 5.670000e-04 2.100000e+11 0.300000
2 3 4 5.670000e-04 2.100000e+11 0.300000
...

BOUNDARY_CONDITIONS
# Node_ID Ux Uy Uz (0=free, 1=fixed)
1 1 1 1
2 1 1 1
...

LOADS
# Node_ID Fx Fy Fz (Newtons)
29 0.000000 0.000000 -2000.000000

END
```

### Results File Format (structure.res)

```
DISPLACEMENTS
# Node_ID Ux(m) Uy(m) Uz(m) Magnitude(m)
1 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
29 -4.790000e-05 -0.000000e+00 -4.135000e-04 4.162000e-04
...

REACTIONS
# Node_ID Rx(N) Ry(N) Rz(N)
1 233.490000 39.650000 277.590000
2 233.490000 -39.650000 277.590000
...

ELEMENT_FORCES
# Elem_ID Axial_Force(N) Stress(Pa) Strain
1 12345.670000 2.178000e+07 1.030000e-04
2 -8901.230000 -4.709800e+07 -2.240000e-04
...

SUMMARY
Total_Applied_Force -2000.000000
Total_Reaction_Force 2000.000000
Max_Displacement 4.135000e-04
Max_Stress 4.709800e+07

END
```

## ✨ Features

### Pre-Processor
- ✅ Automatic geometry generation
- ✅ Element creation (axis-aligned + diagonal bracing)
- ✅ Cross-section assignment based on element length
- ✅ Boundary conditions and loads definition
- ✅ Interactive 3D visualization of geometry
- ✅ Text file export (.dat format)

### Solver
- ✅ Text file input parsing
- ✅ Global stiffness matrix assembly
- ✅ Boundary conditions application
- ✅ Linear system solution
- ✅ Reaction forces calculation
- ✅ Element forces and stresses calculation
- ✅ Results export to text file (.res format)

### Post-Processor
- ✅ Results file reading
- ✅ Deformed structure visualization
- ✅ Magnified deformation plot (1000x)
- ✅ Stress distribution visualization (color-coded)
- ✅ Interactive 3D plots with Plotly
- ✅ HTML export for all visualizations

## 🔧 Customization

### Modifying Geometry

Edit in `preprocessor.py`:

```python
# Change structure parameters
self.L = 1.5 * 1.13  # Length
self.A = 1.2 * 1.69  # Width
self.phi = 60 + 9.1  # Rotation angle
self.A_0 = 6 * (0.5 + 0.13)  # Cross-section factor
```

### Changing Boundary Conditions

```python
# In preprocessor.py, modify set_default_bcs_and_loads()
fixed_nodes = [1, 2, 19, 25, 22, 28]  # Change these nodes
self.add_load(29, fx=0.0, fy=0.0, fz=-2000.0)  # Change load
```

### Adjusting Visualization

```python
# In postprocessor.py
postprocessor.plot_deformed_structure(magnification=10.0)  # Change magnification
postprocessor.plot_magnified_deformation(magnification=500)  # Adjust scale
```

## 🎯 Advantages of Modular Approach

1. **Modularity**: Each component has a clear, single responsibility
2. **Reusability**: Can run solver multiple times with different BCs without recreating geometry
3. **Debugging**: Easier to identify and fix issues in specific modules
4. **Text-based I/O**: Easy to version control, inspect, and modify input files
5. **Standard Workflow**: Follows industry-standard FEM process
6. **Independence**: Original `FEM_Main.py` remains unchanged

## 📊 Comparison with Original Code

| Aspect | Original (FEM_Main.py) | Modular System |
|--------|------------------------|----------------|
| Structure | Monolithic | Modular (3 files) |
| Geometry changes | Re-run everything | Only re-run preprocessor |
| BC changes | Re-run everything | Re-run from solver |
| Visualization | Embedded | Separate module |
| Data format | In-memory only | Text files (.dat, .res) |
| Debugging | Difficult | Easy (isolated modules) |
| Collaboration | Hard to split work | Easy to split work |

## 🔍 Testing

To verify the modules work correctly, compare results with original code:

```bash
# Run original code
python FEM_Main.py

# Run modular system
python preprocessor.py
python solver.py
python postprocessor.py

# Results should match!
```

## 📚 Dependencies

- `numpy` - Numerical computations
- `pandas` - Data structures
- `plotly` - Interactive 3D visualization
- `matplotlib` - (optional, for original code)

## 🐛 Troubleshooting

**Problem**: "File not found" error in solver  
**Solution**: Make sure to run `preprocessor.py` first to generate `data/structure.dat`

**Problem**: "File not found" error in postprocessor  
**Solution**: Make sure to run `solver.py` first to generate `data/structure.res`

**Problem**: Plots not showing  
**Solution**: Check `BROWSER` setting in module headers, set to your browser or `None`

## 📞 Support

For issues or questions, refer to the original `FEM_Main.py` for reference implementation.

---

**Created by**: GitHub Copilot  
**Date**: November 2025  
**Version**: 1.0
