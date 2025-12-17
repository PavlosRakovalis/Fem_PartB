"""
FEM Main Execution Script
=========================
This script runs the complete FEM analysis pipeline:
1. Pre-processor: Creates geometry, boundary conditions, and exports to data file
2. Solver: Solves the FEM system and calculates displacements, forces, and stresses
3. Post-processor: Visualizes results with interactive plots

Author: Rakovalis Pavlos 6931
Date: December 2025
"""

import sys
import os

def main():
    """Execute the complete FEM analysis pipeline"""
    
    print("\n" + "="*80)
    print("FEM ANALYSIS - COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Pre-processor
    print("STEP 1: Running Pre-Processor...")
    print("-"*80)
    import preprocessor
    preprocessor.main()
    
    print("\n" + "="*80 + "\n")
    
    # Step 2: Solver
    print("STEP 2: Running Solver...")
    print("-"*80)
    import solver
    solver.main()
    
    print("\n" + "="*80 + "\n")
    
    # Step 3: Post-processor
    print("STEP 3: Running Post-Processor...")
    print("-"*80)
    import postprocessor
    postprocessor.main()
    
    print("\n" + "="*80)
    print("FEM ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    print("Results available in:")
    print("  - data/structure.dat (input data)")
    print("  - data/structure.res (results)")
    print("  - plots/ (interactive visualizations)")
    print("  - output_html/ (data tables)")
    print("\nOpen the HTML files in plots/ folder to view interactive results.")


if __name__ == "__main__":
    main()
