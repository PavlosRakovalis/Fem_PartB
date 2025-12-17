"""
FEM Solver Module
=================
This module reads the structure data from a .dat file, assembles the global
stiffness matrix, solves the FEM system, and exports results to a .res file.

Author: Rakovalis Pavlos 6931
Date: November 2025
"""

import numpy as np
import pandas as pd
import os


class FEMSolver:
    """Handles FEM system assembly and solution"""
    
    def __init__(self):
        self.nodes = pd.DataFrame()
        self.elements = pd.DataFrame()
        self.boundary_conditions = {}
        self.loads = {}
        self.materials = {}
        self.parameters = {}
        
        # Solution arrays
        self.K_global = None
        self.F_global = None
        self.u_global = None
        self.reactions = None
        
    def read_input_file(self, filename='data/structure.dat'):
        """Read structure data from input file"""
        print("="*80)
        print("READING INPUT FILE")
        print("="*80)
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Input file not found: {filename}")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers
            # NOTE: PARAMETERS section removed - now defined in preprocessor only
            if line == 'MATERIALS':
                current_section = 'MATERIALS'
                continue
            elif line == 'NODES':
                current_section = 'NODES'
                nodes_data = []
                continue
            elif line == 'ELEMENTS':
                current_section = 'ELEMENTS'
                elements_data = []
                continue
            elif line == 'BOUNDARY_CONDITIONS':
                current_section = 'BOUNDARY_CONDITIONS'
                continue
            elif line == 'LOADS':
                current_section = 'LOADS'
                continue
            elif line == 'END':
                break
            
            # Parse data based on current section
            # NOTE: PARAMETERS section removed - no longer parsed
            if current_section == 'MATERIALS':
                parts = line.split()
                if len(parts) == 3:
                    mat_name = parts[0]
                    self.materials[mat_name] = {
                        'E': float(parts[1]),
                        'nu': float(parts[2])
                    }
                    
            elif current_section == 'NODES':
                parts = line.split()
                if len(parts) == 4:
                    nodes_data.append({
                        'Node Number': int(parts[0]),
                        'X': float(parts[1]),
                        'Y': float(parts[2]),
                        'Z': float(parts[3])
                    })
                    
            elif current_section == 'ELEMENTS':
                parts = line.split()
                if len(parts) == 6:
                    elements_data.append({
                        'Element Number': int(parts[0]),
                        'Node1': int(parts[1]),
                        'Node2': int(parts[2]),
                        'Cross Section': float(parts[3]),
                        'E': float(parts[4]),
                        'nu': float(parts[5])
                    })
                    
            elif current_section == 'BOUNDARY_CONDITIONS':
                parts = line.split()
                if len(parts) == 4:
                    node_id = int(parts[0])
                    self.boundary_conditions[node_id] = {
                        'Ux': int(parts[1]),
                        'Uy': int(parts[2]),
                        'Uz': int(parts[3])
                    }
                    
            elif current_section == 'LOADS':
                parts = line.split()
                if len(parts) == 4:
                    node_id = int(parts[0])
                    self.loads[node_id] = {
                        'Fx': float(parts[1]),
                        'Fy': float(parts[2]),
                        'Fz': float(parts[3])
                    }
        
        # Create DataFrames
        self.nodes = pd.DataFrame(nodes_data)
        self.elements = pd.DataFrame(elements_data)
        
        print(f"✓ Input file read successfully: {filename}")
        print(f"  - {len(self.nodes)} nodes")
        print(f"  - {len(self.elements)} elements")
        print(f"  - {len(self.boundary_conditions)} boundary conditions")
        print(f"  - {len(self.loads)} loads")
        
    def assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix"""
        print("\nASSEMBLING GLOBAL STIFFNESS MATRIX")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = 3 * num_nodes
        
        self.K_global = np.zeros((n_dof, n_dof))
        
        # Loop through all elements
        for idx in range(len(self.elements)):
            elem = self.elements.iloc[idx]
            
            node1_num = int(elem['Node1'])
            node2_num = int(elem['Node2'])
            E = elem['E']
            A = elem['Cross Section']
            
            # Get node coordinates
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            x1, y1, z1 = node1['X'], node1['Y'], node1['Z']
            x2, y2, z2 = node2['X'], node2['Y'], node2['Z']
            
            # Calculate element length
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Direction cosines
            lx = (x2 - x1) / L
            ly = (y2 - y1) / L
            lz = (z2 - z1) / L
            
            # Local stiffness matrix in global coordinates
            k = (A * E / L)
            K_e = k * np.array([
                [lx**2,    lx*ly,    lx*lz,   -lx**2,   -lx*ly,   -lx*lz  ],
                [lx*ly,    ly**2,    ly*lz,   -lx*ly,   -ly**2,   -ly*lz  ],
                [lx*lz,    ly*lz,    lz**2,   -lx*lz,   -ly*lz,   -lz**2  ],
                [-lx**2,   -lx*ly,   -lx*lz,   lx**2,    lx*ly,    lx*lz  ],
                [-lx*ly,   -ly**2,   -ly*lz,   lx*ly,    ly**2,    ly*lz  ],
                [-lx*lz,   -ly*lz,   -lz**2,   lx*lz,    ly*lz,    lz**2  ]
            ])
            
            # Global DOF indices
            dofs = [
                (node1_num - 1) * 3 + 0,  # Node 1 x
                (node1_num - 1) * 3 + 1,  # Node 1 y
                (node1_num - 1) * 3 + 2,  # Node 1 z
                (node2_num - 1) * 3 + 0,  # Node 2 x
                (node2_num - 1) * 3 + 1,  # Node 2 y
                (node2_num - 1) * 3 + 2   # Node 2 z
            ]
            
            # Assemble into global matrix
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    self.K_global[gi, gj] += K_e[i, j]
        
        print(f"✓ Global stiffness matrix assembled ({n_dof} × {n_dof})")
        print(f"  Non-zero elements: {np.count_nonzero(self.K_global)}")
        print(f"  Matrix condition number: {np.linalg.cond(self.K_global):.2e}")
        
    def assemble_force_vector(self):
        """Assemble global force vector"""
        print("\nASSEMBLING FORCE VECTOR")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = 3 * num_nodes
        
        self.F_global = np.zeros(n_dof)
        
        # Apply loads
        for node_id, load in self.loads.items():
            idx_base = (node_id - 1) * 3
            self.F_global[idx_base + 0] = load['Fx']
            self.F_global[idx_base + 1] = load['Fy']
            self.F_global[idx_base + 2] = load['Fz']
        
        total_force = np.sum(np.abs(self.F_global))
        print(f"✓ Force vector assembled ({n_dof} DOFs)")
        print(f"  Total applied force magnitude: {total_force:.2f} N")
        print(f"  Non-zero force DOFs: {np.count_nonzero(self.F_global)}")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions and solve system"""
        print("\nAPPLYING BOUNDARY CONDITIONS AND SOLVING")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = 3 * num_nodes
        
        # Identify known and unknown DOFs
        known_dofs = []
        unknown_dofs = []
        
        for node_id, bc in self.boundary_conditions.items():
            idx_base = (node_id - 1) * 3
            
            if bc['Ux'] == 1:
                known_dofs.append(idx_base + 0)
            else:
                unknown_dofs.append(idx_base + 0)
                
            if bc['Uy'] == 1:
                known_dofs.append(idx_base + 1)
            else:
                unknown_dofs.append(idx_base + 1)
                
            if bc['Uz'] == 1:
                known_dofs.append(idx_base + 2)
            else:
                unknown_dofs.append(idx_base + 2)
        
        # All other DOFs are free
        for i in range(n_dof):
            if i not in known_dofs and i not in unknown_dofs:
                unknown_dofs.append(i)
        
        print(f"  Fixed DOFs: {len(known_dofs)}")
        print(f"  Free DOFs: {len(unknown_dofs)}")
        
        # Extract reduced system
        K_reduced = self.K_global[np.ix_(unknown_dofs, unknown_dofs)]
        F_reduced = self.F_global[unknown_dofs]
        
        print(f"  Reduced system size: {K_reduced.shape[0]} × {K_reduced.shape[1]}")
        print(f"  Condition number: {np.linalg.cond(K_reduced):.2e}")
        
        # Solve
        print(f"  Solving linear system...")
        try:
            u_unknown = np.linalg.solve(K_reduced, F_reduced)
            print(f"  ✓ System solved successfully")
        except np.linalg.LinAlgError as e:
            print(f"  ✗ Error: {e}")
            u_unknown = np.zeros(len(unknown_dofs))
        
        # Reconstruct full displacement vector
        self.u_global = np.zeros(n_dof)
        self.u_global[known_dofs] = 0.0
        self.u_global[unknown_dofs] = u_unknown
        
        max_disp = np.abs(self.u_global).max()
        print(f"\n✓ Solution obtained")
        print(f"  Max displacement: {max_disp:.6e} m ({max_disp*1000:.6f} mm)")
        
    def calculate_reactions(self):
        """Calculate reaction forces at supports"""
        print("\nCALCULATING REACTION FORCES")
        print("="*80)
        
        # Reactions = K * u - F
        self.reactions = self.K_global @ self.u_global - self.F_global
        
        print(f"✓ Reaction forces calculated")
        
        # Display reactions at support nodes
        for node_id, bc in self.boundary_conditions.items():
            idx_base = (node_id - 1) * 3
            
            rx = self.reactions[idx_base + 0]
            ry = self.reactions[idx_base + 1]
            rz = self.reactions[idx_base + 2]
            
            print(f"  Node {node_id}: Rx={rx:10.2f} N, Ry={ry:10.2f} N, Rz={rz:10.2f} N")
        
        # Check equilibrium
        total_applied = np.sum(self.F_global)
        total_reaction = np.sum(self.reactions)
        difference = abs(total_applied + total_reaction)
        
        print(f"\n  Equilibrium check:")
        print(f"    Total applied force:  {total_applied:12.2f} N")
        print(f"    Total reaction force: {total_reaction:12.2f} N")
        print(f"    Difference:           {difference:12.2e} N")
        
    def calculate_element_forces(self):
        """Calculate axial forces, stresses, and strains in elements"""
        print("\nCALCULATING ELEMENT FORCES AND STRESSES")
        print("="*80)
        
        element_results = []
        
        for idx in range(len(self.elements)):
            elem = self.elements.iloc[idx]
            
            node1_num = int(elem['Node1'])
            node2_num = int(elem['Node2'])
            E = elem['E']
            A = elem['Cross Section']
            
            # Get node coordinates
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            x1, y1, z1 = node1['X'], node1['Y'], node1['Z']
            x2, y2, z2 = node2['X'], node2['Y'], node2['Z']
            
            # Calculate element length
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Direction cosines
            lx = (x2 - x1) / L
            ly = (y2 - y1) / L
            lz = (z2 - z1) / L
            
            # Get displacements
            idx_base_1 = (node1_num - 1) * 3
            idx_base_2 = (node2_num - 1) * 3
            
            u1 = self.u_global[idx_base_1:idx_base_1+3]
            u2 = self.u_global[idx_base_2:idx_base_2+3]
            
            # Axial strain: ε = (u2 - u1) · direction / L
            direction = np.array([lx, ly, lz])
            elongation = np.dot(u2 - u1, direction)
            strain = elongation / L
            
            # Stress: σ = E * ε
            stress = E * strain
            
            # Axial force: F = σ * A
            axial_force = stress * A
            
            element_results.append({
                'Element Number': int(elem['Element Number']),
                'Axial Force (N)': axial_force,
                'Stress (Pa)': stress,
                'Strain': strain
            })
        
        self.element_results = pd.DataFrame(element_results)
        
        max_stress = np.abs(self.element_results['Stress (Pa)']).max()
        max_force = np.abs(self.element_results['Axial Force (N)']).max()
        
        print(f"✓ Element forces calculated for {len(self.elements)} elements")
        print(f"  Max stress: {max_stress:.2e} Pa ({max_stress/1e6:.2f} MPa)")
        print(f"  Max axial force: {max_force:.2f} N")
        
    def export_results(self, filename='data/structure.res'):
        """Export results to text file"""
        print("\nEXPORTING RESULTS")
        print("="*80)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            # Header
            f.write("# FEM Solution Results File\n")
            f.write(f"# Generated by FEM Solver\n")
            f.write(f"# Date: {pd.Timestamp.now()}\n\n")
            
            # Displacements
            f.write("DISPLACEMENTS\n")
            f.write("# Node_ID Ux(m) Uy(m) Uz(m) Magnitude(m)\n")
            for node_num in range(1, len(self.nodes) + 1):
                idx_base = (node_num - 1) * 3
                ux = self.u_global[idx_base + 0]
                uy = self.u_global[idx_base + 1]
                uz = self.u_global[idx_base + 2]
                mag = np.sqrt(ux**2 + uy**2 + uz**2)
                f.write(f"{node_num} {ux:.6e} {uy:.6e} {uz:.6e} {mag:.6e}\n")
            f.write("\n")
            
            # Reactions
            f.write("REACTIONS\n")
            f.write("# Node_ID Rx(N) Ry(N) Rz(N)\n")
            for node_id in self.boundary_conditions.keys():
                idx_base = (node_id - 1) * 3
                rx = self.reactions[idx_base + 0]
                ry = self.reactions[idx_base + 1]
                rz = self.reactions[idx_base + 2]
                f.write(f"{node_id} {rx:.6f} {ry:.6f} {rz:.6f}\n")
            f.write("\n")
            
            # Element forces
            f.write("ELEMENT_FORCES\n")
            f.write("# Elem_ID Axial_Force(N) Stress(Pa) Strain\n")
            for idx in range(len(self.element_results)):
                result = self.element_results.iloc[idx]
                f.write(f"{int(result['Element Number'])} {result['Axial Force (N)']:.6f} "
                       f"{result['Stress (Pa)']:.6e} {result['Strain']:.6e}\n")
            f.write("\n")
            
            # Summary
            f.write("SUMMARY\n")
            total_applied = np.sum(self.F_global)
            total_reaction = np.sum(self.reactions)
            max_disp = np.abs(self.u_global).max()
            max_stress = np.abs(self.element_results['Stress (Pa)']).max()
            
            f.write(f"Total_Applied_Force {total_applied:.6f}\n")
            f.write(f"Total_Reaction_Force {total_reaction:.6f}\n")
            f.write(f"Max_Displacement {max_disp:.6e}\n")
            f.write(f"Max_Stress {max_stress:.6e}\n")
            f.write("\n")
            
            f.write("END\n")
        
        print(f"✓ Results exported to: {filename}")
        print(f"  - {len(self.nodes)} node displacements")
        print(f"  - {len(self.boundary_conditions)} reaction forces")
        print(f"  - {len(self.element_results)} element forces/stresses")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FEM SOLVER")
    print("="*80 + "\n")
    
    # Create solver
    solver = FEMSolver()
    
    # Read input
    solver.read_input_file('data/structure.dat')
    
    # Assemble system
    solver.assemble_stiffness_matrix()
    solver.assemble_force_vector()
    
    # Solve
    solver.apply_boundary_conditions()
    
    # Post-process
    solver.calculate_reactions()
    solver.calculate_element_forces()
    
    # Export
    solver.export_results('data/structure.res')
    
    print("\n" + "="*80)
    print("SOLVER COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    print("Next step: Run postprocessor.py to visualize results")
    print("  Command: python postprocessor.py")


if __name__ == "__main__":
    main()
