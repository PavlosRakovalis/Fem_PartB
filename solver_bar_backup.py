"""
FEM Solver Module - BEAM ELEMENTS
==================================
This module reads the structure data from a .dat file, assembles the global
stiffness matrix for 3D beam elements (6 DOF per node), solves the FEM system,
and exports results to a .res file.

Author: Rakovalis Pavlos 6931
Date: December 2025
"""

import numpy as np
import pandas as pd
import os


class FEMBeamSolver:
    """Handles FEM system assembly and solution for 3D beam elements"""
    
    def __init__(self):
        self.nodes = pd.DataFrame()
        self.elements = pd.DataFrame()
        self.boundary_conditions = {}
        self.loads = {}
        self.materials = {}
        
        # Beam parameters
        self.E = 0.0
        self.A = 0.0
        self.Iz = 0.0
        self.Iy = 0.0
        self.J = 0.0
        self.G = 0.0
        self.diameter = 0.0
        
        # Solution arrays (6 DOF per node)
        self.dof_per_node = 6
        self.K_global = None
        self.F_global = None
        self.u_global = None
        self.reactions = None
        
    def read_input_file(self, filename='data/structure.dat'):
        """Read structure data from input file"""
        print("="*80)
        print("READING INPUT FILE (BEAM ELEMENTS)")
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
            if line == 'PARAMETERS':
                current_section = 'PARAMETERS'
                continue
            elif line == 'MATERIALS':
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
            if current_section == 'PARAMETERS':
                parts = line.split()
                if len(parts) >= 7:
                    self.E = float(parts[0])
                    self.A = float(parts[1])
                    self.Iz = float(parts[2])
                    self.Iy = float(parts[3])
                    self.J = float(parts[4])
                    self.G = float(parts[5])
                    self.diameter = float(parts[6])
                    
            elif current_section == 'MATERIALS':
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
                if len(parts) == 7:  # Ux Uy Uz Rx Ry Rz
                    node_id = int(parts[0])
                    self.boundary_conditions[node_id] = {
                        'Ux': int(parts[1]),
                        'Uy': int(parts[2]),
                        'Uz': int(parts[3]),
                        'Rx': int(parts[4]),
                        'Ry': int(parts[5]),
                        'Rz': int(parts[6])
                    }
                    
            elif current_section == 'LOADS':
                parts = line.split()
                if len(parts) == 7:  # Fx Fy Fz Mx My Mz
                    node_id = int(parts[0])
                    self.loads[node_id] = {
                        'Fx': float(parts[1]),
                        'Fy': float(parts[2]),
                        'Fz': float(parts[3]),
                        'Mx': float(parts[4]),
                        'My': float(parts[5]),
                        'Mz': float(parts[6])
                    }
        
        # Create DataFrames
        self.nodes = pd.DataFrame(nodes_data)
        self.elements = pd.DataFrame(elements_data)
        
        print(f"✓ Input file read successfully: {filename}")
        print(f"  - {len(self.nodes)} nodes")
        print(f"  - {len(self.elements)} elements")
        print(f"  - {len(self.boundary_conditions)} boundary conditions")
        print(f"  - {len(self.loads)} loads")
        print(f"\\n  Beam Properties:")
        print(f"    Diameter: {self.diameter*1000:.2f} mm")
        print(f"    Area: {self.A*1e6:.4f} mm²")
        print(f"    Iz = Iy: {self.Iz*1e12:.4f} mm⁴")
        print(f"    J: {self.J*1e12:.4f} mm⁴")
        
    def _compute_transformation_matrix(self, x1, y1, z1, x2, y2, z2):
        """Compute 3x3 transformation matrix for beam local coordinate system"""
        # Element length and direction cosines
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Local x-axis (along beam element)
        x_local = np.array([dx/L, dy/L, dz/L])
        
        # Define auxiliary vector for determining local y and z axes
        # Use global Y-axis as reference, unless element is parallel to Y
        if abs(x_local[1]) > 0.9:  # Nearly vertical element
            aux = np.array([1.0, 0.0, 0.0])  # Use global X as auxiliary
        else:
            aux = np.array([0.0, 1.0, 0.0])  # Use global Y as auxiliary
        
        # Local z-axis (perpendicular to x_local and aux)
        z_local = np.cross(x_local, aux)
        z_local = z_local / np.linalg.norm(z_local)
        
        # Local y-axis (perpendicular to x and z)
        y_local = np.cross(z_local, x_local)
        y_local = y_local / np.linalg.norm(y_local)
        
        # 3x3 transformation matrix [x_local, y_local, z_local] as rows
        T = np.vstack([x_local, y_local, z_local])
        
        return T, L
    
    def _expand_transformation_matrix(self, T):
        """Expand 3x3 transformation to 12x12 for 6 DOF per node"""
        T_12x12 = np.zeros((12, 12))
        # Apply transformation to each 3x3 block (translations and rotations)
        for i in range(4):
            T_12x12[3*i:3*i+3, 3*i:3*i+3] = T
        return T_12x12
    
    def _create_local_stiffness_matrix(self, L):
        """Create 12x12 local stiffness matrix for 3D beam element"""
        k_local = np.zeros((12, 12))
        
        # Axial stiffness (DOF 0 and 6)
        EA_L = self.E * self.A / L
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L
        
        # Torsional stiffness (DOF 3 and 9)
        GJ_L = self.G * self.J / L
        k_local[3, 3] = GJ_L
        k_local[3, 9] = -GJ_L
        k_local[9, 3] = -GJ_L
        k_local[9, 9] = GJ_L
        
        # Bending in local XY plane (about Z axis) - DOF 1, 5, 7, 11
        EIz_L3 = self.E * self.Iz / (L**3)
        k_local[1, 1] = 12 * EIz_L3
        k_local[1, 5] = 6 * EIz_L3 * L
        k_local[1, 7] = -12 * EIz_L3
        k_local[1, 11] = 6 * EIz_L3 * L
        
        k_local[5, 1] = 6 * EIz_L3 * L
        k_local[5, 5] = 4 * EIz_L3 * L**2
        k_local[5, 7] = -6 * EIz_L3 * L
        k_local[5, 11] = 2 * EIz_L3 * L**2
        
        k_local[7, 1] = -12 * EIz_L3
        k_local[7, 5] = -6 * EIz_L3 * L
        k_local[7, 7] = 12 * EIz_L3
        k_local[7, 11] = -6 * EIz_L3 * L
        
        k_local[11, 1] = 6 * EIz_L3 * L
        k_local[11, 5] = 2 * EIz_L3 * L**2
        k_local[11, 7] = -6 * EIz_L3 * L
        k_local[11, 11] = 4 * EIz_L3 * L**2
        
        # Bending in local XZ plane (about Y axis) - DOF 2, 4, 8, 10
        EIy_L3 = self.E * self.Iy / (L**3)
        k_local[2, 2] = 12 * EIy_L3
        k_local[2, 4] = -6 * EIy_L3 * L
        k_local[2, 8] = -12 * EIy_L3
        k_local[2, 10] = -6 * EIy_L3 * L
        
        k_local[4, 2] = -6 * EIy_L3 * L
        k_local[4, 4] = 4 * EIy_L3 * L**2
        k_local[4, 8] = 6 * EIy_L3 * L
        k_local[4, 10] = 2 * EIy_L3 * L**2
        
        k_local[8, 2] = -12 * EIy_L3
        k_local[8, 4] = 6 * EIy_L3 * L
        k_local[8, 8] = 12 * EIy_L3
        k_local[8, 10] = 6 * EIy_L3 * L
        
        k_local[10, 2] = -6 * EIy_L3 * L
        k_local[10, 4] = 2 * EIy_L3 * L**2
        k_local[10, 8] = 6 * EIy_L3 * L
        k_local[10, 10] = 4 * EIy_L3 * L**2
        
        return k_local
    
    def assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix for beam elements"""
        print("\\nASSEMBLING GLOBAL STIFFNESS MATRIX (BEAM ELEMENTS)")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = self.dof_per_node * num_nodes  # 6 DOF per node
        
        self.K_global = np.zeros((n_dof, n_dof))
        
        # Loop through all elements
        for idx in range(len(self.elements)):
            elem = self.elements.iloc[idx]
            
            node1_num = int(elem['Node1'])
            node2_num = int(elem['Node2'])
            
            # Get node coordinates
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            x1, y1, z1 = node1['X'], node1['Y'], node1['Z']
            x2, y2, z2 = node2['X'], node2['Y'], node2['Z']
            
            # Compute transformation matrix and length
            T, L = self._compute_transformation_matrix(x1, y1, z1, x2, y2, z2)
            
            # Create local stiffness matrix
            k_local = self._create_local_stiffness_matrix(L)
            
            # Expand transformation matrix to 12x12
            T_12x12 = self._expand_transformation_matrix(T)
            
            # Transform to global coordinates
            k_global = T_12x12.T @ k_local @ T_12x12
            
            # Global DOF indices (6 per node)
            dofs = [
                (node1_num - 1) * 6 + 0,  # Node 1 Ux
                (node1_num - 1) * 6 + 1,  # Node 1 Uy
                (node1_num - 1) * 6 + 2,  # Node 1 Uz
                (node1_num - 1) * 6 + 3,  # Node 1 Rx
                (node1_num - 1) * 6 + 4,  # Node 1 Ry
                (node1_num - 1) * 6 + 5,  # Node 1 Rz
                (node2_num - 1) * 6 + 0,  # Node 2 Ux
                (node2_num - 1) * 6 + 1,  # Node 2 Uy
                (node2_num - 1) * 6 + 2,  # Node 2 Uz
                (node2_num - 1) * 6 + 3,  # Node 2 Rx
                (node2_num - 1) * 6 + 4,  # Node 2 Ry
                (node2_num - 1) * 6 + 5   # Node 2 Rz
            ]
            
            # Assemble into global matrix
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    self.K_global[gi, gj] += k_global[i, j]
        
        print(f"✓ Global stiffness matrix assembled ({n_dof} × {n_dof})")
        print(f"  Non-zero elements: {np.count_nonzero(self.K_global)}")
        print(f"  Matrix condition number: {np.linalg.cond(self.K_global):.2e}")
        
    def assemble_force_vector(self):
        """Assemble global force vector (forces and moments)"""
        print("\\nASSEMBLING FORCE VECTOR")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = self.dof_per_node * num_nodes
        
        self.F_global = np.zeros(n_dof)
        
        # Apply loads and moments
        for node_id, load in self.loads.items():
            idx_base = (node_id - 1) * 6
            self.F_global[idx_base + 0] = load['Fx']
            self.F_global[idx_base + 1] = load['Fy']
            self.F_global[idx_base + 2] = load['Fz']
            self.F_global[idx_base + 3] = load['Mx']
            self.F_global[idx_base + 4] = load['My']
            self.F_global[idx_base + 5] = load['Mz']
        
        total_force = np.sqrt(np.sum(self.F_global[0::6]**2 + 
                                      self.F_global[1::6]**2 + 
                                      self.F_global[2::6]**2))
        print(f"✓ Force vector assembled ({n_dof} DOFs)")
        print(f"  Total applied force magnitude: {total_force:.2f} N")
        print(f"  Non-zero force/moment DOFs: {np.count_nonzero(self.F_global)}")
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions and solve system"""
        print("\\nAPPLYING BOUNDARY CONDITIONS AND SOLVING")
        print("="*80)
        
        num_nodes = len(self.nodes)
        n_dof = self.dof_per_node * num_nodes
        
        # Identify known and unknown DOFs
        known_dofs = []
        unknown_dofs = []
        
        for node_id, bc in self.boundary_conditions.items():
            idx_base = (node_id - 1) * 6
            
            # Check each DOF
            dof_names = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
            for i, dof_name in enumerate(dof_names):
                if bc[dof_name] == 1:
                    known_dofs.append(idx_base + i)
                else:
                    unknown_dofs.append(idx_base + i)
        
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
        
        # Get max displacement and rotation
        displacements = np.sqrt(self.u_global[0::6]**2 + 
                               self.u_global[1::6]**2 + 
                               self.u_global[2::6]**2)
        rotations = np.sqrt(self.u_global[3::6]**2 + 
                           self.u_global[4::6]**2 + 
                           self.u_global[5::6]**2)
        max_disp = np.max(displacements)
        max_rot = np.max(rotations)
        
        print(f"\\n✓ Solution obtained")
        print(f"  Max displacement: {max_disp:.6e} m ({max_disp*1000:.6f} mm)")
        print(f"  Max rotation: {max_rot:.6e} rad ({np.degrees(max_rot):.6f} deg)")
        
    def calculate_reactions(self):
        """Calculate reaction forces and moments at supports"""
        print("\\nCALCULATING REACTION FORCES AND MOMENTS")
        print("="*80)
        
        # Reactions = K * u - F
        self.reactions = self.K_global @ self.u_global - self.F_global
        
        print(f"✓ Reaction forces and moments calculated")
        
        # Display reactions at support nodes
        for node_id, bc in self.boundary_conditions.items():
            idx_base = (node_id - 1) * 6
            
            rx = self.reactions[idx_base + 0]
            ry = self.reactions[idx_base + 1]
            rz = self.reactions[idx_base + 2]
            mx = self.reactions[idx_base + 3]
            my = self.reactions[idx_base + 4]
            mz = self.reactions[idx_base + 5]
            
            print(f"  Node {node_id}:")
            print(f"    Forces:  Rx={rx:10.2f} N, Ry={ry:10.2f} N, Rz={rz:10.2f} N")
            print(f"    Moments: Mx={mx:10.2f} N·m, My={my:10.2f} N·m, Mz={mz:10.2f} N·m")
        
        # Check equilibrium
        total_fx = np.sum(self.F_global[0::6]) + np.sum(self.reactions[0::6])
        total_fy = np.sum(self.F_global[1::6]) + np.sum(self.reactions[1::6])
        total_fz = np.sum(self.F_global[2::6]) + np.sum(self.reactions[2::6])
        
        print(f"\\n  Equilibrium check:")
        print(f"    ΣFx: {total_fx:12.2e} N")
        print(f"    ΣFy: {total_fy:12.2e} N")
        print(f"    ΣFz: {total_fz:12.2e} N")
        
    def calculate_element_forces(self):
        """Calculate internal forces and moments in beam elements"""
        print("\\nCALCULATING ELEMENT INTERNAL FORCES AND MOMENTS")
        print("="*80)
        
        element_results = []
        
        for idx in range(len(self.elements)):
            elem = self.elements.iloc[idx]
            
            node1_num = int(elem['Node1'])
            node2_num = int(elem['Node2'])
            
            # Get node coordinates
            node1 = self.nodes[self.nodes['Node Number'] == node1_num].iloc[0]
            node2 = self.nodes[self.nodes['Node Number'] == node2_num].iloc[0]
            
            x1, y1, z1 = node1['X'], node1['Y'], node1['Z']
            x2, y2, z2 = node2['X'], node2['Y'], node2['Z']
            
            # Compute transformation matrix and length
            T, L = self._compute_transformation_matrix(x1, y1, z1, x2, y2, z2)
            
            # Get global displacements for this element
            idx_base_1 = (node1_num - 1) * 6
            idx_base_2 = (node2_num - 1) * 6
            
            u_global_elem = np.concatenate([
                self.u_global[idx_base_1:idx_base_1+6],
                self.u_global[idx_base_2:idx_base_2+6]
            ])
            
            # Transform to local coordinates
            T_12x12 = self._expand_transformation_matrix(T)
            u_local = T_12x12 @ u_global_elem
            
            # Calculate local forces: f_local = k_local * u_local
            k_local = self._create_local_stiffness_matrix(L)
            f_local = k_local @ u_local
            
            # Extract forces and moments at node 1 (in local coordinates)
            axial_force = f_local[0]  # Fx at node 1
            shear_y = f_local[1]       # Fy at node 1
            shear_z = f_local[2]       # Fz at node 1
            torsion = f_local[3]       # Mx at node 1
            moment_y = f_local[4]      # My at node 1
            moment_z = f_local[5]      # Mz at node 1
            
            # Calculate stress and strain
            stress_axial = axial_force / self.A
            strain_axial = stress_axial / self.E
            
            element_results.append({
                'Element Number': int(elem['Element Number']),
                'Axial Force (N)': axial_force,
                'Shear Y (N)': shear_y,
                'Shear Z (N)': shear_z,
                'Torsion (N·m)': torsion,
                'Moment Y (N·m)': moment_y,
                'Moment Z (N·m)': moment_z,
                'Stress (Pa)': stress_axial,
                'Strain': strain_axial
            })
        
        self.element_results = pd.DataFrame(element_results)
        
        max_stress = np.abs(self.element_results['Stress (Pa)']).max()
        max_axial = np.abs(self.element_results['Axial Force (N)']).max()
        max_moment = max(np.abs(self.element_results['Moment Y (N·m)']).max(),
                        np.abs(self.element_results['Moment Z (N·m)']).max())
        
        print(f"✓ Element forces calculated for {len(self.elements)} elements")
        print(f"  Max axial stress: {max_stress:.2e} Pa ({max_stress/1e6:.2f} MPa)")
        print(f"  Max axial force: {max_axial:.2f} N")
        print(f"  Max bending moment: {max_moment:.2f} N·m")
        
    def export_results(self, filename='data/structure.res'):
        """Export results to text file"""
        print("\\nEXPORTING RESULTS")
        print("="*80)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            # Header
            f.write("# FEM Solution Results File - BEAM ELEMENTS\\n")
            f.write(f"# Generated by FEM Beam Solver\\n")
            f.write(f"# Date: {pd.Timestamp.now()}\\n\\n")
            
            # Displacements and Rotations
            f.write("DISPLACEMENTS\\n")
            f.write("# Node_ID Ux(m) Uy(m) Uz(m) Rx(rad) Ry(rad) Rz(rad) |U|(m) |R|(rad)\\n")
            for node_num in range(1, len(self.nodes) + 1):
                idx_base = (node_num - 1) * 6
                ux = self.u_global[idx_base + 0]
                uy = self.u_global[idx_base + 1]
                uz = self.u_global[idx_base + 2]
                rx = self.u_global[idx_base + 3]
                ry = self.u_global[idx_base + 4]
                rz = self.u_global[idx_base + 5]
                mag_u = np.sqrt(ux**2 + uy**2 + uz**2)
                mag_r = np.sqrt(rx**2 + ry**2 + rz**2)
                f.write(f"{node_num} {ux:.6e} {uy:.6e} {uz:.6e} {rx:.6e} {ry:.6e} {rz:.6e} {mag_u:.6e} {mag_r:.6e}\\n")
            f.write("\\n")
            
            # Reactions
            f.write("REACTIONS\\n")
            f.write("# Node_ID Rx(N) Ry(N) Rz(N) Mx(N·m) My(N·m) Mz(N·m)\\n")
            for node_id in self.boundary_conditions.keys():
                idx_base = (node_id - 1) * 6
                rx = self.reactions[idx_base + 0]
                ry = self.reactions[idx_base + 1]
                rz = self.reactions[idx_base + 2]
                mx = self.reactions[idx_base + 3]
                my = self.reactions[idx_base + 4]
                mz = self.reactions[idx_base + 5]
                f.write(f"{node_id} {rx:.6f} {ry:.6f} {rz:.6f} {mx:.6f} {my:.6f} {mz:.6f}\\n")
            f.write("\\n")
            
            # Element forces
            f.write("ELEMENT_FORCES\\n")
            f.write("# Elem_ID Axial(N) Shear_Y(N) Shear_Z(N) Torsion(N·m) Moment_Y(N·m) Moment_Z(N·m) Stress(Pa) Strain\\n")
            for idx in range(len(self.element_results)):
                result = self.element_results.iloc[idx]
                f.write(f"{int(result['Element Number'])} "
                       f"{result['Axial Force (N)']:.6f} "
                       f"{result['Shear Y (N)']:.6f} "
                       f"{result['Shear Z (N)']:.6f} "
                       f"{result['Torsion (N·m)']:.6f} "
                       f"{result['Moment Y (N·m)']:.6f} "
                       f"{result['Moment Z (N·m)']:.6f} "
                       f"{result['Stress (Pa)']:.6e} "
                       f"{result['Strain']:.6e}\\n")
            f.write("\\n")
            
            # Summary
            f.write("SUMMARY\\n")
            displacements = np.sqrt(self.u_global[0::6]**2 + 
                                   self.u_global[1::6]**2 + 
                                   self.u_global[2::6]**2)
            rotations = np.sqrt(self.u_global[3::6]**2 + 
                               self.u_global[4::6]**2 + 
                               self.u_global[5::6]**2)
            max_disp = np.max(displacements)
            max_rot = np.max(rotations)
            max_stress = np.abs(self.element_results['Stress (Pa)']).max()
            
            f.write(f"Max_Displacement {max_disp:.6e}\\n")
            f.write(f"Max_Rotation {max_rot:.6e}\\n")
            f.write(f"Max_Stress {max_stress:.6e}\\n")
            f.write("\\n")
            
            f.write("END\\n")
        
        print(f"✓ Results exported to: {filename}")
        print(f"  - {len(self.nodes)} node displacements/rotations")
        print(f"  - {len(self.boundary_conditions)} reaction forces/moments")
        print(f"  - {len(self.element_results)} element forces/moments/stresses")


def main():
    """Main execution function"""
    print("\\n" + "="*80)
    print("FEM BEAM SOLVER")
    print("="*80 + "\\n")
    
    # Create solver
    solver = FEMBeamSolver()
    
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
    
    print("\\n" + "="*80)
    print("BEAM SOLVER COMPLETED SUCCESSFULLY")
    print("="*80 + "\\n")
    print("Next step: Run postprocessor.py to visualize results")
    print("  Command: python postprocessor.py")


if __name__ == "__main__":
    main()
