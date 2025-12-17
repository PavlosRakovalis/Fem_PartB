"""
FEM Pre-Processor Module
========================
This module handles geometry creation, boundary conditions definition,
and exports the structure data to a text file for the solver.

Author: Rakovalis Pavlos 6931
Date: November 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from geometry_utils import rotate_points_around_y
import os

# Configuration
BROWSER = 'chrome'
if BROWSER:
    pio.renderers.default = 'browser'

class FEMPreProcessor:
    """Handles geometry creation and boundary conditions for FEM analysis"""
    
    def __init__(self):
        self.points = pd.DataFrame(columns=['Node Number', 'X', 'Y', 'Z'])
        self.elements = pd.DataFrame(columns=['Element Number', 'Node1', 'Node2', 'E', 'V'])
        self.boundary_conditions = {}  # node_id: {'Ux': 0/1, 'Uy': 0/1, 'Uz': 0/1}
        self.loads = {}  # node_id: {'Fx': value, 'Fy': value, 'Fz': value}
        self.materials = {'STEEL': {'E': 210e9, 'nu': 0.3}}
        
        
    ################################################################################
    #                                                                              #
    #                        GEOMETRY DEFINITION SECTION                           #
    #                                                                              #
    #  This section defines the geometry, materials, and parameters for the        #
    #  specific structure being analyzed.                                          #
    #                                                                              #
    #  HOW TO DEFINE YOUR STRUCTURE:                                               #
    #  -----------------------------                                               #
    #  1. PARAMETERS: Define geometric parameters (L, A, phi, A_0, etc.)          #
    #     These are specific to this problem and won't be saved to .dat file      #
    #                                                                              #
    #  2. NODES: Use self._add_node(x, y, z) to add nodes                         #
    #     - Nodes are automatically numbered sequentially                          #
    #     - Coordinates are in meters                                              #
    #                                                                              #
    #  3. ELEMENTS: Elements are created automatically based on connectivity       #
    #     - Axis-aligned elements (parallel to X, Y, or Z)                         #
    #     - Diagonal bracing (coplanar diagonals)                                  #
    #     - Manual connections can be added later                                  #
    #                                                                              #
    #  4. BOUNDARY CONDITIONS: Use self.add_boundary_condition(node_id, ux, uy, uz)#
    #     - node_id: Node number (1-based)                                         #
    #     - ux, uy, uz: 1=fixed, 0=free                                            #
    #                                                                              #
    #  5. LOADS: Use self.add_load(node_id, fx, fy, fz)                           #
    #     - node_id: Node number (1-based)                                         #
    #     - fx, fy, fz: Force components in Newtons                                #
    #                                                                              #
    ################################################################################
    
    def _define_geometry_parameters(self):
        """Define geometry-specific parameters (not saved to .dat file)"""
        # SIMPLE GEOMETRY (COMMENTED OUT)
        # self.cross_section = 4e-4  # 4 cm² = 4×10⁻⁴ m²
        
        # COMPLEX GEOMETRY: Original crane/truss structure
        # Geometry parameters for this specific crane/truss structure
        self.L = 1.5 * 1.13      # Element length (m)
        self.A = 1.2 * 1.69      # Base dimension (m)
        self.B = 1.93            # Additional parameter B (m)
        self.phi = 60 + 9.1      # Rotation angle (degrees)
        self.A_0 = 6 * (0.5 + 0.13)  # Base cross-section parameter (dimensionless)
        
    def create_geometry(self):
        """Create the truss geometry"""
        print("="*80)
        print("CREATING GEOMETRY")
        print("="*80)
        
        # Define geometry parameters first
        self._define_geometry_parameters()
        
        # Add initial nodes (WITHOUT rotation)
        self._add_initial_nodes()
        
        print(f"✓ Created {len(self.points)} nodes")
        
    def _add_initial_nodes(self):
        """Add all nodes to the structure"""
        # SIMPLE GEOMETRY (COMMENTED OUT)
        # # Node 1: Origin (0, 0, 0) - Fixed
        # self._add_node(0.0, 0.0, 0.0)
        # 
        # # Node 2: Along X-axis (1, 0, 0) - Fixed
        # self._add_node(1.0, 0.0, 0.0)
        # 
        # # Node 3: Along Y-axis (0, 3, 0) - Loaded with 1000 N in +X direction
        # self._add_node(0.0, 3.0, 0.0)
        # 
        # # Node 4: Along Z-axis (0, 0, 1) - Free
        # self._add_node(0.0, 0.0, 1.0)
        
        # COMPLEX GEOMETRY: Original crane structure
        L, A = self.L, self.A
        
        # Node 1: x = A, y = -L/2, z = 0
        self._add_node(A, -L/2, 0.0)
        
        # Node 2: x = A, y = +L/2, z = 0
        self._add_node(A, L/2, 0.0)
        
        # Nodes 3-9: z = -L/2, y = -L/2
        for i in range(7):
            self._add_node(A + L + i * L, -L/2, -L/2)
        
        # Nodes 10-16: z = -L/2, y = L/2
        for i in range(7):
            self._add_node(A + L + i * L, L/2, -L/2)
        
        # Nodes 17-22: z = +L/2, y = -L/2
        for i in range(6):
            self._add_node(A + L + i * L, -L/2, L/2)
        
        # Nodes 23-28: z = L/2, y = L/2
        for i in range(6):
            self._add_node(A + L + i * L, L/2, L/2)
        
        # Node 29: Load application point
        self._add_node(A + L * 6.5, 0, -1.5 * L)
        
    def _add_node(self, x, y, z):
        """Add a single node"""
        new_node = pd.DataFrame({
            'Node Number': [len(self.points) + 1],
            'X': [x],
            'Y': [y],
            'Z': [z]
        })
        self.points = pd.concat([self.points, new_node], ignore_index=True)
        
    def create_elements(self):
        """Create all truss elements"""
        print("\nCREATING ELEMENTS")
        print("="*80)
        
        # SIMPLE GEOMETRY (COMMENTED OUT)
        # # Base triangle (on XY plane, z=0)
        # # Element 1: Node 1 → Node 2 (base edge, along X-axis)
        # self._add_element(1, 1, 2)
        # 
        # # Element 2: Node 2 → Node 3 (base edge)
        # self._add_element(2, 2, 3)
        # 
        # # Element 3: Node 3 → Node 1 (base edge)
        # self._add_element(3, 3, 1)
        # 
        # # Pyramid edges (connecting apex node 4 to base)
        # # Element 4: Node 4 → Node 1 (vertical edge from origin)
        # self._add_element(4, 4, 1)
        # 
        # # Element 5: Node 4 → Node 2 (pyramid edge)
        # self._add_element(5, 4, 2)
        # 
        # # Element 6: Node 4 → Node 3 (pyramid edge)
        # self._add_element(6, 4, 3)
        # 
        # # Calculate element lengths and assign cross sections
        # self._calculate_element_properties()
        
        # COMPLEX GEOMETRY: Original crane structure
        element_counter = 1
        tolerance = 1e-6
        max_element_length = 1.55 * self.L
        
        # Create axis-aligned elements
        element_counter = self._create_axis_aligned_elements(element_counter, tolerance, max_element_length)
        print(f"  After axis-aligned: {len(self.elements)} elements")
        
        # Add diagonal bracing
        element_counter = self._add_diagonal_bracing(element_counter, tolerance, max_element_length)
        print(f"  After diagonal bracing: {len(self.elements)} elements")
        
        # Add connections from Node 29
        element_counter = self._add_node29_connections(element_counter)
        print(f"  After Node 29 connections: {len(self.elements)} elements")
        
        # Add specific connections
        element_counter = self._add_specific_connections(element_counter)
        print(f"  After specific connections: {len(self.elements)} elements")
        
        # Calculate element lengths and assign cross sections
        self._calculate_element_properties()
        
        print(f"✓ Created {len(self.elements)} elements")
    
    def rotate_structure(self):
        """Rotate the structure around Y-axis"""
        print("\nROTATING STRUCTURE")
        print("="*80)
        self.points = rotate_points_around_y(self.points, -self.phi, origin=(self.A, 0.0, 0.0))
        print(f"✓ Structure rotated by {-self.phi:.1f} degrees around Y-axis")
    
    def add_post_rotation_elements(self):
        """Add additional nodes and elements after rotation"""
        print("\nADDING POST-ROTATION ELEMENTS")
        print("="*80)
        
        # Add 3 new nodes with exact coordinates as specified:
        # x = -A, z = +B, y = 0 / -L/2 / +L/2
        A = self.A
        B = self.B
        L = self.L
        
        # Node 30: x = -A, y = 0, z = +B
        self._add_node(-A, 0, B)
        
        # Node 31: x = -A, y = -L/2, z = +B
        self._add_node(-A, -L/2, B)
        
        # Node 32: x = -A, y = +L/2, z = +B
        self._add_node(-A, L/2, B)
        
        print(f"  Added 3 new nodes (30, 31, 32)")
        
        # Add 4 new elements with specific cross-section
        element_counter = len(self.elements) + 1
        element_counter = self._add_additional_elements(element_counter)
        
        # Recalculate element properties for new elements only
        self._calculate_new_element_properties()
        
        print(f"✓ Added 4 new elements (total: {len(self.elements)} elements)")
    
    def _calculate_new_element_properties(self):
        """Calculate properties only for the last 4 elements"""
        # Get the last 4 elements
        start_idx = len(self.elements) - 4
        
        for i in range(start_idx, len(self.elements)):
            node1_idx = int(self.elements.iloc[i]['Node1']) - 1
            node2_idx = int(self.elements.iloc[i]['Node2']) - 1
            
            x1, y1, z1 = self.points.iloc[node1_idx][['X', 'Y', 'Z']]
            x2, y2, z2 = self.points.iloc[node2_idx][['X', 'Y', 'Z']]
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Set element properties
            self.elements.at[i, 'Element Length'] = length
            self.elements.at[i, 'Element Cross Section'] = 0.5 * self.A_0 * 1e-4
        
    def _create_axis_aligned_elements(self, element_counter, tolerance, max_length):
        """Create elements parallel to X, Y, or Z axes"""
        axis_aligned_count = 0
        for i in range(len(self.points)):
            node1 = self.points.iloc[i]
            node1_num = int(node1['Node Number'])
            
            for j in range(i + 1, len(self.points)):
                node2 = self.points.iloc[j]
                node2_num = int(node2['Node Number'])
                
                dx = abs(node2['X'] - node1['X'])
                dy = abs(node2['Y'] - node1['Y'])
                dz = abs(node2['Z'] - node1['Z'])
                
                element_length = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if element_length > max_length:
                    continue
                
                # Check axis alignment
                is_x_aligned = dx > tolerance and dy < tolerance and dz < tolerance
                is_y_aligned = dx < tolerance and dy > tolerance and dz < tolerance
                is_z_aligned = dx < tolerance and dy < tolerance and dz > tolerance
                
                if is_x_aligned or is_y_aligned or is_z_aligned:
                    self._add_element(element_counter, node1_num, node2_num)
                    element_counter += 1
                    
        return element_counter
    
    def _add_diagonal_bracing(self, element_counter, tolerance, max_length):
        """Add diagonal bracing elements (DISABLED - no diagonal bracing added)"""
        # Diagonal bracing has been disabled - return without adding any elements
        return element_counter
    
    def _add_node29_connections(self, element_counter):
        """Add connections from Node 29"""
        connecting_nodes = [8, 9, 15, 16]
        for target in connecting_nodes:
            self._add_element(element_counter, 29, target)
            element_counter += 1
        return element_counter
    
    def _add_additional_elements(self, element_counter):
        """Add 4 additional elements with specific cross-section"""
        # Elements connecting to new nodes with cross-section = 0.5 * A_0 * 1e-4
        additional_connections = [
            (22, 30),  # From node 22 to node 30 (x=-A, z=+B, y=0)
            (28, 30),  # From node 28 to node 30 (x=-A, z=+B, y=0)
            (19, 31),  # From node 19 to node 31 (x=-A, z=+B, y=-L/2)
            (25, 32)   # From node 25 to node 32 (x=-A, z=+B, y=+L/2)
        ]
        
        for n1, n2 in additional_connections:
            self._add_element(element_counter, n1, n2)
            element_counter += 1
        
        return element_counter
    
    def _add_specific_connections(self, element_counter):
        """Add specific connections"""
        # Add user-defined specific connections
        specific_connections = [
            (16, 28),  # Connect node 16 to node 28
            (10, 2),   # Connect node 10 to node 2
            (23, 2),   # Connect node 23 to node 2
            (17, 1),   # Connect node 17 to node 1
            (3, 1),    # Connect node 3 to node 1
            (22, 9)    # Connect node 22 to node 9
        ]
        
        for n1, n2 in specific_connections:
            self._add_element(element_counter, n1, n2)
            element_counter += 1
        
        return element_counter
    
    def _add_element(self, elem_num, node1, node2):
        """Add a single element"""
        new_elem = pd.DataFrame({
            'Element Number': [elem_num],
            'Node1': [node1],
            'Node2': [node2],
            'E': [210e9],
            'V': [0.3]
        })
        self.elements = pd.concat([self.elements, new_elem], ignore_index=True)
    
    def _calculate_element_properties(self):
        """Calculate element lengths and assign cross sections"""
        element_lengths = []
        cross_sections = []
        
        for i in range(len(self.elements)):
            node1_idx = int(self.elements.iloc[i]['Node1']) - 1
            node2_idx = int(self.elements.iloc[i]['Node2']) - 1
            
            x1, y1, z1 = self.points.iloc[node1_idx][['X', 'Y', 'Z']]
            x2, y2, z2 = self.points.iloc[node2_idx][['X', 'Y', 'Z']]
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            element_lengths.append(length)
            
            # SIMPLE GEOMETRY (COMMENTED OUT)
            # # All elements have same cross-section (4 cm²)
            # cross_sections.append(self.cross_section)
            
            # COMPLEX GEOMETRY: Assign cross section based on length or specific element
            # Check if this is one of the 4 additional elements (last 4 elements)
            elem_num = int(self.elements.iloc[i]['Element Number'])
            node1 = int(self.elements.iloc[i]['Node1'])
            node2 = int(self.elements.iloc[i]['Node2'])
            
            # Additional elements connecting to nodes 30, 31, 32
            is_additional = (node1 in [22, 28, 19, 25] and node2 in [30, 31, 32]) or \
                           (node2 in [22, 28, 19, 25] and node1 in [30, 31, 32])
            
            if is_additional:
                cross_sections.append(0.5 * self.A_0 * 1e-4)  # Specified cross-section
            elif length < 1.9:
                cross_sections.append(1.5 * self.A_0 * 1e-4)  # Straight elements (m²)
            elif length > 2:
                cross_sections.append(0.5 * self.A_0 * 1e-4)  # Diagonal elements (m²)
            else:
                cross_sections.append(1.0 * self.A_0 * 1e-4)  # Default
        
        self.elements['Element Length'] = element_lengths
        self.elements['Element Cross Section'] = cross_sections
        
    
    ################################################################################
    #                                                                              #
    #                     END OF GEOMETRY DEFINITION SECTION                       #
    #                                                                              #
    ################################################################################
    
    
    ################################################################################
    #                                                                              #
    #                   BOUNDARY CONDITIONS AND LOADS SECTION                      #
    #                                                                              #
    #  This section defines the supports (boundary conditions) and external        #
    #  loads applied to the structure.                                             #
    #                                                                              #
    #  HOW TO DEFINE BOUNDARY CONDITIONS:                                          #
    #  ----------------------------------                                          #
    #  Use: self.add_boundary_condition(node_id, ux, uy, uz)                      #
    #    - node_id: Node number (1-based indexing)                                 #
    #    - ux, uy, uz: Constraints for each DOF (0=free, 1=fixed)                 #
    #                                                                              #
    #  Examples:                                                                   #
    #    self.add_boundary_condition(1, ux=1, uy=1, uz=1)  # Fully fixed node     #
    #    self.add_boundary_condition(2, ux=1, uy=0, uz=0)  # Only X constrained   #
    #                                                                              #
    #  HOW TO DEFINE LOADS:                                                        #
    #  --------------------                                                        #
    #  Use: self.add_load(node_id, fx, fy, fz)                                    #
    #    - node_id: Node number (1-based indexing)                                 #
    #    - fx, fy, fz: Force components in Newtons                                 #
    #                                                                              #
    #  Examples:                                                                   #
    #    self.add_load(3, fx=1000, fy=0, fz=0)      # 1000 N in +X direction      #
    #    self.add_load(5, fx=0, fy=-500, fz=0)      # 500 N in -Y direction       #
    #    self.add_load(7, fx=100, fy=200, fz=-300)  # Combined load               #
    #                                                                              #
    ################################################################################
    
    def add_boundary_condition(self, node_id, ux=0, uy=0, uz=0):
        """Add boundary condition (0=free, 1=fixed)"""
        self.boundary_conditions[node_id] = {'Ux': ux, 'Uy': uy, 'Uz': uz}
        
    def add_load(self, node_id, fx=0.0, fy=0.0, fz=0.0):
        """Add load at node (in Newtons)"""
        self.loads[node_id] = {'Fx': fx, 'Fy': fy, 'Fz': fz}
        
    def set_default_bcs_and_loads(self):
        """Set default boundary conditions and loads for this structure"""
        print("\nSETTING BOUNDARY CONDITIONS AND LOADS")
        print("="*80)
        
        # SIMPLE GEOMETRY (COMMENTED OUT)
        # # Fixed supports at nodes 1, 2, and 4 (all DOFs constrained)
        # self.add_boundary_condition(1, ux=1, uy=1, uz=1)
        # self.add_boundary_condition(2, ux=1, uy=1, uz=1)
        # self.add_boundary_condition(4, ux=1, uy=1, uz=1)
        # 
        # print(f"  Fixed supports at nodes: [1, 2, 4]")
        # 
        # # Original load: Only Fx=1000N at Node 3
        # self.add_load(3, fx=1000.0, fy=0.0, fz=0.0)
        # print(f"  Applied load: Fx = 1000 N at Node 3")
        
        # COMPLEX GEOMETRY: Original crane structure
        # Fixed supports at new nodes (30, 31, 32)
        fixed_nodes = [1, 2, 30, 31, 32]
        for node in fixed_nodes:
            self.add_boundary_condition(node, ux=1, uy=1, uz=1)
        
        print(f"  Fixed supports at nodes: {fixed_nodes}")
        
        # Applied load
        self.add_load(29, fx=0.0, fy=0.0, fz=-2000.0)
        print(f"  Applied load: Fz = -2000 N at Node 29")
        
        print(f"✓ Boundary conditions and loads defined")
        
    def visualize_geometry(self, show_plot=True):
        """Visualize the structure geometry with Plotly"""
        print("\nVISUALIZING GEOMETRY")
        print("="*80)
        
        fig = go.Figure()
        
        # Add elements
        edge_x, edge_y, edge_z = [], [], []
        for idx in range(len(self.elements)):
            node1_idx = int(self.elements.iloc[idx]['Node1']) - 1
            node2_idx = int(self.elements.iloc[idx]['Node2']) - 1
            
            x1, y1, z1 = self.points.iloc[node1_idx][['X', 'Y', 'Z']]
            x2, y2, z2 = self.points.iloc[node2_idx][['X', 'Y', 'Z']]
            
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])
            edge_z.extend([z1, z2, None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=3),
            name='Elements'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=self.points['X'],
            y=self.points['Y'],
            z=self.points['Z'],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[f"{int(n)}" for n in self.points['Node Number']],
            textposition='top center',
            name='Nodes'
        ))
        
        # Highlight fixed supports
        fixed_nodes = list(self.boundary_conditions.keys())
        if fixed_nodes:
            fixed_coords = self.points[self.points['Node Number'].isin(fixed_nodes)]
            fig.add_trace(go.Scatter3d(
                x=fixed_coords['X'],
                y=fixed_coords['Y'],
                z=fixed_coords['Z'],
                mode='markers',
                marker=dict(size=10, color='blue', symbol='diamond'),
                name='Fixed Supports'
            ))
        
        # Highlight loaded nodes
        loaded_nodes = list(self.loads.keys())
        if loaded_nodes:
            loaded_coords = self.points[self.points['Node Number'].isin(loaded_nodes)]
            fig.add_trace(go.Scatter3d(
                x=loaded_coords['X'],
                y=loaded_coords['Y'],
                z=loaded_coords['Z'],
                mode='markers',
                marker=dict(size=10, color='green', symbol='square'),
                name='Loaded Nodes'
            ))
        
        # Add force arrows
        force_scale = 0.0008  # Scale factor for arrow length
        arrow_color = 'magenta'
        arrow_width = 8
        cone_size_ratio = 0.25  # Size of arrowhead relative to total length
        
        for idx, (node_num, load) in enumerate(self.loads.items()):
            fx, fy, fz = load['Fx'], load['Fy'], load['Fz']
            magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
            
            if magnitude < 1e-6:
                continue
            
            # Get node position
            node_pos = self.points[self.points['Node Number'] == node_num].iloc[0]
            x0, y0, z0 = node_pos['X'], node_pos['Y'], node_pos['Z']
            
            # Calculate arrow direction and length
            arrow_len = magnitude * force_scale
            dx_total = fx / magnitude * arrow_len
            dy_total = fy / magnitude * arrow_len
            dz_total = fz / magnitude * arrow_len
            
            # Calculate shaft and cone portions
            shaft_ratio = 1 - cone_size_ratio
            dx_shaft = dx_total * shaft_ratio
            dy_shaft = dy_total * shaft_ratio
            dz_shaft = dz_total * shaft_ratio
            
            dx_cone = dx_total * cone_size_ratio
            dy_cone = dy_total * cone_size_ratio
            dz_cone = dz_total * cone_size_ratio
            
            # Add force line (shaft of arrow)
            show_in_legend = (idx == 0)
            legend_label = f"External Force: {magnitude:.0f} N" if show_in_legend else None
            
            fig.add_trace(go.Scatter3d(
                x=[x0, x0 + dx_shaft],
                y=[y0, y0 + dy_shaft],
                z=[z0, z0 + dz_shaft],
                mode='lines',
                line=dict(color=arrow_color, width=arrow_width),
                showlegend=show_in_legend,
                name=legend_label,
                hovertemplate=f'<b>External Force on Node {node_num}</b><br>Magnitude: {magnitude:.0f} N<br>Fx: {fx:.0f} N<br>Fy: {fy:.0f} N<br>Fz: {fz:.0f} N<extra></extra>'
            ))
            
            # Add arrowhead using cone
            fig.add_trace(go.Cone(
                x=[x0 + dx_shaft],
                y=[y0 + dy_shaft],
                z=[z0 + dz_shaft],
                u=[dx_cone],
                v=[dy_cone],
                w=[dz_cone],
                colorscale=[[0, arrow_color], [1, arrow_color]],
                showscale=False,
                sizemode='absolute',
                sizeref=arrow_len * 0.4,
                anchor='tail',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title='FEM Structure Geometry',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1200,
            height=900
        )
        
        if show_plot:
            fig.show()
        
        # Save HTML to plots folder
        os.makedirs('plots', exist_ok=True)
        html_path = 'plots/geometry_visualization.html'
        fig.write_html(html_path)
        print(f"✓ Geometry visualization saved to: {html_path}")
        
        return fig
    
    def export_to_file(self, filename='data/structure.dat'):
        """Export structure data to text file"""
        print("\nEXPORTING TO FILE")
        print("="*80)
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            # Header
            f.write("# FEM Structure Data File\n")
            f.write(f"# Generated by FEM Pre-Processor\n")
            f.write(f"# Date: {pd.Timestamp.now()}\n\n")
            
            # NOTE: Geometry parameters (L, A, phi, A_0) are NOT exported
            # They are defined in _define_geometry_parameters() and specific to this problem
            
            # Materials
            f.write("MATERIALS\n")
            for mat_name, props in self.materials.items():
                f.write(f"{mat_name} {props['E']:.6e} {props['nu']:.6f}\n")
            f.write("\n")
            
            # Nodes
            f.write("NODES\n")
            f.write("# Node_ID X Y Z\n")
            for idx in range(len(self.points)):
                node = self.points.iloc[idx]
                f.write(f"{int(node['Node Number'])} {node['X']:.6f} {node['Y']:.6f} {node['Z']:.6f}\n")
            f.write("\n")
            
            # Elements
            f.write("ELEMENTS\n")
            f.write("# Elem_ID Node1 Node2 CrossSection E nu\n")
            for idx in range(len(self.elements)):
                elem = self.elements.iloc[idx]
                f.write(f"{int(elem['Element Number'])} {int(elem['Node1'])} {int(elem['Node2'])} "
                       f"{elem['Element Cross Section']:.6e} {elem['E']:.6e} {elem['V']:.6f}\n")
            f.write("\n")
            
            # Boundary Conditions
            f.write("BOUNDARY_CONDITIONS\n")
            f.write("# Node_ID Ux Uy Uz (0=free, 1=fixed)\n")
            for node_id, bc in self.boundary_conditions.items():
                f.write(f"{node_id} {bc['Ux']} {bc['Uy']} {bc['Uz']}\n")
            f.write("\n")
            
            # Loads
            f.write("LOADS\n")
            f.write("# Node_ID Fx Fy Fz (Newtons)\n")
            for node_id, load in self.loads.items():
                f.write(f"{node_id} {load['Fx']:.6f} {load['Fy']:.6f} {load['Fz']:.6f}\n")
            f.write("\n")
            
            f.write("END\n")
        
        print(f"✓ Structure data exported to: {filename}")
        print(f"  - {len(self.points)} nodes")
        print(f"  - {len(self.elements)} elements")
        print(f"  - {len(self.boundary_conditions)} boundary conditions")
        print(f"  - {len(self.loads)} loads")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FEM PRE-PROCESSOR")
    print("="*80 + "\n")
    
    # Create preprocessor
    preprocessor = FEMPreProcessor()
    
    # Create geometry
    preprocessor.create_geometry()
    
    # Create elements
    preprocessor.create_elements()
    
    # Rotate structure (for complex geometry)
    preprocessor.rotate_structure()
    
    # Add additional nodes and elements after rotation
    preprocessor.add_post_rotation_elements()
    
    # Set boundary conditions and loads
    preprocessor.set_default_bcs_and_loads()
    
    # Visualize
    preprocessor.visualize_geometry(show_plot=True)
    
    # Export to file
    preprocessor.export_to_file('data/structure.dat')
    
    print("\n" + "="*80)
    print("PRE-PROCESSOR COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    print("Next step: Run solver.py to solve the FEM system")
    print("  Command: python solver.py")


if __name__ == "__main__":
    main()
