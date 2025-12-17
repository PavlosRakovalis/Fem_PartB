import numpy as np

def rotation_matrix_from_element(element_num, elements_df, nodes_df):
    """
    Υπολογίζει το μητρώο μετασχηματισμού Τ για ένα στοιχείο με βάση τον αριθμό του.
    
    Είσοδος:
        element_num: Ο αριθμός του στοιχείου
        elements_df: DataFrame με στήλες ['Element Number', 'Node1', 'Node2', 'E', 'V']
        nodes_df: DataFrame με στήλες ['Node', 'X', 'Y', 'Z']
    
    Έξοδος:
        T: Πίνακας μετασχηματισμού
        L: Μήκος στοιχείου
    """
    # Βρες τη γραμμή του στοιχείου
    element_row = elements_df[elements_df['Element Number'] == element_num].iloc[0]
    
    node1_id = element_row['Node1']
    node2_id = element_row['Node2']
    
    # Βρες τις συντεταγμένες των κόμβων
    coords1 = nodes_df[nodes_df['Node'] == node1_id][['X', 'Y', 'Z']].values[0]
    coords2 = nodes_df[nodes_df['Node'] == node2_id][['X', 'Y', 'Z']].values[0]
    
    # Υπολογισμός μήκους και συντελεστών κατεύθυνσης
    x1, y1, z1 = coords1
    x2, y2, z2 = coords2
    
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    l = (x2 - x1) / L
    m = (y2 - y1) / L
    n = (z2 - z1) / L
    
    # Δημιουργία πίνακα μετασχηματισμού
    T = np.array([
        [l, m, n, 0, 0, 0],
        [0, 0, 0, l, m, n]
    ])
    
    return T, L


    # Example usage:
    # 
    # import pandas as pd
    # 
    # # Define nodes
    # nodes_df = pd.DataFrame({
    #     'Node': [1, 2, 3],
    #     'X': [0.0, 3.0, 3.0],
    #     'Y': [0.0, 0.0, 4.0],
    #     'Z': [0.0, 0.0, 0.0]
    # })
    # 
    # # Define elements
    # elements_df = pd.DataFrame({
    #     'Element Number': [1, 2],
    #     'Node1': [1, 2],
    #     'Node2': [2, 3],
    #     'E': [200e9, 200e9],
    #     'V': [0.3, 0.3]
    # })
    # 
    # # Calculate transformation matrix for element 1
    # T, L = rotation_matrix_from_element(1, elements_df, nodes_df)
    # 
    # # Returns:
    # # T = array([[1., 0., 0., 0., 0., 0.],
    # #            [0., 0., 0., 1., 0., 0.]])
    # # L = 3.0
    # 
    # # For element 2 (connecting nodes 2 and 3):
    # T2, L2 = rotation_matrix_from_element(2, elements_df, nodes_df)
    # 
    # # Returns:
    # # T2 = array([[0., 1., 0., 0., 0., 0.],
    # #             [0., 0., 0., 0., 1., 0.]])
    # # L2 = 4.0