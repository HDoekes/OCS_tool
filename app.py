import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import json
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Optimal Contribution Selection Tool",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if 'num_animals' not in st.session_state:
    st.session_state.num_animals = 5
if 'animal_data' not in st.session_state:
    st.session_state.animal_data = []
if 'A_matrix' not in st.session_state:
    st.session_state.A_matrix = None
if 'lambda_penalty' not in st.session_state:
    st.session_state.lambda_penalty = 1.0
if 'frontier_data' not in st.session_state:
    st.session_state.frontier_data = None

def objective_function(c, g, A, lambda_penalty):
    """Objective function: c'g - lambda * 0.5 * c'Ac"""
    genetic_merit = np.dot(c, g)
    coancestry = 0.5 * np.dot(c, np.dot(A, c))
    return -(genetic_merit - lambda_penalty * coancestry)

def initialize_animals(n):
    """Initialize animal data structure"""
    st.session_state.animal_data = []
    for i in range(n):
        st.session_state.animal_data.append({
            'sex': 'Male',
            'breeding_value': 0.0,
            'contribution': 0.0
        })
    st.session_state.A_matrix = np.eye(n)

def optimize_contributions(lambda_penalty, g, A, male_indices, female_indices):
    """Find optimal contributions"""
    constraints = []
    
    if male_indices:
        constraints.append({
            'type': 'eq',
            'fun': lambda c: np.sum(c[male_indices]) - 0.5
        })
    
    if female_indices:
        constraints.append({
            'type': 'eq',
            'fun': lambda c: np.sum(c[female_indices]) - 0.5
        })
    
    bounds = [(0, None) for _ in range(len(g))]
    
    x0 = np.zeros(len(g))
    if male_indices:
        x0[male_indices] = 0.5 / len(male_indices)
    if female_indices:
        x0[female_indices] = 0.5 / len(female_indices)
    
    result = minimize(
        fun=lambda c: objective_function(c, g, A, lambda_penalty),
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    return result

# Title and description
st.title("🧬 Optimal Contribution Selection Tool")
st.markdown("""
This tool helps optimize breeding contributions to maximize genetic merit while controlling inbreeding.
It implements Optimal Contribution Selection (OCS) theory for genetic improvement programs.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Setup", "⚡ Optimization", "📊 Interactive Analysis"])

# TAB 1: SETUP
with tab1:
    st.header("Setup Parameters")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Number of Animals")
        num_animals = st.number_input("Number of animals:", min_value=2, max_value=50, value=st.session_state.num_animals, step=1)
        
        if st.button("Initialize Animals"):
            st.session_state.num_animals = num_animals
            initialize_animals(num_animals)
            st.success(f"Initialized {num_animals} animals!")
            st.rerun()
        
        st.subheader("Inbreeding Penalty")
        st.session_state.lambda_penalty = st.number_input("Lambda (λ):", min_value=0.0, value=st.session_state.lambda_penalty, step=0.1, format="%.2f")
    
    with col2:
        st.subheader("File Operations")
        
        # Save setup
        if st.button("💾 Save Setup"):
            if st.session_state.animal_data and st.session_state.A_matrix is not None:
                setup_data = {
                    'num_animals': st.session_state.num_animals,
                    'animal_data': st.session_state.animal_data,
                    'A_matrix': st.session_state.A_matrix.tolist(),
                    'lambda_penalty': st.session_state.lambda_penalty
                }
                json_str = json.dumps(setup_data, indent=2)
                st.download_button(
                    label="Download Setup JSON",
                    data=json_str,
                    file_name="ocs_setup.json",
                    mime="application/json"
                )
            else:
                st.warning("Please initialize animals first!")
        
        # Load setup
        uploaded_file = st.file_uploader("📂 Load Setup (JSON)", type=['json'])
        if uploaded_file is not None:
            try:
                setup_data = json.load(uploaded_file)
                st.session_state.num_animals = setup_data['num_animals']
                st.session_state.animal_data = setup_data['animal_data']
                st.session_state.A_matrix = np.array(setup_data['A_matrix'])
                st.session_state.lambda_penalty = setup_data['lambda_penalty']
                st.success("Setup loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading setup: {str(e)}")
    
    # Animal data entry
    if st.session_state.animal_data:
        st.subheader("Animal Data")
        
        # Create dataframe for easier editing
        df_data = []
        for i, animal in enumerate(st.session_state.animal_data):
            df_data.append({
                'Animal': i + 1,
                'Sex': animal['sex'],
                'Breeding Value': animal['breeding_value'],
                'Contribution': animal['contribution']
            })
        
        df = pd.DataFrame(df_data)
        
        # Editable dataframe
        edited_df = st.data_editor(
            df,
            column_config={
                "Animal": st.column_config.NumberColumn("Animal", disabled=True),
                "Sex": st.column_config.SelectboxColumn("Sex", options=["Male", "Female"], required=True),
                "Breeding Value": st.column_config.NumberColumn("Breeding Value", format="%.4f"),
                "Contribution": st.column_config.NumberColumn("Contribution", format="%.6f", min_value=0.0, max_value=1.0)
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Update session state from edited dataframe
        for i in range(len(edited_df)):
            st.session_state.animal_data[i]['sex'] = edited_df.loc[i, 'Sex']
            st.session_state.animal_data[i]['breeding_value'] = float(edited_df.loc[i, 'Breeding Value'])
            st.session_state.animal_data[i]['contribution'] = float(edited_df.loc[i, 'Contribution'])
        
        # A-matrix input
        st.subheader("A-Matrix (Relationship Matrix)")
        st.markdown("Enter the additive genetic relationship matrix. The matrix should be symmetric and positive definite.")
        
        # Show current A-matrix for editing
        A_df = pd.DataFrame(st.session_state.A_matrix, 
                           columns=[f"Animal {i+1}" for i in range(st.session_state.num_animals)],
                           index=[f"Animal {i+1}" for i in range(st.session_state.num_animals)])
        
        edited_A = st.data_editor(A_df, use_container_width=True)
        
        # Update A-matrix
        st.session_state.A_matrix = edited_A.values
        
        # Make symmetric
        if st.button("Make Matrix Symmetric"):
            A = st.session_state.A_matrix
            st.session_state.A_matrix = (A + A.T) / 2
            st.success("Matrix made symmetric!")
            st.rerun()

# TAB 2: OPTIMIZATION
with tab2:
    st.header("Optimization")
    
    if not st.session_state.animal_data:
        st.warning("Please initialize animals in the Setup tab first!")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Find Optimal Contributions", type="primary", use_container_width=True):
                g = np.array([animal['breeding_value'] for animal in st.session_state.animal_data])
                A = st.session_state.A_matrix
                
                male_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Male']
                female_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Female']
                
                if not male_indices or not female_indices:
                    st.error("Need both males and females for optimization!")
                else:
                    with st.spinner("Optimizing..."):
                        result = optimize_contributions(st.session_state.lambda_penalty, g, A, male_indices, female_indices)
                        
                        if result.success:
                            for i, contrib in enumerate(result.x):
                                st.session_state.animal_data[i]['contribution'] = contrib
                            st.success("Optimization completed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Optimization failed: {result.message}")
        
        with col2:
            if st.button("📊 Calculate Current Metrics", use_container_width=True):
                st.session_state.show_metrics = True
        
        with col3:
            if st.button("🔄 Reset Contributions", use_container_width=True):
                for animal in st.session_state.animal_data:
                    animal['contribution'] = 0.0
                st.success("Contributions reset!")
                st.rerun()
        
        # Display metrics
        st.subheader("Current Results")
        
        c = np.array([animal['contribution'] for animal in st.session_state.animal_data])
        g = np.array([animal['breeding_value'] for animal in st.session_state.animal_data])
        A = st.session_state.A_matrix
        
        genetic_merit = np.dot(c, g)
        mean_coancestry = 0.5 * np.dot(c, np.dot(A, c))
        objective_value = genetic_merit - st.session_state.lambda_penalty * mean_coancestry
        
        male_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Male']
        female_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Female']
        
        male_sum = np.sum(c[male_indices]) if male_indices else 0
        female_sum = np.sum(c[female_indices]) if female_indices else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Genetic Merit (c'g)", f"{genetic_merit:.6f}")
        with col2:
            st.metric("Mean Coancestry (0.5*c'Ac)", f"{mean_coancestry:.6f}")
        with col3:
            st.metric("Objective Value", f"{objective_value:.6f}")
        
        st.subheader("Constraint Check")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Male contributions sum", f"{male_sum:.6f}", delta=f"{male_sum - 0.5:.6f}" if abs(male_sum - 0.5) > 1e-6 else "✓")
        with col2:
            st.metric("Female contributions sum", f"{female_sum:.6f}", delta=f"{female_sum - 0.5:.6f}" if abs(female_sum - 0.5) > 1e-6 else "✓")
        
        # Individual contributions
        st.subheader("Individual Contributions")
        contrib_data = []
        for i, animal in enumerate(st.session_state.animal_data):
            contrib_data.append({
                'Animal': i + 1,
                'Sex': animal['sex'],
                'Breeding Value': animal['breeding_value'],
                'Contribution': animal['contribution']
            })
        
        st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)

# TAB 3: INTERACTIVE ANALYSIS
with tab3:
    st.header("Interactive Frontier Analysis")
    
    if not st.session_state.animal_data:
        st.warning("Please initialize animals in the Setup tab first!")
    else:
        st.markdown("""
        Generate the Pareto frontier showing the trade-off between genetic merit and mean coancestry
        across different values of the penalty parameter λ.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_lambda = st.number_input("Min Lambda", min_value=0.01, value=0.1, step=0.1)
        with col2:
            max_lambda = st.number_input("Max Lambda", min_value=0.1, value=10.0, step=0.5)
        with col3:
            step_size = st.number_input("Step Size", min_value=0.1, value=0.5, step=0.1)
        with col4:
            st.write("")  # Spacing
            generate_button = st.button("🚀 Generate Frontier", type="primary", use_container_width=True)
        
        if generate_button:
            if min_lambda >= max_lambda:
                st.error("Min lambda must be less than max lambda!")
            else:
                g = np.array([animal['breeding_value'] for animal in st.session_state.animal_data])
                A = st.session_state.A_matrix
                
                male_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Male']
                female_indices = [i for i, animal in enumerate(st.session_state.animal_data) if animal['sex'] == 'Female']
                
                if not male_indices or not female_indices:
                    st.error("Need both males and females for analysis!")
                else:
                    num_steps = int((max_lambda - min_lambda) / step_size) + 1
                    lambda_values = np.linspace(min_lambda, max_lambda, num_steps)
                    
                    genetic_merits = []
                    coancestries = []
                    contributions_list = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, lambda_val in enumerate(lambda_values):
                        status_text.text(f"Processing λ = {lambda_val:.3f} ({i+1}/{len(lambda_values)})")
                        
                        result = optimize_contributions(lambda_val, g, A, male_indices, female_indices)
                        
                        if result.success:
                            c_opt = result.x
                            genetic_merit = np.dot(c_opt, g)
                            coancestry = 0.5 * np.dot(c_opt, np.dot(A, c_opt))
                            
                            genetic_merits.append(genetic_merit)
                            coancestries.append(coancestry)
                            contributions_list.append(c_opt.copy())
                        else:
                            genetic_merits.append(np.nan)
                            coancestries.append(np.nan)
                            contributions_list.append(None)
                        
                        progress_bar.progress((i + 1) / len(lambda_values))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store frontier data
                    st.session_state.frontier_data = {
                        'lambda_values': lambda_values,
                        'genetic_merits': np.array(genetic_merits),
                        'coancestries': np.array(coancestries),
                        'contributions': contributions_list,
                        'g': g,
                        'A': A
                    }
                    
                    st.success("Frontier generated successfully!")
        
        # Display frontier if available
        if st.session_state.frontier_data is not None:
            frontier = st.session_state.frontier_data
            
            # Filter valid points
            valid_mask = ~(np.isnan(frontier['genetic_merits']) | np.isnan(frontier['coancestries']))
            valid_lambda = frontier['lambda_values'][valid_mask]
            valid_merits = frontier['genetic_merits'][valid_mask]
            valid_coancestries = frontier['coancestries'][valid_mask]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(valid_coancestries, valid_merits, c=valid_lambda, 
                               cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
            ax.plot(valid_coancestries, valid_merits, 'k--', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Mean Coancestry', fontsize=12)
            ax.set_ylabel('Genetic Merit', fontsize=12)
            ax.set_title('Pareto Frontier of Optimal Solutions', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Lambda (λ)', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Point selection
            st.subheader("Explore Specific Solutions")
            
            selected_lambda = st.select_slider(
                "Select a λ value to view details:",
                options=valid_lambda.tolist(),
                format_func=lambda x: f"{x:.3f}"
            )
            
            # Find the index of selected lambda
            selected_idx = np.where(frontier['lambda_values'] == selected_lambda)[0][0]
            
            if frontier['contributions'][selected_idx] is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lambda (λ)", f"{selected_lambda:.3f}")
                with col2:
                    st.metric("Genetic Merit", f"{frontier['genetic_merits'][selected_idx]:.6f}")
                with col3:
                    st.metric("Mean Coancestry", f"{frontier['coancestries'][selected_idx]:.6f}")
                
                st.subheader("Optimal Contributions for Selected λ")
                
                contrib_data = []
                for i, contrib in enumerate(frontier['contributions'][selected_idx]):
                    animal = st.session_state.animal_data[i]
                    contrib_data.append({
                        'Animal': i + 1,
                        'Sex': animal['sex'],
                        'Breeding Value': animal['breeding_value'],
                        'Optimal Contribution': contrib
                    })
                
                st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
                
                if st.button("Apply These Contributions"):
                    for i, contrib in enumerate(frontier['contributions'][selected_idx]):
                        st.session_state.animal_data[i]['contribution'] = contrib
                    st.success("Contributions applied! Go to the Optimization tab to see results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Optimal Contribution Selection Tool | Built with Streamlit</p>
    <p style='font-size: 0.8em; color: gray;'>For educational purposes in animal breeding and genetics</p>
</div>
""", unsafe_allow_html=True)