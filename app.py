import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import json
from io import StringIO
import plotly.graph_objects as go

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
tab1, tab2, tab3, tab4 = st.tabs(["📋 Setup", "⚡ Optimization", "📊 Results", "📈 Interactive Frontier"])

# TAB 1: SETUP
with tab1:
    st.header("Setup Parameters")
    st.markdown("""
    Initiate the population below, by indicating the number of animals, filling in their EBVs, 
    filling in the A-matrix and (if desired), setting predefined contributions. 
    **Make sure to save the Animal data and the A-matrix before moving to the other tabs!**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Number of Animals")
        num_animals = st.number_input("Number of animals:", min_value=2, max_value=50, value=st.session_state.num_animals, step=1)
        
        if st.button("Initialize Animals"):
            st.session_state.num_animals = num_animals
            initialize_animals(num_animals)
            st.success(f"Initialized {num_animals} animals!")
            st.rerun()
    
    # Animal data entry
    if st.session_state.animal_data:
        st.subheader("Animal Data")
        
        with st.form("animal_data_form"):
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
                use_container_width=True,
                key="animal_editor"
            )
            
            # Submit button
            submitted = st.form_submit_button("💾 Save Animal Data", use_container_width=True)
            
            if submitted:
                # Update session state from edited dataframe
                for i in range(len(edited_df)):
                    st.session_state.animal_data[i]['sex'] = edited_df.loc[i, 'Sex']
                    st.session_state.animal_data[i]['breeding_value'] = float(edited_df.loc[i, 'Breeding Value'])
                    st.session_state.animal_data[i]['contribution'] = float(edited_df.loc[i, 'Contribution'])
                st.success("Animal data saved!")
        
        # A-matrix input
        st.subheader("A-Matrix (Relationship Matrix)")
        st.markdown("Enter the **lower triangle** of the additive genetic relationship matrix (including diagonal). The upper triangle will be filled automatically to maintain symmetry.")
        
        with st.form("matrix_form"):
            # Create manual input for lower triangle
            n = st.session_state.num_animals
            
            st.markdown("**Edit the lower triangle (cells at or below the diagonal):**")
            
            # Create columns for better layout
            col_labels = st.columns([1] + [1]*n)
            col_labels[0].write("")
            for j in range(n):
                col_labels[j+1].markdown(f"**A{j+1}**")
            
            # Create input rows
            new_A = np.copy(st.session_state.A_matrix)
            
            for i in range(n):
                cols = st.columns([1] + [1]*n)
                cols[0].markdown(f"**Animal {i+1}**")
                
                for j in range(n):
                    if j <= i:  # Lower triangle including diagonal
                        new_val = cols[j+1].number_input(
                            f"a_{i}_{j}",
                            value=float(st.session_state.A_matrix[i, j]),
                            format="%.4f",
                            label_visibility="collapsed",
                            key=f"matrix_{i}_{j}"
                        )
                        new_A[i, j] = new_val
                        new_A[j, i] = new_val  # Mirror to upper triangle
                    else:  # Upper triangle - show as disabled
                        cols[j+1].text_input(
                            f"a_{i}_{j}_disabled",
                            value="",
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"matrix_disabled_{i}_{j}"
                        )
            
            # Submit button
            matrix_submitted = st.form_submit_button("💾 Save A-Matrix", use_container_width=True)
            
            if matrix_submitted:
                st.session_state.A_matrix = new_A
                st.success("A-Matrix saved!")
        
        # Show the full symmetric matrix (read-only)
        with st.expander("View Full Symmetric Matrix"):
            full_A_df = pd.DataFrame(
                st.session_state.A_matrix,
                columns=[f"Animal {i+1}" for i in range(n)],
                index=[f"Animal {i+1}" for i in range(n)]
            )
            st.dataframe(full_A_df, use_container_width=True)
    
    # File operations at the bottom
    st.markdown("---")
    st.subheader("Saving and Reading the Population Setup")
    st.markdown("""
    With the buttons below you can save the population setup to a .json file, which you can 
    later read into the application for further usage (without needing to manually fill in all values).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Save Setup**")
        if st.button("💾 Generate JSON File", use_container_width=True):
            if st.session_state.animal_data and st.session_state.A_matrix is not None:
                setup_data = {
                    'num_animals': st.session_state.num_animals,
                    'animal_data': st.session_state.animal_data,
                    'A_matrix': st.session_state.A_matrix.tolist(),
                    'lambda_penalty': st.session_state.lambda_penalty
                }
                json_str = json.dumps(setup_data, indent=2)
                st.download_button(
                    label="📥 Download Setup JSON",
                    data=json_str,
                    file_name="ocs_setup.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("Please initialize animals first!")
    
    with col2:
        st.markdown("**Load Setup**")
        uploaded_file = st.file_uploader("📂 Upload JSON file", type=['json'], label_visibility="collapsed")
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

# TAB 2: OPTIMIZATION
with tab2:
    st.header("Optimization")
    
    if not st.session_state.animal_data:
        st.warning("Please initialize animals in the Setup tab first!")
    else:
        # Lambda parameter
        st.subheader("Inbreeding Penalty")
        st.session_state.lambda_penalty = st.number_input(
            "Lambda (λ):", 
            min_value=0.0, 
            value=st.session_state.lambda_penalty, 
            step=0.1, 
            format="%.2f",
            help="Controls the trade-off between genetic merit and inbreeding control. Higher values penalize inbreeding more."
        )
        
        st.markdown("---")
        
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
            if st.button("🔄 Reset Contributions", use_container_width=True):
                for animal in st.session_state.animal_data:
                    animal['contribution'] = 0.0
                st.success("Contributions reset!")
                st.rerun()
        
        # Display current contributions
        st.subheader("Current Contributions")
        contrib_data = []
        for i, animal in enumerate(st.session_state.animal_data):
            contrib_data.append({
                'Animal': i + 1,
                'Sex': animal['sex'],
                'Breeding Value': animal['breeding_value'],
                'Contribution': animal['contribution']
            })
        
        st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)

# TAB 3: RESULTS
with tab3:
    st.header("Results")
    st.markdown("""
    Below are the results, which are either based on the contributions set by the user in the "Setup" tab, 
    or based on the optimized contributions (for a given inbreeding penalty) when the optimization tab has been run.
    """)
    
    if not st.session_state.animal_data:
        st.warning("Please initialize animals in the Setup tab first!")
    else:
        
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
        
        st.subheader("Summary Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Genetic Merit (c'g)", f"{genetic_merit:.6f}")
        with col2:
            st.metric("Mean Coancestry (0.5*c'Ac)", f"{mean_coancestry:.6f}")
        with col3:
            st.metric("Objective Function (c'g - λc'Ac)", f"{objective_value:.6f}")
        
        st.markdown("---")
        st.subheader("Constraint Check")
        col1, col2 = st.columns(2)
        with col1:
            constraint_met = abs(male_sum - 0.5) < 1e-6
            st.metric(
                "Male contributions sum", 
                f"{male_sum:.6f}", 
                delta=None if constraint_met else f"{male_sum - 0.5:.6f}",
                delta_color="normal" if constraint_met else "inverse"
            )
            if constraint_met:
                st.success("✓ Constraint satisfied")
            else:
                st.warning("⚠ Should equal 0.5")
        
        with col2:
            constraint_met = abs(female_sum - 0.5) < 1e-6
            st.metric(
                "Female contributions sum", 
                f"{female_sum:.6f}", 
                delta=None if constraint_met else f"{female_sum - 0.5:.6f}",
                delta_color="normal" if constraint_met else "inverse"
            )
            if constraint_met:
                st.success("✓ Constraint satisfied")
            else:
                st.warning("⚠ Should equal 0.5")
        
        st.markdown("---")
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

# TAB 4: INTERACTIVE FRONTIER
with tab4:
    st.header("Interactive Frontier Analysis")
    
    if not st.session_state.animal_data:
        st.warning("Please initialize animals in the Setup tab first!")
    else:
        st.markdown("""
        Generate the frontier showing the trade-off between genetic merit and mean coancestry
        across different values of the penalty parameter λ.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_lambda = st.number_input("Min Lambda", min_value=0.01, value=0.1, step=0.1)
        with col2:
            max_lambda = st.number_input("Max Lambda", min_value=0.1, value=10.0, step=0.5)
        with col3:
            num_points = st.number_input("Number of Points", min_value=5, max_value=100, value=20, step=5)
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
            
            # Create plot with Plotly for interactivity
            fig = go.Figure()
            
            # Add scatter plot with hover information
            fig.add_trace(go.Scatter(
                x=valid_coancestries,
                y=valid_merits,
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color='#2E86AB',  # Single color - blue
                    line=dict(width=1.5, color='black')
                ),
                line=dict(width=2.5, color='#2E86AB'),
                hovertemplate='<b>λ = %{customdata:.3f}</b><br>' +
                             'Coancestry = %{x:.6f}<br>' +
                             'Genetic Merit = %{y:.6f}<br>' +
                             '<extra></extra>',
                customdata=valid_lambda,
                showlegend=False
            ))
            
            # Calculate x-axis range and create ticks every 0.05
            x_min = np.floor(valid_coancestries.min() * 20) / 20  # Round down to nearest 0.05
            x_max = np.ceil(valid_coancestries.max() * 20) / 20   # Round up to nearest 0.05
            x_ticks = np.arange(x_min, x_max + 0.05, 0.05)
            
            fig.update_layout(
                title=dict(
                    text='Frontier of Optimal Solutions',
                    font=dict(size=32, color='black')
                ),
                xaxis=dict(
                    title=dict(text='Mean Coancestry', font=dict(size=24, color='black')),
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,  # Only bottom line
                    tickfont=dict(size=20, color='black'),
                    tickmode='array',
                    tickvals=x_ticks,
                    tickformat='.2f'
                ),
                yaxis=dict(
                    title=dict(text='Genetic Merit', font=dict(size=24, color='black')),
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    showline=True,
                    linewidth=2,
                    linecolor='black',
                    mirror=False,  # Only left line
                    tickfont=dict(size=20, color='black')
                ),
                height=520,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
