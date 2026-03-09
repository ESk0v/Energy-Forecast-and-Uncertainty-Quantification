import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def compare_tuning_results(results_csv, output_dir='.'):
    """
    Analyze and visualize hyperparameter tuning results.
    
    Args:
        results_csv: Path to Optuna results CSV file
        output_dir: Directory to save plots
    """
    
    # Load results
    df = pd.read_csv(results_csv)
    
    # Filter out pruned/failed trials
    df_complete = df[df['state'] == 'COMPLETE'].copy()
    
    if len(df_complete) == 0:
        print("No completed trials found!")
        return
    
    print("=" * 70)
    print("HYPERPARAMETER TUNING ANALYSIS")
    print("=" * 70)
    print(f"\nTotal trials: {len(df)}")
    print(f"Completed trials: {len(df_complete)}")
    print(f"Pruned trials: {len(df[df['state'] == 'PRUNED'])}")
    print(f"Failed trials: {len(df[df['state'] == 'FAIL'])}")
    
    # Best trial
    best_idx = df_complete['value'].idxmin()
    best_trial = df_complete.loc[best_idx]
    
    print(f"\n{'─' * 70}")
    print("BEST TRIAL")
    print('─' * 70)
    print(f"Trial number: {best_trial['number']}")
    print(f"Validation Loss: {best_trial['value']:.6f}")
    print(f"\nHyperparameters:")
    
    param_cols = [col for col in df_complete.columns if col.startswith('params_')]
    for col in param_cols:
        param_name = col.replace('params_', '')
        print(f"  {param_name}: {best_trial[col]}")
    
    # Statistics
    print(f"\n{'─' * 70}")
    print("PERFORMANCE STATISTICS")
    print('─' * 70)
    print(f"Best loss:    {df_complete['value'].min():.6f}")
    print(f"Worst loss:   {df_complete['value'].max():.6f}")
    print(f"Median loss:  {df_complete['value'].median():.6f}")
    print(f"Mean loss:    {df_complete['value'].mean():.6f}")
    print(f"Std dev:      {df_complete['value'].std():.6f}")
    
    # Improvement
    if len(df_complete) > 1:
        first_trial_loss = df_complete.iloc[0]['value']
        improvement = ((first_trial_loss - best_trial['value']) / first_trial_loss) * 100
        print(f"\nImprovement from first trial: {improvement:.2f}%")
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Optimization history
    ax1 = plt.subplot(3, 3, 1)
    trials = df_complete['number'].values
    values = df_complete['value'].values
    
    # Running minimum
    running_min = np.minimum.accumulate(values)
    
    ax1.scatter(trials, values, alpha=0.4, s=30, label='Trial loss')
    ax1.plot(trials, running_min, 'r-', linewidth=2, label='Best so far')
    ax1.axhline(best_trial['value'], color='green', linestyle='--', 
                alpha=0.7, label=f'Best: {best_trial["value"]:.4f}')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Validation Loss (MSE)')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2-6. Parameter distributions for each hyperparameter
    param_plots = [
        ('params_hidden_size', 'Hidden Size'),
        ('params_num_layers', 'Number of Layers'),
        ('params_dropout', 'Dropout Rate'),
        ('params_batch_size', 'Batch Size'),
        ('params_learning_rate', 'Learning Rate (log scale)')
    ]
    
    for idx, (param_col, param_name) in enumerate(param_plots, start=2):
        if param_col not in df_complete.columns:
            continue
        
        ax = plt.subplot(3, 3, idx)
        
        # Scatter plot: parameter value vs loss
        x = df_complete[param_col].values
        y = df_complete['value'].values
        
        ax.scatter(x, y, alpha=0.5, s=40, c=trials, cmap='viridis')
        
        # Highlight best trial
        ax.scatter([best_trial[param_col]], [best_trial['value']], 
                  color='red', s=200, marker='*', edgecolors='black', 
                  linewidths=2, label='Best', zorder=10)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'{param_name} vs Loss')
        
        # Use log scale for learning rate
        if 'learning_rate' in param_col:
            ax.set_xscale('log')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Loss distribution
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(df_complete['value'], bins=30, alpha=0.7, edgecolor='black')
    ax7.axvline(best_trial['value'], color='red', linestyle='--', 
                linewidth=2, label=f'Best: {best_trial["value"]:.4f}')
    ax7.set_xlabel('Validation Loss')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Loss Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Trial duration
    if 'duration' in df_complete.columns:
        ax8 = plt.subplot(3, 3, 8)
        durations = df_complete['duration'].dt.total_seconds() / 60  # Convert to minutes
        ax8.scatter(trials, durations, alpha=0.5, s=30)
        ax8.set_xlabel('Trial Number')
        ax8.set_ylabel('Duration (minutes)')
        ax8.set_title('Trial Duration')
        ax8.grid(True, alpha=0.3)
    
    # 9. Pruning analysis
    ax9 = plt.subplot(3, 3, 9)
    state_counts = df['state'].value_counts()
    colors = {'COMPLETE': 'green', 'PRUNED': 'orange', 'FAIL': 'red'}
    ax9.bar(state_counts.index, state_counts.values, 
           color=[colors.get(state, 'gray') for state in state_counts.index])
    ax9.set_ylabel('Count')
    ax9.set_title('Trial States')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'tuning_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n{'─' * 70}")
    print(f"Analysis plot saved to: {output_file}")
    print('─' * 70)
    
    plt.close()
    
    # Print parameter importance (simple variance-based)
    print(f"\n{'─' * 70}")
    print("PARAMETER SENSITIVITY (based on variance)")
    print('─' * 70)
    
    param_importance = {}
    for col in param_cols:
        param_name = col.replace('params_', '')
        # Calculate correlation between parameter and loss
        if df_complete[col].dtype in [np.float64, np.int64]:
            correlation = abs(df_complete[col].corr(df_complete['value']))
            param_importance[param_name] = correlation
    
    # Sort by importance
    sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
    
    for param, importance in sorted_params:
        bar = '█' * int(importance * 50)
        print(f"{param:20s} {bar} {importance:.4f}")
    
    print("\nNote: Higher values indicate stronger correlation with validation loss")
    print("      (doesn't prove causation, but suggests importance)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize hyperparameter tuning results'
    )
    parser.add_argument('results_csv', type=str,
                       help='Path to Optuna results CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save plots (default: current directory)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_csv):
        print(f"Error: File not found: {args.results_csv}")
        return
    
    compare_tuning_results(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()