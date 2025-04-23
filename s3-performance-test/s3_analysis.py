#!/usr/bin/env python3
"""
AWS S3 Performance Analysis Script

This script analyzes the results from the S3 performance testing script,
generating visualizations and statistical summaries.

Usage:
  python s3_analysis.py --results-file s3_performance_results_YYYYMMDD_HHMMSS.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_results(file_path):
    """Load test results from CSV file."""
    df = pd.read_csv(file_path)
    return df

def generate_latency_plots(df, output_dir):
    """Generate plots comparing latency across storage classes and file sizes."""
    plt.figure(figsize=(12, 8))
    
    # Filter for single PUT operations
    single_put = df[df['operation'] == 'single_put']
    
    # Plot upload latency
    plt.subplot(2, 1, 1)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.errorbar(
            class_data['file_size_mb'], 
            class_data['upload_time_mean_s'],
            yerr=class_data['upload_time_ci95_s'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Upload Time (s)')
    plt.title('Upload Latency by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot download latency
    plt.subplot(2, 1, 2)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.errorbar(
            class_data['file_size_mb'], 
            class_data['download_time_mean_s'],
            yerr=class_data['download_time_ci95_s'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Download Time (s)')
    plt.title('Download Latency by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300)
    plt.close()

def generate_throughput_plots(df, output_dir):
    """Generate plots comparing throughput across storage classes and file sizes."""
    plt.figure(figsize=(12, 8))
    
    # Filter for single PUT operations
    single_put = df[df['operation'] == 'single_put']
    
    # Plot upload throughput
    plt.subplot(2, 1, 1)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.plot(
            class_data['file_size_mb'], 
            class_data['upload_throughput_mbps'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Upload Throughput by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot download throughput
    plt.subplot(2, 1, 2)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.plot(
            class_data['file_size_mb'], 
            class_data['download_throughput_mbps'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Download Throughput by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300)
    plt.close()

def generate_ttfb_plot(df, output_dir):
    """Generate plot comparing Time To First Byte across storage classes."""
    plt.figure(figsize=(10, 6))
    
    # Filter for single PUT operations
    single_put = df[df['operation'] == 'single_put']
    
    # Group by storage class and file size
    grouped = single_put.groupby(['storage_class', 'file_size_mb'])['ttfb_mean_ms'].mean().reset_index()
    
    # Pivot for plotting
    pivot_data = grouped.pivot(index='file_size_mb', columns='storage_class', values='ttfb_mean_ms')
    
    # Plot
    ax = pivot_data.plot(kind='bar', figsize=(10, 6))
    plt.xlabel('File Size (MB)')
    plt.ylabel('Time to First Byte (ms)')
    plt.title('Time to First Byte by Storage Class and File Size')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Storage Class')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttfb_comparison.png'), dpi=300)
    plt.close()

def generate_upload_method_comparison(df, output_dir):
    """Generate plot comparing single PUT vs. multipart upload performance."""
    # Filter results for files tested with both methods
    large_files = df[df['file_size_mb'] >= 50].copy()
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Aggregate data by storage class, file size, and operation
    agg_data = large_files.groupby(['storage_class', 'file_size_mb', 'operation'])[
        ['upload_time_mean_s', 'upload_throughput_mbps']
    ].mean().reset_index()
    
    # Pivot for plotting
    time_data = agg_data.pivot_table(
        index=['storage_class', 'file_size_mb'],
        columns='operation',
        values='upload_time_mean_s'
    ).reset_index()
    
    # Create subplots
    plt.subplot(2, 1, 1)
    
    # Group by storage class
    for storage_class in time_data['storage_class'].unique():
        class_data = time_data[time_data['storage_class'] == storage_class]
        
        # Calculate improvement percentage
        class_data['improvement'] = ((class_data['single_put'] - class_data['multipart']) / class_data['single_put']) * 100
        
        plt.plot(
            class_data['file_size_mb'],
            class_data['improvement'],
            marker='o',
            label=storage_class
        )
    
    plt.xlabel('File Size (MB)')
    plt.ylabel('Time Improvement (%)')
    plt.title('Upload Time Improvement: Multipart vs. Single PUT')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Throughput comparison
    throughput_data = agg_data.pivot_table(
        index=['storage_class', 'file_size_mb'],
        columns='operation',
        values='upload_throughput_mbps'
    ).reset_index()
    
    plt.subplot(2, 1, 2)
    
    # Group by storage class
    for storage_class in throughput_data['storage_class'].unique():
        class_data = throughput_data[throughput_data['storage_class'] == storage_class]
        
        # Calculate improvement percentage
        class_data['improvement'] = ((class_data['multipart'] - class_data['single_put']) / class_data['single_put']) * 100
        
        plt.plot(
            class_data['file_size_mb'],
            class_data['improvement'],
            marker='o',
            label=storage_class
        )
    
    plt.xlabel('File Size (MB)')
    plt.ylabel('Throughput Improvement (%)')
    plt.title('Upload Throughput Improvement: Multipart vs. Single PUT')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'upload_method_comparison.png'), dpi=300)
    plt.close()

def generate_cost_efficiency_plots(df, output_dir):
    """Generate plots showing cost efficiency metrics."""
    plt.figure(figsize=(12, 8))
    
    # Filter for single PUT operations
    single_put = df[df['operation'] == 'single_put']
    
    # Plot cost per GB
    plt.subplot(2, 1, 1)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.plot(
            class_data['file_size_mb'], 
            class_data['cost_per_gb_usd'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Cost per GB (USD)')
    plt.title('Cost per GB by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot performance per dollar
    plt.subplot(2, 1, 2)
    for storage_class in single_put['storage_class'].unique():
        class_data = single_put[single_put['storage_class'] == storage_class]
        plt.plot(
            class_data['file_size_mb'], 
            class_data['cost_efficiency_mbps_per_cent'],
            marker='o',
            label=storage_class
        )
    
    plt.xscale('log')
    plt.xlabel('File Size (MB)')
    plt.ylabel('MB/s per $0.01')
    plt.title('Performance per Dollar by Storage Class and File Size')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_efficiency.png'), dpi=300)
    plt.close()

def generate_summary_tables(df, output_dir):
    """Generate summary tables with key performance metrics."""
    # Filter for single PUT operations
    single_put = df[df['operation'] == 'single_put']
    
    # Calculate upload/download ratio
    single_put['upload_download_ratio'] = single_put['download_throughput_mbps'] / single_put['upload_throughput_mbps']
    
    # Group by storage class and file size
    grouped = single_put.groupby(['storage_class', 'file_size_mb']).agg({
        'upload_time_mean_s': 'mean',
        'download_time_mean_s': 'mean',
        'upload_throughput_mbps': 'mean',
        'download_throughput_mbps': 'mean',
        'ttfb_mean_ms': 'mean',
        'upload_download_ratio': 'mean',
        'total_cost_usd': 'mean',
        'cost_per_gb_usd': 'mean',
        'cost_efficiency_mbps_per_cent': 'mean'
    }).reset_index()
    
    # Round to reasonable precision
    for col in grouped.columns:
        if col not in ['storage_class', 'file_size_mb']:
            grouped[col] = grouped[col].round(4)
    
    # Save summary table
    summary_file = os.path.join(output_dir, 'performance_summary.csv')
    grouped.to_csv(summary_file, index=False)
    
    # Create multipart vs single PUT comparison table (for files ≥ 100MB)
    large_files = df[df['file_size_mb'] >= 100].copy()
    
    # Calculate improvement percentages
    upload_methods = large_files.pivot_table(
        index=['storage_class', 'file_size_mb'],
        columns='operation',
        values=['upload_time_mean_s', 'upload_throughput_mbps', 'total_cost_usd']
    ).reset_index()
    
    # Flatten the column names
    upload_methods.columns = ['_'.join(col).strip('_') for col in upload_methods.columns.values]
    
    # Calculate improvements
    upload_methods['time_improvement_pct'] = ((upload_methods['upload_time_mean_s_single_put'] - 
                                              upload_methods['upload_time_mean_s_multipart']) / 
                                             upload_methods['upload_time_mean_s_single_put'] * 100).round(2)
    
    upload_methods['throughput_improvement_pct'] = ((upload_methods['upload_throughput_mbps_multipart'] - 
                                                    upload_methods['upload_throughput_mbps_single_put']) / 
                                                   upload_methods['upload_throughput_mbps_single_put'] * 100).round(2)
    
    # Select relevant columns for the summary
    method_comparison = upload_methods[[
        'storage_class', 
        'file_size_mb', 
        'upload_time_mean_s_single_put',
        'upload_time_mean_s_multipart',
        'time_improvement_pct',
        'upload_throughput_mbps_single_put',
        'upload_throughput_mbps_multipart',
        'throughput_improvement_pct'
    ]]
    
    # Save method comparison table
    method_file = os.path.join(output_dir, 'upload_method_comparison.csv')
    method_comparison.to_csv(method_file, index=False)
    
    return grouped, method_comparison

def main():
    """Main function to run the S3 performance analysis."""
    parser = argparse.ArgumentParser(description='Analyze AWS S3 performance test results')
    parser.add_argument('--results-file', required=True, help='CSV file with test results')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"s3_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = load_results(args.results_file)
    
    # Generate plots
    generate_latency_plots(df, output_dir)
    generate_throughput_plots(df, output_dir)
    generate_ttfb_plot(df, output_dir)
    generate_upload_method_comparison(df, output_dir)
    generate_cost_efficiency_plots(df, output_dir)
    
    # Generate summary tables
    summary, method_comparison = generate_summary_tables(df, output_dir)
    
    print(f"Analysis completed. Results saved to {output_dir}/")

if __name__ == '__main__':
    main()