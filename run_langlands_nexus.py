#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the Langlands NEXUS neural-symbolic architecture
"""

import os
import sys
import argparse
import torch
import numpy as np
import random

# Import Langlands NEXUS components
try:
    from langlands_nexus_architecture import (
        LanglandsNEXUSModel, 
        LanglandsMathDataset, 
        run_langlands_nexus_experiment
    )
    print("Successfully imported Langlands NEXUS components!")
except ImportError as e:
    print(f"Error: Could not import Langlands NEXUS architecture. {e}")
    print("Please ensure the langlands_nexus_architecture.py file is in the same directory.")
    sys.exit(1)

def analyze_custom_mathematical_object(model, properties, problem_type='general'):
    """
    Analyze a custom mathematical object using the trained NEXUS model
    
    Args:
        model: Trained LanglandsNEXUSModel
        properties: List of mathematical properties
        problem_type: Type of mathematical problem
        
    Returns:
        Analysis result
    """
    # Create feature vector
    feature_vector = torch.zeros(1, len(model.symbol_names), dtype=torch.float32)
    
    # Set active properties
    for prop in properties:
        if prop in model.symbol_to_id:
            feature_vector[0, model.symbol_to_id[prop]] = 1
            
    # Set confidence modifiers based on known mathematical results
    modifiers = {
        'has_proof': any(p in ['modular_form', 'functional_equation'] for p in properties),
        'partial_proof': any(p in ['galois_representation', 'automorphic_representation'] for p in properties),
        'empirical_evidence': any(p in ['l_function', 'zeta_function'] for p in properties),
        'contradicts_axiom': False  # No contradictions in this example
    }
    
    # Analyze the mathematical object
    result = model.analyze_mathematical_object(
        feature_vector, 
        active_properties=properties,
        problem_type=problem_type,
        modifiers=modifiers
    )
    
    return result

def run_demo(samples=1000, epochs=5, batch_size=64, lr=0.001, output_dir="results", device=None, seed=42):
    """Run a demonstration of the Langlands NEXUS architecture"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("\n1. Running experiment to train and evaluate model...")
    experiment_results = run_langlands_nexus_experiment(
        num_samples=samples,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_dir=output_dir,
        device=device,
        random_state=seed
    )
    
    model = experiment_results['model']
    
    print("\n2. Analyzing example mathematical objects...")
    
    # Example 1: Galois Group
    print("\nExample 1: Analyzing a Galois Group")
    properties1 = ['algebraic_extension', 'profinite_group', 'galois_representation', 'number_field']
    result1 = analyze_custom_mathematical_object(model, properties1, problem_type='galois_theory')
    print(model.explain_analysis(result1, detail_level='high'))
    
    # Example 2: Automorphic Form
    print("\nExample 2: Analyzing an Automorphic Form")
    properties2 = ['automorphic_representation', 'modular_form', 'cusp_form', 'functional_equation']
    result2 = analyze_custom_mathematical_object(model, properties2, problem_type='automorphic_forms')
    print(model.explain_analysis(result2, detail_level='high'))
    
    # Example 3: L-Function
    print("\nExample 3: Analyzing an L-Function")
    properties3 = ['l_function', 'functional_equation', 'euler_product', 'meromorphic_continuation']
    result3 = analyze_custom_mathematical_object(model, properties3, problem_type='l_functions')
    print(model.explain_analysis(result3, detail_level='high'))
    
    print("\n3. Experiment complete!")
    
    return experiment_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Langlands NEXUS Demo")
    parser.add_argument("--samples", type=int, default=1000, 
                      help="Number of synthetic mathematical objects (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Output directory (default: results)")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage even if CUDA is available")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Langlands NEXUS Neural-Symbolic Architecture Demo")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Number of samples: {args.samples}")
    print(f"Number of epochs: {args.epochs}")
    
    run_demo(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed
    )
    
    print("=" * 80)