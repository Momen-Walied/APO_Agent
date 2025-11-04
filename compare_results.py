"""Compare experiment results and generate summary table."""
import json
import sys
from pathlib import Path

def load_report(path):
    with open(path) as f:
        return json.load(f)

def main():
    reports_dir = Path("reports")
    
    experiments = [
        ("Baseline (no opt)", "baseline_finer_ord.json"),
        ("Phase1 Optimized", "phase1_optimized_finer_ord.json"),
        ("Tiered Models", "tiered_models_finer_ord.json"),
    ]
    
    print("=" * 80)
    print("Phase 1 Experiment Results Comparison")
    print("=" * 80)
    print()
    
    results = []
    for name, filename in experiments:
        path = reports_dir / filename
        if not path.exists():
            print(f"⚠️  {name}: {filename} not found")
            continue
        
        rep = load_report(path)
        results.append({
            "name": name,
            "samples": rep.get("num_samples", 0),
            "baseline_f1": rep.get("baseline_span_f1", 0),
            "post_f1": rep.get("post_adaptation_span_f1", 0),
            "f1_improvement": rep.get("span_f1_improvement", 0),
            "adaptation_steps": rep.get("adaptation_steps_added", 0),
            "reflector_calls": rep.get("usage", {}).get("by_stage", {}).get("reflector", {}).get("calls", 0),
            "reflector_latency": rep.get("usage", {}).get("by_stage", {}).get("reflector", {}).get("latency_seconds", 0),
            "total_calls": rep.get("usage", {}).get("calls", 0),
        })
    
    if not results:
        print("No results found. Run experiments first:")
        print("  powershell -File run_phase1_experiments.ps1")
        sys.exit(1)
    
    # Print table
    print(f"{'Experiment':<20} {'Samples':>7} {'Base F1':>8} {'Post F1':>8} {'ΔF1':>7} {'Steps':>6} {'Ref Calls':>10} {'Ref Latency':>12} {'Total Calls':>11}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['name']:<20} {r['samples']:>7} {r['baseline_f1']:>8.3f} {r['post_f1']:>8.3f} "
              f"{r['f1_improvement']:>+7.3f} {r['adaptation_steps']:>6} {r['reflector_calls']:>10} "
              f"{r['reflector_latency']:>12.1f}s {r['total_calls']:>11}")
    
    print()
    print("Key Metrics:")
    if len(results) >= 2:
        baseline = results[0]
        optimized = results[1]
        
        call_reduction = (baseline["reflector_calls"] - optimized["reflector_calls"]) / max(1, baseline["reflector_calls"]) * 100
        latency_reduction = (baseline["reflector_latency"] - optimized["reflector_latency"]) / max(1, baseline["reflector_latency"]) * 100
        f1_delta = optimized["post_f1"] - baseline["post_f1"]
        
        print(f"  Reflector calls:   {baseline['reflector_calls']:>3} → {optimized['reflector_calls']:>3} ({call_reduction:+.1f}%)")
        print(f"  Reflector latency: {baseline['reflector_latency']:>6.1f}s → {optimized['reflector_latency']:>6.1f}s ({latency_reduction:+.1f}%)")
        print(f"  Post-adapt F1:     {baseline['post_f1']:.3f} → {optimized['post_f1']:.3f} ({f1_delta:+.3f})")
        print()
        
        if abs(f1_delta) <= 0.02 and (call_reduction > 15 or latency_reduction > 15):
            print("✅ Phase 1 optimizations accepted: Cost/latency reduced with <2% F1 loss")
        elif abs(f1_delta) <= 0.05:
            print("⚠️  Phase 1 optimizations weak: Consider tuning convergence_threshold")
        else:
            print("❌ Phase 1 optimizations rejected: F1 loss > 5%")

if __name__ == "__main__":
    main()
