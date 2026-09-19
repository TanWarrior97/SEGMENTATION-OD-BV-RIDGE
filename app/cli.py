"""
Production Command-Line Interface for ROP Tri-Modal Retinal Segmentation.
Supports single-file and batch evaluation for Demarcation Ridge, Optic Disc,
Blood Vessels, and ROP Zone Mapping.
"""

import os
import sys
import argparse
import json
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.predictor import ROPPredictor

def format_console_report(all_res, input_file):
    results = all_res["results"]
    ridge = results["ridge"]
    od = results["optic_disc"]
    bv = results["blood_vessels"]
    zones = results["zones"]

    out = "\n" + "=" * 60 + "\n"
    out += "  EPICS ROP TRI-MODAL CLINICAL DIAGNOSTIC REPORT\n"
    out += "=" * 60 + "\n"
    out += f"Input Image       : {os.path.basename(input_file)}\n"
    dims = all_res["image_dimensions"]
    out += f"Dimensions        : {dims['width']} x {dims['height']} px\n"
    out += f"Compute Device    : {all_res['device']}\n"
    out += f"Inference Time    : {all_res['total_inference_time_sec']}s\n"
    out += "-" * 60 + "\n"

    # Ridge Summary
    ridge_status = "DETECTED" if ridge["detected"] else "NOT DETECTED"
    out += f"🔵 DEMARCATION RIDGE: [{ridge_status}]\n"
    if ridge["detected"]:
        m = ridge["metrics"]
        out += f"   - Ridge Pixels : {m['ridge_pixels']:,} px\n"
        out += f"   - Area Coverage: {m['coverage_percent']:.4f}%\n"
        out += f"   - Mean Conf.   : {m['mean_confidence']*100:.2f}%\n"
        out += f"   - Threshold    : {m['threshold_used']}\n"

    # OD Summary
    od_status = "DETECTED" if od["detected"] else "NOT DETECTED"
    out += f"\n🔴 OPTIC DISC (OD)   : [{od_status}]\n"
    if od["detected"] and od.get("geometry"):
        g = od["geometry"]
        out += f"   - Center (X, Y): {g['center']}\n"
        out += f"   - Diameter     : {g['diameter']} px\n"
        out += f"   - Area Pixels  : {g['area_pixels']:,} px\n"

    # BV Summary
    bv_status = "DETECTED" if bv["detected"] else "NOT DETECTED"
    out += f"\n🟢 BLOOD VESSELS (BV): [{bv_status}]\n"
    if bv["detected"]:
        m = bv["metrics"]
        out += f"   - Vessel Pixels: {m['vessel_pixels']:,} px\n"
        out += f"   - Density      : {m['vessel_density_percent']:.4f}%\n"

    # Zone Summary
    if zones["detected"] and zones.get("geometry"):
        zg = zones["geometry"]
        out += f"\n🗺️  ROP ZONE ANALYSIS:\n"
        out += f"   - Macula Dist  : {zg['macula_distance']} px\n"
        out += f"   - Zone I Radius: {zg['zone1_radius']} px\n"
        out += f"   - Zone II Radius: {zg['zone2_radius']} px\n"

    saved = all_res.get("saved_paths", {})
    if saved:
        out += "-" * 60 + "\n"
        out += "Saved Outputs:\n"
        for k, p in saved.items():
            out += f"   - {k:17}: {os.path.basename(p)}\n"
    out += "=" * 60 + "\n"
    return out

def main():
    parser = argparse.ArgumentParser(
        description="EPICS ROP Tri-Modal Diagnostic Segmentation CLI (Ridge, Optic Disc, Blood Vessels)"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input retinal image or directory of images")
    parser.add_argument("--target", "-t", choices=["all", "ridge", "od", "bv", "zones"], default="all",
                        help="Target segmentation modality (default: all)")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory to save generated masks and overlays")
    parser.add_argument("--model-dir", "-m", default="model", help="Path to model directory")
    parser.add_argument("--json", action="store_true", help="Output results in machine-readable JSON")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("Initializing ROP Tri-Modal Predictor...")
    predictor = ROPPredictor(model_dir=args.model_dir)

    # Collect files
    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
    else:
        files = [args.input]

    if not files:
        print(f"No valid images found in '{args.input}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} image(s) to process.")

    all_outputs = []
    for filepath in files:
        base_name = os.path.basename(filepath)
        if args.target == "all":
            res = predictor.predict_all(filepath, save_dir=args.output_dir, base_name=base_name)
            if args.json:
                serializable = {
                    "input": filepath,
                    "dimensions": res["image_dimensions"],
                    "inference_time_sec": res["total_inference_time_sec"],
                    "ridge": res["results"]["ridge"]["metrics"],
                    "optic_disc": res["results"]["optic_disc"]["metrics"],
                    "blood_vessels": res["results"]["blood_vessels"]["metrics"],
                    "zones": res["results"]["zones"]["geometry"],
                    "saved_paths": res["saved_paths"]
                }
                all_outputs.append(serializable)
            else:
                print(format_console_report(res, filepath))
        else:
            # Single modality target
            if args.target == "ridge":
                r = predictor.predict_ridge(filepath)
            elif args.target == "od":
                r = predictor.predict_od(filepath)
            elif args.target == "bv":
                r = predictor.predict_bv(filepath)
            elif args.target == "zones":
                r = predictor.predict_zones(filepath)

            if args.json:
                all_outputs.append({
                    "input": filepath,
                    "modality": args.target,
                    "detected": r["detected"],
                    "metrics": r.get("metrics") or r.get("geometry")
                })
            else:
                print(f"[{args.target.upper()}] Processed {base_name}: Detected = {r['detected']}")

    if args.json:
        print(json.dumps(all_outputs if len(all_outputs) > 1 else all_outputs[0], indent=2))

if __name__ == "__main__":
    main()
