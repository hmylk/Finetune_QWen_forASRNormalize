import argparse
import shutil
import subprocess
from pathlib import Path


PYTHON_BIN = "python"


def parse_args():
    parser = argparse.ArgumentParser(description="Export merged Hugging Face model to GGUF and optional quantized GGUF.")
    parser.add_argument(
        "--merged-model-dir",
        default="./qwen2.5-0.5b-merged",
        help="Merged Hugging Face model directory",
    )
    parser.add_argument(
        "--llamacpp-dir",
        default="/data/nfs_rt18/hongmi/LLM/llama.cpp",
        help="llama.cpp repository directory",
    )
    parser.add_argument(
        "--gguf-output",
        default="./qwen2.5-0.5b-merged-llamacpp.gguf",
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--quantized-output",
        default="./qwen2.5-0.5b-merged-llamacpp-Q4_K_M.gguf",
        help="Quantized GGUF output path",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        help="GGUF output type passed to convert_hf_to_gguf.py",
    )
    parser.add_argument(
        "--quant-type",
        default="Q4_K_M",
        help="Quantization type passed to llama-quantize",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Only export the base GGUF file",
    )
    return parser.parse_args()


def run_command(command, description):
    print(f"[RUN] {description}")
    print(" ".join(command))
    subprocess.run(command, check=True)


def export_gguf(merged_model_dir, gguf_output, llamacpp_dir, outtype):
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing convert script: {convert_script}")

    gguf_output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            PYTHON_BIN,
            str(convert_script),
            str(merged_model_dir),
            "--outfile",
            str(gguf_output),
            "--outtype",
            outtype,
        ],
        "Export Hugging Face model to GGUF",
    )


def quantize_gguf(llamacpp_dir, gguf_output, quantized_output, quant_type):
    quantize_bin = llamacpp_dir / "llama-quantize"
    if not quantize_bin.exists():
        build_bin = llamacpp_dir / "build" / "bin" / "llama-quantize"
        if build_bin.exists():
            quantize_bin = build_bin
        else:
            raise FileNotFoundError(f"Missing llama-quantize binary in {llamacpp_dir}")

    quantized_output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(quantize_bin),
            str(gguf_output),
            str(quantized_output),
            quant_type,
        ],
        "Quantize GGUF model",
    )


def restore_export_snapshot(merged_model_dir):
    snapshot_dir = merged_model_dir.parent / f"{merged_model_dir.name}-llamacpp"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(merged_model_dir, snapshot_dir)
    print(f"[OK] Restored export snapshot: {snapshot_dir}")


def main():
    args = parse_args()
    merged_model_dir = Path(args.merged_model_dir).resolve()
    llamacpp_dir = Path(args.llamacpp_dir).resolve()
    gguf_output = Path(args.gguf_output).resolve()
    quantized_output = Path(args.quantized_output).resolve()

    if not merged_model_dir.exists():
        raise FileNotFoundError(f"Merged model directory not found: {merged_model_dir}")
    if not llamacpp_dir.exists():
        raise FileNotFoundError(f"llama.cpp directory not found: {llamacpp_dir}")

    restore_export_snapshot(merged_model_dir)
    export_gguf(merged_model_dir, gguf_output, llamacpp_dir, args.outtype)

    if not args.skip_quantize:
        quantize_gguf(llamacpp_dir, gguf_output, quantized_output, args.quant_type)

    print("[OK] Export completed")
    print(f"GGUF: {gguf_output}")
    if not args.skip_quantize:
        print(f"Quantized: {quantized_output}")


if __name__ == "__main__":
    main()
