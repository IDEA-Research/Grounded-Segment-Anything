import platform
import sys
import time

import torch


EXPECTED_DEVICE_NAME = "RTX 5060"
EXPECTED_CAPABILITY = (12, 0)
EXPECTED_ARCH = "sm_120"
MATMUL_SHAPE = (1024, 1024)


def print_line(label, value):
    print(f"{label:<28}: {value}")


def fail(message):
    print(f"\n[FAIL] {message}")
    raise SystemExit(1)


def main():
    print("=== Blackwell CUDA Validation ===")
    print_line("Python", sys.version.replace("\n", " "))
    print_line("Platform", platform.platform())
    print_line("Torch version", torch.__version__)
    print_line("Torch CUDA runtime", torch.version.cuda)

    cuda_available = torch.cuda.is_available()
    print_line("torch.cuda.is_available", cuda_available)
    if not cuda_available:
        fail("CUDA is not available. Install the Nightly cu128 wheels before continuing.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    arch = f"sm_{capability[0]}{capability[1]}0"
    total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 3
    arch_list = torch.cuda.get_arch_list()

    print_line("CUDA device index", device_index)
    print_line("CUDA device name", device_name)
    print_line("CUDA capability", capability)
    print_line("Derived arch", arch)
    print_line("Total VRAM (GiB)", f"{total_memory_gb:.2f}")
    print_line("Torch arch list", ", ".join(arch_list))
    print_line(f"{EXPECTED_ARCH} in arch list", EXPECTED_ARCH in arch_list)

    if EXPECTED_DEVICE_NAME.lower() not in device_name.lower():
        fail(f"Expected to detect {EXPECTED_DEVICE_NAME}, but got {device_name}.")
    if capability != EXPECTED_CAPABILITY:
        fail(f"Expected CUDA capability {EXPECTED_CAPABILITY}, but got {capability}.")
    if EXPECTED_ARCH not in arch_list:
        fail(f"{EXPECTED_ARCH} is missing from torch.cuda.get_arch_list().")

    print("\n=== Minimal GPU Tensor Smoke Test ===")
    torch.cuda.reset_peak_memory_stats(device_index)
    a = torch.randn(MATMUL_SHAPE, device="cuda", dtype=torch.float16)
    b = torch.randn(MATMUL_SHAPE, device="cuda", dtype=torch.float16)

    start = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize(device_index)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print_line("Matmul shape", MATMUL_SHAPE)
    print_line("Result shape", tuple(c.shape))
    print_line("Result dtype", c.dtype)
    print_line("Result mean", f"{c.float().mean().item():.6f}")
    print_line("Result checksum", f"{c.float().abs().sum().item():.6f}")
    print_line("Elapsed (ms)", f"{elapsed_ms:.3f}")
    print_line("Peak VRAM (MiB)", f"{torch.cuda.max_memory_allocated(device_index) / 1024 ** 2:.2f}")

    print("\n[PASS] Nightly CUDA stack can see the RTX 5060, reports sm_120, and runs GPU matmul successfully.")


if __name__ == "__main__":
    main()
