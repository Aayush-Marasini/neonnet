import numpy as np
import struct
import os

def read_tensor(f):

    rows, cols = struct.unpack('<II', f.read(8))

    data_bytes = f.read(rows * cols * 4)
    data = np.frombuffer(data_bytes, dtype=np.float32)

    return data.reshape((rows,cols))


def main():
    weights_path = "artifacts/weights.bin"
    samples_path = "artifacts/test_samples.bin"

    if not os.path.exists(weights_path) or not os.path.exists(samples_path):
        print("Error: Could not find artifact files. Run train.py first.")
        return
    
    print(f"Loading {weights_path}...")
    with open(weights_path, "rb") as f:
        magic, count = struct.unpack('<II', f.read(8))
        assert magic == 0x4E454F4E, f"Invalid weights magic: {hex(magic)}"
        assert count == 4, f"Expected 4 tensors, got {count}"

        W1 = read_tensor(f)
        b1 = read_tensor(f)
        W2 = read_tensor(f)
        b2 = read_tensor(f)

    print("Parsed Shapes:")
    print(f"  W1: {W1.shape}")
    print(f"  b1: {b1.shape}")
    print(f"  W2: {W2.shape}")
    print(f"  b2: {b2.shape}\n")


    print(f"Verifying forward pass on {samples_path}...")
    with open(samples_path, "rb") as f:
        magic, num_samples = struct.unpack('<II', f.read(8))
        assert magic == 0x44415441, f"Invalid samples magic: {hex(magic)}"

        max_error = 0.0

        for i in range(num_samples):

            label = struct.unpack('<I', f.read(4))[0]

            image = np.frombuffer(f.read(784 * 4), dtype = np.float32)

            expected_logits = np.frombuffer(f.read(10*4), dtype = np.float32)

            z1 = np.matmul(image, W1) + b1.flatten()
            
            a1 = np.maximum(0, z1) 
            
            my_logits = np.matmul(a1, W2) + b2.flatten()
            
            diff = np.max(np.abs(my_logits - expected_logits))
            max_error = max(max_error, diff)
            
            if diff > 1e-4:
                print(f"\n❌ Mismatch at sample {i}! Max diff: {diff:.8f}")
                print(f"Label: {label}")
                print(f"Expected: {expected_logits}")
                print(f"Got:      {my_logits}")
                return

    print(f"✅ Success! All {num_samples} manual forward passes match PyTorch.")
    print(f"Maximum float rounding difference: {max_error:.8e}")

if __name__ == "__main__":
    main()