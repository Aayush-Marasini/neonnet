# NeonNet Binary Weight Format

All integers are 32-bit unsigned, little-endian. All floats are 32-bit (IEEE 754), little-endian. Matrices are stored in row-major order.

PyTorch's nn.Linear stores its weights as (out_features, in_features) and internally computes x @ W.T . To keep the C matrix library generic (mat_mul(a, b) requires a.cols == b.rows), weights are exported transposed, as (in_features, out_features) — so W1 is 784×128, not PyTorch's native 128×784.

## Header
1. **Magic Number:** `0x4E454F4E` (Spells "NEON" in ASCII) - Used by C to verify this is the right file type.
2. **Tensor Count:** `4` (W1, b1, W2, b2)

## Tensor Data (Repeated 4 times)
For each tensor, in this exact order (W1, b1, W2, b2):
1. **Rows:** (uint32)
2. **Cols:** (uint32)
3. **Data:** (float32 array of size `Rows * Cols`)

## Visual Layout
[Magic: 4 bytes] [Count: 4 bytes] 
[W1 Rows: 4b] [W1 Cols: 4b] [W1 Data: 784 * 128 * 4 bytes]
[b1 Rows: 4b] [b1 Cols: 4b] [b1 Data: 1 * 128 * 4 bytes]
[W2 Rows: 4b] [W2 Cols: 4b] [W2 Data: 128 * 10 * 4 bytes]
[b2 Rows: 4b] [b2 Cols: 4b] [b2 Data: 1 * 10 * 4 bytes]

## Test Samples Format (test_samples.bin)

Used to validate the C forward pass against PyTorch's own predictions.

### Header
1. **Magic Number:** `0x44415441` ("DATA" in ASCII)
2. **Sample Count:** (uint32) — number of samples in this file

### Per-Sample Data (repeated `Sample Count` times)
1. **Label:** (uint32) — the true digit, 0-9
2. **Image:** (float32 array, size 784) — flattened 28x28 pixel values, rescales every pixel from the 0-255 range down to 0.0-1.0 by dividing every value by 255 . 
3. **Output Logits:** (float32 array, size 10) — PyTorch's raw output scores for this image, one per digit class