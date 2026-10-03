# Foveated Variable-Resolution 2.5D Grid Design

## 1. Motivation

Standard autonomous vehicle perception pipelines represent the environment using either:
- **Uniform 2.5D Elevation Grids**: Uniform cell size across the entire sensing radius. At fine resolutions (e.g., 5 cm across a 100m radius), a uniform grid requires $2000 \times 2000 = 4,000,000$ cells, consuming massive memory bandwidth and cache capacity.
- **3D Voxel Grids**: Extremely memory-intensive ($O(N^3)$), often exceeding several gigabytes per scan.

The **Adaptive Foveated 2.5D Grid** matches the physical physics of Lidar beam divergence: point density is dense near the sensor and sparse at long range. By scaling cell resolution dynamically across concentric distance zones, high resolution is concentrated where safety-critical decisions happen, while reducing total cell count by **>94%**.

---

## 2. Multi-Zone Configuration

The grid spans a radius of 100 meters partitioned into four concentric zones:

| Zone | Distance Range ($r_{\min} \to r_{\max}$) | Cell Size ($\Delta$) | Dimensions ($W \times H$) | Theoretical Area | Theoretical Cells |
|---|---|---|---|---|---|
| **0: Immediate** | $0 \text{ m} \to 10 \text{ m}$ | $0.05 \text{ m}$ (5 cm) | $400 \times 400$ | $314.16 \text{ m}^2$ | $\approx 125,664$ |
| **1: Near** | $10 \text{ m} \to 30 \text{ m}$ | $0.10 \text{ m}$ (10 cm) | $600 \times 600$ | $2,513.27 \text{ m}^2$ | $\approx 251,327$ |
| **2: Mid** | $30 \text{ m} \to 60 \text{ m}$ | $0.25 \text{ m}$ (25 cm) | $480 \times 480$ | $8,482.30 \text{ m}^2$ | $\approx 135,717$ |
| **3: Far** | $60 \text{ m} \to 100 \text{ m}$ | $0.50 \text{ m}$ (50 cm) | $400 \times 400$ | $20,106.19 \text{ m}^2$ | $\approx 80,425$ |
| **Total (Foveated)** | **$0 \to 100 \text{ m}$** | **Variable (5–50 cm)** | — | **$31,415.93 \text{ m}^2$** | **$\approx 593,133$ cells** |
| *Uniform 5 cm Grid* | *$0 \to 100 \text{ m}$* | *0.05 m (uniform)* | *$4000 \times 4000$* | *$31,415.93 \text{ m}^2$* | *$12,566,370$ cells* |

**Memory Savings**: Over **94.3%** reduction in cell count and memory footprint compared to a uniform 5 cm grid.

---

## 3. Cell Data Structure

Each cell is designed as a cache-aligned struct (32 bytes):

```cpp
struct alignas(32) Cell {
    float min_z;             // Minimum elevation (4 bytes)
    float max_z;             // Maximum elevation (4 bytes)
    float mean_z;            // Mean elevation (4 bytes)
    uint32_t point_count;    // Total points falling in cell (4 bytes)
    float semantic_conf;     // Confidence of dominant semantic class (4 bytes)
    uint8_t semantic_class;  // Dominant class ID: 0-5 (1 byte)
    uint8_t occupancy;       // 0: Free, 1: Occupied, 2: Unknown (1 byte)
    uint8_t zone_id;         // Zone index: 0-3 (1 byte)
    uint8_t reserved[5];     // Padding to exactly 32 bytes (5 bytes)
};
```

---

## 4. Coordinate Projection & Indexing

Given point coordinates $(x, y, z)$:

1. **Radial Distance**:
   $$r = \sqrt{x^2 + y^2}$$

2. **Zone Selection**:
   $$z_{\text{idx}} = \text{find\_zone}(r) \quad \text{such that} \quad r_{\min}^{(i)} \le r < r_{\max}^{(i)}$$

3. **Local Grid Coordinates**:
   $$u = \left\lfloor \frac{x}{\Delta^{(i)}} + \text{offset}_x^{(i)} \right\rfloor, \quad v = \left\lfloor \frac{y}{\Delta^{(i)}} + \text{offset}_y^{(i)} \right\rfloor$$

4. **Linear Flat Index**:
   $$\text{index} = \text{base\_idx}^{(i)} + (v \times W^{(i)} + u)$$

On the GPU, this projection is executed via HIP kernel threads in parallel across thousands of points per millisecond, updating cell stats with atomic operations (`atomicMin`, `atomicMax`, `atomicAdd`).
