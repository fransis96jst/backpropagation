# backpropagation
utdi321

Kita punya jaringan saraf tiruan untuk XOR dengan arsitektur:   
<img src="img/arsitektur_jaringan_s1.png" width="500" alt="jaringan1">

- 2 neuron input: X₁, X₂  
- 1 bias input (selalu = 1)  
- 3 neuron hidden layer: Z₁, Z₂, Z₃  
- 1 neuron output: Y₁  
- Fungsi aktivasi: **sigmoid** di semua neuron (yang paling umum untuk backpropagation klasik)

Diketahui bobot awal:

**Tabel a** → bobot dari input ke hidden layer (termasuk bias)

| Dari \ Ke | Z₁    | Z₂    | Z₃    |
|-----------|-------|-------|-------|
| X₁        | -0.3  | 0.3   | -0.3  |
| X₂        | 0.3   | -0.2  | 0.3   |
| bias (1)  | -0.1  | 0.3   | -0.1  |

**Tabel b** → bobot dari hidden layer ke output (termasuk bias)

| Dari \ Ke | Y₁    |
|-----------|-------|
| Z₁        | -0.1  |
| Z₂        | 0.5   |
| Z₃        | -0.3  |
| bias (1)  | 0.2   |

Laju pembelajaran α = 0.2  
Pola pertama yang akan kita latih: **X₁=1, X₂=1, target t = 0** (karena 1 XOR 1 = 0)

### Langkah 1: Forward Pass

Hitung input ke setiap neuron hidden:

Z₁(in) = X₁·w₁₁ + X₂·w₂₁ + 1·bias₁ = (1)(-0.3) + (1)(0.3) + (1)(-0.1) = **-0.3 + 0.3 - 0.1 = -0.1**

Z₂(in) = (1)(0.3) + (1)(-0.2) + (1)(0.3) = 0.3 - 0.2 + 0.3 = **0.4**

Z₃(in) = (1)(-0.3) + (1)(0.3) + (1)(-0.1) = -0.3 + 0.3 - 0.1 = **-0.1**

Sekarang aktivasi sigmoid:

σ(x) = 1 / (1 + e⁻ˣ)

Z₁ = σ(-0.1) ≈ 0.4750  
Z₂ = σ(0.4)  ≈ 0.5987  
Z₃ = σ(-0.1) ≈ 0.4750

Hitung input ke output Y₁:

Y₁(in) = Z₁·v₁ + Z₂·v₂ + Z₃·v₃ + bias·v₀  
= (0.4750)(-0.1) + (0.5987)(0.5) + (0.4750)(-0.3) + (1)(0.2)  
= -0.0475 + 0.29935 - 0.1425 + 0.2  
= **0.30935**

Y₁ = σ(0.30935) ≈ **0.5768**

Jadi output jaringan saat ini ≈ **0.577** (padahal targetnya 0 → error besar)

### Langkah 2: Backpropagation – Hitung Error dan Delta

Error di output:  
δ_Y = (t - Y₁) · Y₁ · (1 - Y₁)  
= (0 - 0.5768) · 0.5768 · (1 - 0.5768)  
= (-0.5768) · 0.5768 · 0.4232 ≈ **-0.1407**

Sekarang delta untuk setiap neuron hidden (Z₁, Z₂, Z₃):

δ_Zj = δ_Y · v_j · Zj · (1 - Zj)

δ_Z₁ = -0.1407 · (-0.1) · 0.4750 · (1-0.4750) = 0.01407 · 0.4750 · 0.5250 ≈ **0.00351**  
δ_Z₂ = -0.1407 · (0.5)  · 0.5987 · (1-0.5987) = -0.07035 · 0.5987 · 0.4013 ≈ **-0.01690**  
δ_Z₃ = -0.1407 · (-0.3) · 0.4750 · 0.5250 ≈ 0.04221 · 0.4750 · 0.5250 ≈ **0.01051**

### Langkah 3: Update Bobot (α = 0.2)

#### Update bobot dari hidden ke output (Tabel b)

Δv_j = α · δ_Y · Zj  
Δv_bias = α · δ_Y · 1

v₁ baru = -0.1 + 0.2 · (-0.1407) · 0.4750 ≈ -0.1 - 0.01337 ≈ **-0.1134**  
v₂ baru = 0.5 + 0.2 · (-0.1407) · 0.5987 ≈ 0.5 - 0.01685 ≈ **0.4832**  
v₃ baru = -0.3 + 0.2 · (-0.1407) · 0.4750 ≈ -0.3 - 0.01337 ≈ **-0.3134**  
bias baru = 0.2 + 0.2 · (-0.1407) · 1 ≈ 0.2 - 0.02814 ≈ **0.1719**

#### Update bobot dari input ke hidden (Tabel a)

Δw_ij = α · δ_Zj · Xi  
(ingat X₁=1, X₂=1, bias=1)

**Ke Z₁ (δ_Z₁ ≈ 0.00351):**

w(X₁→Z₁) = -0.3 + 0.2·0.00351·1 ≈ -0.3 + 0.000702 ≈ **-0.2993**  
w(X₂→Z₁) = 0.3 + 0.000702 ≈ **0.3007**  
w(bias→Z₁) = -0.1 + 0.000702 ≈ **-0.0993**

**Ke Z₂ (δ_Z₂ ≈ -0.01690):**

w(X₁→Z₂) = 0.3 + 0.2·(-0.01690)·1 ≈ 0.3 - 0.00338 ≈ **0.2966**  
w(X₂→Z₂) = -0.2 - 0.00338 ≈ **-0.2034**  
w(bias→Z₂) = 0.3 - 0.00338 ≈ **0.2966**

**Ke Z₃ (δ_Z₃ ≈ 0.01051):**

w(X₁→Z₃) = -0.3 + 0.2·0.01051·1 ≈ -0.3 + 0.002102 ≈ **-0.2979**  
w(X₂→Z₃) = 0.3 + 0.002102 ≈ **0.3021**  
w(bias→Z₃) = -0.1 + 0.002102 ≈ **-0.0979**

### Hasil Bobot Baru Setelah 1 Iterasi (pola X₁=1, X₂=1, t=0)

**Tabel a baru (input → hidden):**

|       | Z₁     | Z₂     | Z₃     |
|-------|--------|--------|--------|
| X₁    | -0.2993| 0.2966 | -0.2979|
| X₂    | 0.3007 | -0.2034| 0.3021 |
| bias  | -0.0993| 0.2966 | -0.0979|

**Tabel b baru (hidden → output):**

|       | Y₁     |
|-------|--------|
| Z₁    | -0.1134|
| Z₂    | 0.4832 |
| Z₃    | -0.3134|
| bias  | 0.1719 |

Catatan: Nilai di atas dibulatkan 4 desimal untuk kepraktisan. Kalau kamu butuh lebih presisi (misal 6-8 desimal), bisa kita hitung ulang.

Ini baru **satu iterasi** untuk pola (1,1) → 0.  
Untuk melatih XOR sampai konvergen, kita harus ulangi proses ini secara berurutan/acak untuk keempat pola XOR (00→0, 01→1, 10→1, 11→0) ratusan/kali (biasanya 1000–10000 epoch) sampai error kecil.

Tapi sesuai permintaan soal: kamu minta hanya iterasi untuk pola pertama (1,1) dengan target 0 → sudah selesai!

Kalau ada latihan lanjutan (pola kedua dst), bilang aja ya, kita lanjut! 🚀
