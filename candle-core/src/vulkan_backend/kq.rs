//! Experimental K-quant GEMV kernels (Q4_K, Q6_K), llama.cpp style:
//! 16 lanes cooperate on one 256-value super-block, reading whole words,
//! with scales applied once per sub-block. One work-group per output row
//! (2-D grid), `m` on z. Toggle with `CANDLE_VULKAN_KQ=0|1` (default 1).
use super::kernels::{Buf, Push};
use super::VulkanDevice;
use crate::quantized::GgmlDType;
use crate::Result;

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_VULKAN_KQ").map_or(true, |v| v != "0"))
}

/// Work-group size (`CANDLE_VULKAN_KQ_WG`, default 64): WG/16 super-blocks in flight per row.
fn wg() -> usize {
    static WG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WG.get_or_init(|| {
        std::env::var("CANDLE_VULKAN_KQ_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

fn nr() -> usize {
    static NR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *NR.get_or_init(|| {
        std::env::var("CANDLE_VULKAN_KQ_NR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}

pub fn supports(dtype: GgmlDType, k: usize) -> bool {
    enabled() && matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K) && k % 256 == 0
}

fn source(wg: usize, nr: usize) -> String {
    let mut s = super::glsl::prelude(wg, 3);
    s += &format!("#define NR {nr}\n");
    s += r#"
layout(std430, set = 0, binding = 0) readonly buffer B0 { float X[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { uint W[]; };
layout(std430, set = 0, binding = 2) buffer B2 { float C[]; };
layout(push_constant) uniform PC { int N; int K; int xoff; int coff; int M; int qt; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];

uint wword(uint off) {
    uint i = off >> 2u;
    uint s = (off & 3u) * 8u;
    return s == 0u ? W[i] : ((W[i] >> s) | (W[i + 1u] << (32u - s)));
}
uint bsel(uint w0, uint w1, uint w2, int i) {
    uint w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (w >> uint(8 * (i & 3))) & 0xFFu;
}

void main() {
    int n0 = grow() * NR;
    int m = int(gl_WorkGroupID.z);
    int t = int(gl_LocalInvocationID.x);
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.0;
    int lane = t & 15;
    int nb = pc.K / 256;
    int x0 = pc.xoff + m * pc.K;
    if (pc.qt == 12) {
        int g = lane >> 2;
        int l = (lane & 3) * 8;
        int j0 = 2 * g;
        int j1 = j0 + 1;
        for (int b = t >> 4; b < nb; b += WG / 16) {
            int xa = x0 + b * 256 + 64 * g + l;
            int xb = xa + 32;
            float xl[8]; float xh[8];
            float slo = 0.0; float shi = 0.0;
            for (int i = 0; i < 8; i++) { xl[i] = X[xa + i]; xh[i] = X[xb + i]; slo += xl[i]; shi += xh[i]; }
            for (int r = 0; r < NR; r++) {
                int n = n0 + r;
                if (n >= pc.N) break;
                uint wb = (uint(n) * uint(nb) + uint(b)) * 36u;
                vec2 dm = unpackHalf2x16(W[wb]);
                uint s0 = W[wb + 1u];
                uint s1 = W[wb + 2u];
                uint s2 = W[wb + 3u];
                float sc0; float mn0; float sc1; float mn1;
                if (j0 < 4) {
                    sc0 = float(bsel(s0, s1, s2, j0) & 63u);
                    mn0 = float(bsel(s0, s1, s2, j0 + 4) & 63u);
                    sc1 = float(bsel(s0, s1, s2, j1) & 63u);
                    mn1 = float(bsel(s0, s1, s2, j1 + 4) & 63u);
                } else {
                    sc0 = float((bsel(s0, s1, s2, j0 + 4) & 15u) | ((bsel(s0, s1, s2, j0 - 4) >> 6u) << 4u));
                    mn0 = float((bsel(s0, s1, s2, j0 + 4) >> 4u) | ((bsel(s0, s1, s2, j0) >> 6u) << 4u));
                    sc1 = float((bsel(s0, s1, s2, j1 + 4) & 15u) | ((bsel(s0, s1, s2, j1 - 4) >> 6u) << 4u));
                    mn1 = float((bsel(s0, s1, s2, j1 + 4) >> 4u) | ((bsel(s0, s1, s2, j1) >> 6u) << 4u));
                }
                uint q0 = W[wb + 4u + uint(lane) * 2u];
                uint q1 = W[wb + 5u + uint(lane) * 2u];
                float dlo = 0.0; float dhi = 0.0;
                for (int i = 0; i < 4; i++) {
                    uint v = (q0 >> uint(8 * i)) & 0xFFu;
                    uint u = (q1 >> uint(8 * i)) & 0xFFu;
                    dlo += xl[i] * float(v & 15u) + xl[4 + i] * float(u & 15u);
                    dhi += xh[i] * float(v >> 4u) + xh[4 + i] * float(u >> 4u);
                }
                acc[r] += dm.x * (sc0 * dlo + sc1 * dhi) - dm.y * (mn0 * slo + mn1 * shi);
            }
        }
    } else {
        int half_ = lane >> 3;
        int l0 = (lane & 7) * 4;
        int is = l0 / 16;
        for (int b = t >> 4; b < nb; b += WG / 16) {
            int xbase = x0 + b * 256 + 128 * half_ + l0;
            float xv[16];
            for (int l = 0; l < 4; l++) {
                xv[l] = X[xbase + l]; xv[4 + l] = X[xbase + 32 + l];
                xv[8 + l] = X[xbase + 64 + l]; xv[12 + l] = X[xbase + 96 + l];
            }
            for (int r = 0; r < NR; r++) {
                int n = n0 + r;
                if (n >= pc.N) break;
                uint p = (uint(n) * uint(nb) + uint(b)) * 210u;
                uint qa = wword(p + uint(64 * half_ + l0));
                uint qb = wword(p + uint(64 * half_ + 32 + l0));
                uint qh = wword(p + 128u + uint(32 * half_ + l0));
                uint dp = p + 208u;
                float d = unpackHalf2x16((W[dp >> 2u] >> ((dp & 3u) * 8u)) & 0xFFFFu).x;
                uint scw0 = wword(p + 192u + uint(8 * half_));
                uint scw1 = wword(p + 196u + uint(8 * half_));
                float sc[4];
                for (int k2 = 0; k2 < 4; k2++) {
                    int idx = is + 2 * k2;
                    uint bv = ((idx < 4 ? scw0 : scw1) >> uint(8 * (idx & 3))) & 0xFFu;
                    sc[k2] = float(int(bv) - (bv >= 128u ? 256 : 0));
                }
                float s1 = 0.0; float s2 = 0.0; float s3 = 0.0; float s4 = 0.0;
                for (int l = 0; l < 4; l++) {
                    uint a = (qa >> uint(8 * l)) & 0xFFu;
                    uint c = (qb >> uint(8 * l)) & 0xFFu;
                    uint h = (qh >> uint(8 * l)) & 0xFFu;
                    s1 += xv[l] * float(int((a & 15u) | ((h & 3u) << 4u)) - 32);
                    s2 += xv[4 + l] * float(int((c & 15u) | (((h >> 2u) & 3u) << 4u)) - 32);
                    s3 += xv[8 + l] * float(int((a >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                    s4 += xv[12 + l] * float(int((c >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                }
                acc[r] += d * (sc[0] * s1 + sc[1] * s2 + sc[2] * s3 + sc[3] * s4);
            }
        }
    }
    for (int r = 0; r < NR; r++) {
        sh[t] = acc[r];
        barrier();
        for (int s = WG / 2; s > 0; s >>= 1) {
            if (t < s) sh[t] += sh[t + s];
            barrier();
        }
        if (t == 0 && n0 + r < pc.N) C[pc.coff + m * pc.N + n0 + r] = sh[0];
        barrier();
    }
}
"#;
    s
}

/// Multi-token variant: each lane dequantizes its 16 weights of a
/// (row, super-block) once into registers and applies them to MT activation
/// rows. Grid: (rows / NR) x (M / MT).
fn source_mm(wg: usize, nr: usize, mt: usize) -> String {
    let mut s = super::glsl::prelude(wg, 3);
    s += &format!("#define NR {nr}\n#define MT {mt}\n");
    s += r#"
layout(std430, set = 0, binding = 0) readonly buffer B0 { float X[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { uint W[]; };
layout(std430, set = 0, binding = 2) buffer B2 { float C[]; };
layout(push_constant) uniform PC { int N; int K; int xoff; int coff; int M; int qt; } pc;
layout(local_size_x = WG) in;
shared float sh[WG];

uint wword(uint off) {
    uint i = off >> 2u;
    uint s = (off & 3u) * 8u;
    return s == 0u ? W[i] : ((W[i] >> s) | (W[i + 1u] << (32u - s)));
}
uint bsel(uint w0, uint w1, uint w2, int i) {
    uint w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (w >> uint(8 * (i & 3))) & 0xFFu;
}

void main() {
    int n0 = grow() * NR;
    int m0 = int(gl_WorkGroupID.z) * MT;
    int t = int(gl_LocalInvocationID.x);
    float acc[NR * MT];
    for (int i = 0; i < NR * MT; i++) acc[i] = 0.0;
    int lane = t & 15;
    int nb = pc.K / 256;
    // xi[16]: k offsets (within the super-block) of this lane's 16 weights
    int xi[16];
    if (pc.qt == 12) {
        int g = lane >> 2; int l = (lane & 3) * 8;
        for (int i = 0; i < 8; i++) { xi[i] = 64 * g + l + i; xi[8 + i] = 64 * g + 32 + l + i; }
    } else {
        int half_ = lane >> 3; int l0 = (lane & 7) * 4;
        for (int q = 0; q < 4; q++) for (int l = 0; l < 4; l++) xi[4 * q + l] = 128 * half_ + l0 + 32 * q + l;
    }
    for (int b = t >> 4; b < nb; b += WG / 16) {
        for (int r = 0; r < NR; r++) {
            int n = n0 + r;
            if (n >= pc.N) break;
            float w[16];
            if (pc.qt == 12) {
                int g = lane >> 2; int j0 = 2 * g; int j1 = j0 + 1;
                uint wb = (uint(n) * uint(nb) + uint(b)) * 36u;
                vec2 dm = unpackHalf2x16(W[wb]);
                uint s0 = W[wb + 1u]; uint s1 = W[wb + 2u]; uint s2 = W[wb + 3u];
                float sc0; float mn0; float sc1; float mn1;
                if (j0 < 4) {
                    sc0 = float(bsel(s0, s1, s2, j0) & 63u); mn0 = float(bsel(s0, s1, s2, j0 + 4) & 63u);
                    sc1 = float(bsel(s0, s1, s2, j1) & 63u); mn1 = float(bsel(s0, s1, s2, j1 + 4) & 63u);
                } else {
                    sc0 = float((bsel(s0, s1, s2, j0 + 4) & 15u) | ((bsel(s0, s1, s2, j0 - 4) >> 6u) << 4u));
                    mn0 = float((bsel(s0, s1, s2, j0 + 4) >> 4u) | ((bsel(s0, s1, s2, j0) >> 6u) << 4u));
                    sc1 = float((bsel(s0, s1, s2, j1 + 4) & 15u) | ((bsel(s0, s1, s2, j1 - 4) >> 6u) << 4u));
                    mn1 = float((bsel(s0, s1, s2, j1 + 4) >> 4u) | ((bsel(s0, s1, s2, j1) >> 6u) << 4u));
                }
                float a0 = dm.x * sc0; float c0 = dm.y * mn0; float a1 = dm.x * sc1; float c1 = dm.y * mn1;
                uint q0 = W[wb + 4u + uint(lane) * 2u]; uint q1 = W[wb + 5u + uint(lane) * 2u];
                for (int i = 0; i < 4; i++) {
                    uint v = (q0 >> uint(8 * i)) & 0xFFu; uint u = (q1 >> uint(8 * i)) & 0xFFu;
                    w[i] = a0 * float(v & 15u) - c0; w[4 + i] = a0 * float(u & 15u) - c0;
                    w[8 + i] = a1 * float(v >> 4u) - c1; w[12 + i] = a1 * float(u >> 4u) - c1;
                }
            } else {
                int half_ = lane >> 3; int l0 = (lane & 7) * 4; int is = l0 / 16;
                uint p = (uint(n) * uint(nb) + uint(b)) * 210u;
                uint qa = wword(p + uint(64 * half_ + l0));
                uint qb = wword(p + uint(64 * half_ + 32 + l0));
                uint qh = wword(p + 128u + uint(32 * half_ + l0));
                uint dp = p + 208u;
                float d = unpackHalf2x16((W[dp >> 2u] >> ((dp & 3u) * 8u)) & 0xFFFFu).x;
                uint scw0 = wword(p + 192u + uint(8 * half_)); uint scw1 = wword(p + 196u + uint(8 * half_));
                float sc[4];
                for (int k2 = 0; k2 < 4; k2++) {
                    int idx = is + 2 * k2;
                    uint bv = ((idx < 4 ? scw0 : scw1) >> uint(8 * (idx & 3))) & 0xFFu;
                    sc[k2] = d * float(int(bv) - (bv >= 128u ? 256 : 0));
                }
                for (int l = 0; l < 4; l++) {
                    uint a = (qa >> uint(8 * l)) & 0xFFu; uint c = (qb >> uint(8 * l)) & 0xFFu; uint h = (qh >> uint(8 * l)) & 0xFFu;
                    w[l] = sc[0] * float(int((a & 15u) | ((h & 3u) << 4u)) - 32);
                    w[4 + l] = sc[1] * float(int((c & 15u) | (((h >> 2u) & 3u) << 4u)) - 32);
                    w[8 + l] = sc[2] * float(int((a >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32);
                    w[12 + l] = sc[3] * float(int((c >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32);
                }
            }
            for (int mi = 0; mi < MT; mi++) {
                int m = m0 + mi;
                if (m >= pc.M) break;
                int xb = pc.xoff + m * pc.K + b * 256;
                float dsum = 0.0;
                for (int i = 0; i < 16; i++) dsum += w[i] * X[xb + xi[i]];
                acc[r * MT + mi] += dsum;
            }
        }
    }
    for (int r = 0; r < NR; r++) {
        for (int mi = 0; mi < MT; mi++) {
            sh[t] = acc[r * MT + mi];
            barrier();
            for (int s = WG / 2; s > 0; s >>= 1) {
                if (t < s) sh[t] += sh[t + s];
                barrier();
            }
            if (t == 0 && n0 + r < pc.N && m0 + mi < pc.M) C[pc.coff + (m0 + mi) * pc.N + n0 + r] = sh[0];
            barrier();
        }
    }
}
"#;
    s
}

/// Tiled prefill GEMM: C[TM tokens x TN rows] per 256-thread work-group;
/// each step dequantizes a TN x 32 weight tile (one sub-block) and loads a
/// TM x 32 activation tile into shared memory; every thread computes a 4x4
/// micro-tile. Grid: (N / TN) x (M / TM).
fn source_tiled() -> String {
    let mut s = super::glsl::prelude(256, 3);
    s += r#"
#define TM 64
#define TN 64
layout(std430, set = 0, binding = 0) readonly buffer B0 { float X[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { uint W[]; };
layout(std430, set = 0, binding = 2) buffer B2 { float C[]; };
layout(push_constant) uniform PC { int N; int K; int xoff; int coff; int M; int qt; } pc;
layout(local_size_x = 256) in;
shared float Ws[TN * 33];
shared float Xs[TM * 33];

uint wword(uint off) {
    uint i = off >> 2u;
    uint s = (off & 3u) * 8u;
    return s == 0u ? W[i] : ((W[i] >> s) | (W[i + 1u] << (32u - s)));
}
uint bsel(uint w0, uint w1, uint w2, int i) {
    uint w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (w >> uint(8 * (i & 3))) & 0xFFu;
}

void main() {
    int nt = int(gl_WorkGroupID.x) * TN;
    int mt = int(gl_WorkGroupID.y) * TM;
    int t = int(gl_LocalInvocationID.x);
    int lr = t >> 2;          // loader row (0..63)
    int lq = (t & 3) * 8;     // loader column offset inside the 32-wide tile
    int tx = t & 15;          // compute: rows tx*4..
    int ty = t >> 4;          // compute: tokens ty*4..
    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0;
    int nb = pc.K / 256;
    int nsub = pc.K / 32;
    for (int sb = 0; sb < nsub; sb++) {
        int b = sb >> 3;
        int j = sb & 7;
        // activations
        int m = mt + lr;
        for (int i = 0; i < 8; i++) {
            Xs[lr * 33 + lq + i] = m < pc.M ? X[pc.xoff + m * pc.K + sb * 32 + lq + i] : 0.0;
        }
        // weights
        int n = nt + lr;
        float w[8];
        for (int i = 0; i < 8; i++) w[i] = 0.0;
        if (n < pc.N) {
            if (pc.qt == 12) {
                uint wb = (uint(n) * uint(nb) + uint(b)) * 36u;
                vec2 dm = unpackHalf2x16(W[wb]);
                uint s0 = W[wb + 1u]; uint s1 = W[wb + 2u]; uint s2 = W[wb + 3u];
                float sc; float mn;
                if (j < 4) { sc = float(bsel(s0, s1, s2, j) & 63u); mn = float(bsel(s0, s1, s2, j + 4) & 63u); }
                else {
                    sc = float((bsel(s0, s1, s2, j + 4) & 15u) | ((bsel(s0, s1, s2, j - 4) >> 6u) << 4u));
                    mn = float((bsel(s0, s1, s2, j + 4) >> 4u) | ((bsel(s0, s1, s2, j) >> 6u) << 4u));
                }
                float a = dm.x * sc; float c = dm.y * mn;
                uint off = wb + 4u + uint((j >> 1) * 8 + (lq >> 2));
                uint q0 = W[off]; uint q1 = W[off + 1u];
                uint sh_ = (j & 1) != 0 ? 4u : 0u;
                for (int i = 0; i < 4; i++) {
                    w[i] = a * float((q0 >> (uint(8 * i) + sh_)) & 15u) - c;
                    w[4 + i] = a * float((q1 >> (uint(8 * i) + sh_)) & 15u) - c;
                }
            } else {
                uint p = (uint(n) * uint(nb) + uint(b)) * 210u;
                int half_ = j >> 2; int q = j & 3;
                uint qa0 = wword(p + uint(64 * half_ + (q & 1) * 32 + lq));
                uint qa1 = wword(p + uint(64 * half_ + (q & 1) * 32 + lq + 4));
                uint qh0 = wword(p + 128u + uint(32 * half_ + lq));
                uint qh1 = wword(p + 128u + uint(32 * half_ + lq + 4));
                uint dp = p + 208u;
                float d = unpackHalf2x16((W[dp >> 2u] >> ((dp & 3u) * 8u)) & 0xFFFFu).x;
                int sidx = 8 * half_ + lq / 16 + 2 * q;
                uint sw = wword(p + 192u + uint(sidx & ~3));
                uint bv = (sw >> uint(8 * (sidx & 3))) & 0xFFu;
                float sc = d * float(int(bv) - (bv >= 128u ? 256 : 0));
                uint lsh = q >= 2 ? 4u : 0u; uint hsh = uint(2 * q);
                for (int i = 0; i < 4; i++) {
                    uint lo0 = (qa0 >> (uint(8 * i) + lsh)) & 15u; uint h0 = (qh0 >> (uint(8 * i) + hsh)) & 3u;
                    uint lo1 = (qa1 >> (uint(8 * i) + lsh)) & 15u; uint h1 = (qh1 >> (uint(8 * i) + hsh)) & 3u;
                    w[i] = sc * float(int(lo0 | (h0 << 4u)) - 32);
                    w[4 + i] = sc * float(int(lo1 | (h1 << 4u)) - 32);
                }
            }
        }
        for (int i = 0; i < 8; i++) Ws[lr * 33 + lq + i] = w[i];
        barrier();
        for (int kk = 0; kk < 32; kk++) {
            float xr[4]; float wr[4];
            for (int i = 0; i < 4; i++) { xr[i] = Xs[(ty * 4 + i) * 33 + kk]; wr[i] = Ws[(tx * 4 + i) * 33 + kk]; }
            for (int a = 0; a < 4; a++) for (int c = 0; c < 4; c++) acc[a * 4 + c] += xr[a] * wr[c];
        }
        barrier();
    }
    for (int a = 0; a < 4; a++) {
        int m = mt + ty * 4 + a;
        if (m >= pc.M) continue;
        for (int c = 0; c < 4; c++) {
            int n = nt + tx * 4 + c;
            if (n < pc.N) C[pc.coff + m * pc.N + n] = acc[a * 4 + c];
        }
    }
}
"#;
    s
}

/// Tiled prefill GEMM v2: 64x64 C tile, 64 threads, 8x8 micro-tile each;
/// thread t loads token row t and weight row t (one full 32-value sub-block).
fn source_tiled8() -> String {
    let mut s = super::glsl::prelude(64, 3);
    s += r#"
#define T 64
#define LD 33
layout(std430, set = 0, binding = 0) readonly buffer B0 { float X[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { uint W[]; };
layout(std430, set = 0, binding = 2) buffer B2 { float C[]; };
layout(push_constant) uniform PC { int N; int K; int xoff; int coff; int M; int qt; } pc;
layout(local_size_x = 64) in;
shared float Ws[T * LD];
shared float Xs[T * LD];

uint wword(uint off) {
    uint i = off >> 2u;
    uint s = (off & 3u) * 8u;
    return s == 0u ? W[i] : ((W[i] >> s) | (W[i + 1u] << (32u - s)));
}
uint bsel(uint w0, uint w1, uint w2, int i) {
    uint w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (w >> uint(8 * (i & 3))) & 0xFFu;
}

void main() {
    int nt = int(gl_WorkGroupID.x) * T;
    int mt = int(gl_WorkGroupID.y) * T;
    int t = int(gl_LocalInvocationID.x);
    int tx = t & 7;
    int ty = t >> 3;
    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 0.0;
    int nb = pc.K / 256;
    int nsub = pc.K / 32;
    int m = mt + t;
    int n = nt + t;
    for (int sb = 0; sb < nsub; sb++) {
        int b = sb >> 3;
        int j = sb & 7;
        for (int i = 0; i < 32; i++) {
            Xs[t * LD + i] = m < pc.M ? X[pc.xoff + m * pc.K + sb * 32 + i] : 0.0;
        }
        if (n < pc.N) {
            if (pc.qt == 12) {
                uint wb = (uint(n) * uint(nb) + uint(b)) * 36u;
                vec2 dm = unpackHalf2x16(W[wb]);
                uint s0 = W[wb + 1u]; uint s1 = W[wb + 2u]; uint s2 = W[wb + 3u];
                float sc; float mn;
                if (j < 4) { sc = float(bsel(s0, s1, s2, j) & 63u); mn = float(bsel(s0, s1, s2, j + 4) & 63u); }
                else {
                    sc = float((bsel(s0, s1, s2, j + 4) & 15u) | ((bsel(s0, s1, s2, j - 4) >> 6u) << 4u));
                    mn = float((bsel(s0, s1, s2, j + 4) >> 4u) | ((bsel(s0, s1, s2, j) >> 6u) << 4u));
                }
                float a = dm.x * sc; float c = dm.y * mn;
                uint off = wb + 4u + uint((j >> 1) * 8);
                uint sh_ = (j & 1) != 0 ? 4u : 0u;
                for (int wi = 0; wi < 8; wi++) {
                    uint q = W[off + uint(wi)];
                    for (int i = 0; i < 4; i++) Ws[t * LD + wi * 4 + i] = a * float((q >> (uint(8 * i) + sh_)) & 15u) - c;
                }
            } else {
                uint p = (uint(n) * uint(nb) + uint(b)) * 210u;
                int half_ = j >> 2; int q = j & 3;
                uint dp = p + 208u;
                float d = unpackHalf2x16((W[dp >> 2u] >> ((dp & 3u) * 8u)) & 0xFFFFu).x;
                int sidx = 8 * half_ + 2 * q;
                uint sw = wword(p + 192u + uint(sidx & ~3));
                uint bv0 = (sw >> uint(8 * (sidx & 3))) & 0xFFu;
                uint bv1 = (sw >> uint(8 * ((sidx & 3) + 1))) & 0xFFu;
                float sc0 = d * float(int(bv0) - (bv0 >= 128u ? 256 : 0));
                float sc1 = d * float(int(bv1) - (bv1 >= 128u ? 256 : 0));
                uint lsh = q >= 2 ? 4u : 0u; uint hsh = uint(2 * q);
                for (int wi = 0; wi < 8; wi++) {
                    uint ql = wword(p + uint(64 * half_ + (q & 1) * 32 + wi * 4));
                    uint qh = wword(p + 128u + uint(32 * half_ + wi * 4));
                    float sc = wi < 4 ? sc0 : sc1;
                    for (int i = 0; i < 4; i++) {
                        uint lo = (ql >> (uint(8 * i) + lsh)) & 15u;
                        uint hi = (qh >> (uint(8 * i) + hsh)) & 3u;
                        Ws[t * LD + wi * 4 + i] = sc * float(int(lo | (hi << 4u)) - 32);
                    }
                }
            }
        } else {
            for (int i = 0; i < 32; i++) Ws[t * LD + i] = 0.0;
        }
        barrier();
        for (int kk = 0; kk < 32; kk++) {
            float xr[8]; float wr[8];
            for (int i = 0; i < 8; i++) { xr[i] = Xs[(ty * 8 + i) * LD + kk]; wr[i] = Ws[(tx * 8 + i) * LD + kk]; }
            for (int a = 0; a < 8; a++) for (int c = 0; c < 8; c++) acc[a * 8 + c] += xr[a] * wr[c];
        }
        barrier();
    }
    for (int a = 0; a < 8; a++) {
        int mm = mt + ty * 8 + a;
        if (mm >= pc.M) continue;
        for (int c = 0; c < 8; c++) {
            int nn = nt + tx * 8 + c;
            if (nn < pc.N) C[pc.coff + mm * pc.N + nn] = acc[a * 8 + c];
        }
    }
}
"#;
    s
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    d: &VulkanDevice,
    dtype: GgmlDType,
    x: Buf,
    w: Buf,
    out: Buf,
    m: usize,
    n: usize,
    k: usize,
    xoff: usize,
) -> Result<()> {
    let qt = match dtype {
        GgmlDType::Q4K => 12,
        _ => 14,
    };
    let wg = wg();
    let tiled_min = env_usize("CANDLE_VULKAN_KQ_TILED_MIN", 16);
    if m >= tiled_min {
        let gx = n.div_ceil(64);
        let gy = m.div_ceil(64);
        let push = Push::default().us(n)?.us(k)?.us(xoff)?.i(0).us(m)?.i(qt);
        if std::env::var("CANDLE_VULKAN_KQ_TILE").as_deref() == Ok("4") {
            return d.launch(
                "kq_tiled_64x64",
                |_| source_tiled(),
                &[x, w, out],
                push,
                &[],
                [gx as u32, gy as u32, 1],
            );
        }
        return d.launch(
            "kq_tiled8_64x64",
            |_| source_tiled8(),
            &[x, w, out],
            push,
            &[],
            [gx as u32, gy as u32, 1],
        );
    }
    if m > 1 {
        let nr = env_usize("CANDLE_VULKAN_KQ_MM_NR", 2);
        let mt = env_usize("CANDLE_VULKAN_KQ_MM_MT", 8);
        let lim = d.limits();
        let groups = n.div_ceil(nr);
        let gx = groups.min(lim.max_groups_x as usize).max(1);
        let gy = groups.div_ceil(gx);
        let push = Push::default().us(n)?.us(k)?.us(xoff)?.i(0).us(m)?.i(qt);
        let key = format!("kq_mm_{wg}_{nr}_{mt}");
        return d.launch(
            &key,
            |_| source_mm(wg, nr, mt),
            &[x, w, out],
            push,
            &[],
            [gx as u32, gy as u32, m.div_ceil(mt) as u32],
        );
    }
    let nr = nr();
    let lim = d.limits();
    let groups = n.div_ceil(nr);
    let gx = groups.min(lim.max_groups_x as usize).max(1);
    let gy = groups.div_ceil(gx);
    let push = Push::default().us(n)?.us(k)?.us(xoff)?.i(0).us(m)?.i(qt);
    let key = format!("kq_gemv_{wg}_{nr}");
    d.launch(
        &key,
        |_| source(wg, nr),
        &[x, w, out],
        push,
        &[],
        [gx as u32, gy as u32, m as u32],
    )
}
