#pragma once
// At-load requantization of Q8_0 weight tensors to Q6_K, so the bandwidth-bound decode GEMVs
// (attn/GDN projections, the #1 decode kernel) read ~23% fewer weight bytes (1.0625 -> 0.8203 B/w)
// via the existing Q6_K dp4a matvec. Q6_K is near-lossless, so this is gated on the accuracy check.
// The output blocks are the standard llama.cpp Q6_K layout (ql[128], qh[64], scales[16], d[2] = 210 B)
// that si_vec_dot_q6_K reads. Faithful port of ggml quantize_row_q6_K_ref + make_qx_quants (rmse_type=1).
#include <cstdint>
#include <cstring>
#include <cmath>

namespace sparkinfer {

inline float rq_h2f(uint16_t h) {
    _Float16 v; std::memcpy(&v, &h, 2); return (float)v;
}
inline uint16_t rq_f2h(float f) {
    _Float16 v = (_Float16)f; uint16_t h; std::memcpy(&h, &v, 2); return h;
}
inline int rq_nint(float x) { return (int)lrintf(x); }

// Best per-group scale (ggml make_qx_quants, rmse_type=1: weight = x^2). n=16, nmax=32 for Q6_K.
inline float rq_make_qx_quants(int n, int nmax, const float* x, int8_t* L) {
    float max = 0.f, amax = 0.f;
    for (int i = 0; i < n; ++i) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < 1e-15f) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = -(float)nmax / max;
    float sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < n; ++i) {
        int l = rq_nint(iscale * x[i]);
        l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
        L[i] = (int8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * (float)l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        float isc = -((float)nmax + 0.1f * is) / max;
        float slx = 0.f, sl2 = 0.f;
        for (int i = 0; i < n; ++i) {
            int l = rq_nint(isc * x[i]);
            l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
            float w = x[i] * x[i];
            slx += w * x[i] * l;
            sl2 += w * (float)l * l;
        }
        if (sl2 > 0.f && slx * slx > best * sl2) {
            for (int i = 0; i < n; ++i) {
                int l = rq_nint(isc * x[i]);
                L[i] = (int8_t)(nmax + (l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l)));
            }
            scale = slx / sl2; best = scale * slx;
        }
    }
    return scale;
}

// src: Q8_0 blocks (34 B: fp16 scale + 32 int8). dst: Q6_K superblocks (210 B). nvals % 256 == 0.
inline void requant_q80_to_q6k(const void* src_, void* dst_, int64_t nvals) {
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_);
    uint8_t* dst = reinterpret_cast<uint8_t*>(dst_);
    const int64_t nsb = nvals / 256;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < nsb; ++s) {
        float x[256];
        for (int b = 0; b < 8; ++b) {
            const uint8_t* blk = src + (size_t)(s * 8 + b) * 34;
            uint16_t dh; std::memcpy(&dh, blk, 2);
            const float d = rq_h2f(dh);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            for (int j = 0; j < 32; ++j) x[b * 32 + j] = d * (float)qs[j];
        }
        uint8_t* y = dst + (size_t)s * 210;
        int8_t L[256]; float scales[16];
        float max_scale = 0.f, max_abs = 0.f;
        for (int ib = 0; ib < 16; ++ib) {
            float sc = rq_make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib);
            scales[ib] = sc; float a = fabsf(sc);
            if (a > max_abs) { max_abs = a; max_scale = sc; }
        }
        if (max_abs < 1e-15f) { std::memset(y, 0, 210); continue; }
        int8_t* qsc = reinterpret_cast<int8_t*>(y + 192);
        const float iscale = -128.f / max_scale;
        const uint16_t dh = rq_f2h(1.f / iscale);
        std::memcpy(y + 208, &dh, 2);
        const float dsb = rq_h2f(dh);
        for (int ib = 0; ib < 16; ++ib) {
            int l = rq_nint(iscale * scales[ib]);
            qsc[ib] = (int8_t)(l < -128 ? -128 : (l > 127 ? 127 : l));
        }
        for (int j = 0; j < 16; ++j) {
            const float dq = dsb * qsc[j];
            if (dq == 0.f) { for (int ii = 0; ii < 16; ++ii) L[16 * j + ii] = 32; continue; }
            for (int ii = 0; ii < 16; ++ii) {
                int l = rq_nint(x[16 * j + ii] / dq);
                l = l < -32 ? -32 : (l > 31 ? 31 : l);
                L[16 * j + ii] = (int8_t)(l + 32);
            }
        }
        uint8_t* ql = y; uint8_t* qh = y + 128;
        std::memset(ql, 0, 128); std::memset(qh, 0, 64);
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF, q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
                ql[l +  0] = q1 | (q3 << 4);
                ql[l + 32] = q2 | (q4 << 4);
                qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2)
                                | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64; qh += 32;
        }
    }
}

} // namespace sparkinfer
