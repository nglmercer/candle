//! Quick Vulkan performance probe (experiments only, not a test).
//!
//! cargo run --release -p candle-core --features vulkan --example vulkan_probe -- model.gguf
//!
//! - decode: one GEMV (m=1) over every quantized 2D weight = the matmul work of one token
//! - prefill: the same with m=64 rows
//! - overhead: N tiny dependent elementwise dispatches
use candle_core::quantized::{gguf_file, GgmlDType, QMatMul};
use candle_core::{Device, Module, Result, Tensor};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: vulkan_probe model.gguf");
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let dev = Device::new_vulkan(0)?;
    let mut f = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut f)?;
    let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
    names.sort();
    let mut mats = Vec::new();
    let mut bytes = 0usize;
    let mut by_type: HashMap<String, (usize, usize)> = HashMap::new();
    for name in names {
        let info = &content.tensor_infos[&name];
        if info.shape.rank() != 2 || matches!(info.ggml_dtype, GgmlDType::F32) {
            continue;
        }
        let q = content.tensor(&mut f, &name, &dev)?;
        let (n, k) = q.shape().dims2()?;
        bytes += q.storage_size_in_bytes();
        let e = by_type.entry(format!("{:?}", q.dtype())).or_default();
        e.0 += 1;
        e.1 += q.storage_size_in_bytes();
        mats.push((name, n, k, QMatMul::from_qtensor(q)?));
    }
    println!(
        "{} quantized matrices, {:.1} MiB; by type: {:?}",
        mats.len(),
        bytes as f64 / 1048576.0,
        by_type
            .iter()
            .map(|(t, (c, b))| format!("{t}: {c} ({:.0} MiB)", *b as f64 / 1048576.0))
            .collect::<Vec<_>>()
    );

    // quick sanity: GPU vs CPU on one matrix per quant type (m=1 and m=4)
    let mut seen = std::collections::HashSet::new();
    for (name, _, k, _) in &mats {
        let info = &content.tensor_infos[name];
        if !seen.insert(format!("{:?}", info.ggml_dtype)) {
            continue;
        }
        let qc = content.tensor(&mut f, name, &Device::Cpu)?;
        let qg = content.tensor(&mut f, name, &dev)?;
        let (wc, wg) = (QMatMul::from_qtensor(qc)?, QMatMul::from_qtensor(qg)?);
        for m in [1usize, 4, 64] {
            let x = Tensor::randn(0f32, 1.0, (m, *k), &Device::Cpu)?;
            let yc = wc.forward(&x)?;
            let yg = wg.forward(&x.to_device(&dev)?)?.to_device(&Device::Cpu)?;
            let err = (&yc - &yg)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            let scale = yc.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            println!(
                "check {:?} {name} m={m}: max abs err {err:.3e} (rel {:.2e})",
                info.ggml_dtype,
                err / scale
            );
        }
    }

    let pms: Vec<usize> = std::env::var("PM")
        .unwrap_or_else(|_| "64".into())
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    for m in std::iter::once(1usize).chain(pms) {
        let mut xs = HashMap::new();
        for (_, _, k, _) in &mats {
            xs.entry(*k)
                .or_insert(Tensor::randn(0f32, 1.0, (m, *k), &dev)?);
        }
        // warmup (pipeline compilation)
        for (_, _, k, w) in &mats {
            w.forward(&xs[k])?;
        }
        dev.synchronize()?;
        // ramp GPU clocks (power management is on `auto`) before timing
        let warm = Instant::now();
        while warm.elapsed().as_secs_f64() < 2.0 {
            for (_, _, k, w) in &mats {
                w.forward(&xs[k])?;
            }
            dev.synchronize()?;
        }
        candle_core::vulkan_backend::reset_stats();
        let t0 = Instant::now();
        for _ in 0..iters {
            for (_, _, k, w) in &mats {
                w.forward(&xs[k])?;
            }
            dev.synchronize()?;
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let st = candle_core::vulkan_backend::stats();
        println!(
            "{:8} m={m:<3} {:7.2} ms/pass  {:6.1} GB/s weights  {:7.1} pass/s  (dispatches/pass {}, gpu wait {:.1} ms/pass)",
            if m == 1 { "decode" } else { "prefill" },
            dt * 1e3,
            bytes as f64 / dt / 1e9,
            1.0 / dt,
            st.native_dispatches / iters,
            st.gpu_wait_ns as f64 / 1e6 / iters as f64
        );
        if std::env::var("PER_MATRIX").is_ok() && m == 1 {
            let mut rows = Vec::new();
            for (name, n, k, w) in &mats {
                let x = &xs[k];
                w.forward(x)?;
                dev.synchronize()?;
                let t = Instant::now();
                for _ in 0..iters {
                    w.forward(x)?;
                }
                dev.synchronize()?;
                rows.push((
                    t.elapsed().as_secs_f64() / iters as f64,
                    name.clone(),
                    *n,
                    *k,
                ));
            }
            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (t, name, n, k) in rows.iter().take(8) {
                println!("   {:7.3} ms  {name} [{n}x{k}]", t * 1e3);
            }
        }
    }

    // raw read bandwidth roof: each invocation sums uvec4s with a grid stride
    {
        use candle_core::backend::BackendStorage;
        use candle_core::vulkan_backend::{glsl, kernels};
        let words = 128usize << 20; // 512 MiB of u32
        let big = Tensor::zeros(words, candle_core::DType::U32, &dev)?;
        let out = Tensor::zeros(1 << 20, candle_core::DType::U32, &dev)?;
        let vk = dev.as_vulkan_device()?;
        let (bs, _) = big.storage_and_layout();
        let (os, _) = out.storage_and_layout();
        let (candle_core::Storage::Vulkan(bs), candle_core::Storage::Vulkan(os)) = (&*bs, &*os)
        else {
            unreachable!()
        };
        for wg in [64usize, 256] {
            let groups = (1usize << 20) / wg;
            let src = move |_| {
                let mut s = glsl::prelude(wg, 2);
                s += "layout(std430, set=0, binding=0) readonly buffer B0 { uvec4 A[]; };\n\
                      layout(std430, set=0, binding=1) buffer B1 { uint O[]; };\n\
                      layout(push_constant) uniform PC { int n4; } pc;\n\
                      layout(local_size_x = WG) in;\n\
                      void main() { int i = gid(); int stride = int(gl_NumWorkGroups.x) * WG; uvec4 acc = uvec4(0u);\n\
                      for (int j = i; j < pc.n4; j += stride) acc ^= A[j];\n\
                      O[i] = acc.x ^ acc.y ^ acc.z ^ acc.w; }\n";
                s
            };
            let launch = || {
                vk.launch(
                    &format!("probe_bw_{wg}"),
                    src,
                    &[bs.buf(), os.buf()],
                    kernels::Push::default().us(words / 4).unwrap(),
                    &[],
                    [groups as u32, 1, 1],
                )
            };
            launch()?;
            dev.synchronize()?;
            let t0 = Instant::now();
            for _ in 0..20 {
                launch()?;
            }
            dev.synchronize()?;
            let dt = t0.elapsed().as_secs_f64() / 20.0;
            println!(
                "bandwidth roof (wg {wg}): {:6.1} GB/s",
                (words * 4) as f64 / dt / 1e9
            );
        }
    }

    // dispatch overhead: dependent tiny ops
    let n_ops = 2000;
    let mut a = Tensor::zeros(2048, candle_core::DType::F32, &dev)?;
    let b = Tensor::ones(2048, candle_core::DType::F32, &dev)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_ops {
        a = (a + &b)?;
    }
    let rec = t0.elapsed().as_secs_f64();
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "host recording {:5.1} us/op (alloc + descriptors + record)",
        rec / n_ops as f64 * 1e6
    );
    println!(
        "overhead {:7.1} us/dispatch (tiny dependent add, {n_ops} ops)",
        dt / n_ops as f64 * 1e6
    );
    Ok(())
}
