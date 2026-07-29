#![cfg(all(feature = "cuda", feature = "flash", feature = "flashinfer"))]

use attention_rs::{flashinfer::prefill_plan, FlashInferMetadata, InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor};
use half::bf16;
use std::sync::Arc;

const BLOCKS: usize = 4;
const PAGE_SIZE: usize = 32;
const HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const QUERY_LEN: usize = 3;
const WINDOW_LEFT: i32 = -1;

#[test]
fn flashinfer_metadata_clones_share_host_vectors_and_plans() -> Result<()> {
    let device = Device::Cpu;
    let indptr_host = Arc::new(vec![0_u32, 1]);
    let last_len_host = Arc::new(vec![1_u32]);
    let kv_len_arr_host = Arc::new(vec![1_u32]);
    let decode_plan_info = Arc::new(vec![0_i64; 10]);
    let metadata = FlashInferMetadata {
        indptr: Tensor::zeros(2, DType::U32, &device)?,
        indptr_host: indptr_host.clone(),
        indices: Tensor::zeros(1, DType::U32, &device)?,
        last_len: Tensor::zeros(1, DType::U32, &device)?,
        last_len_host: Some(last_len_host.clone()),
        kv_len_arr_host: Some(kv_len_arr_host.clone()),
        total_num_rows: Some(1),
        window_left: WINDOW_LEFT,
        batch_indices: None,
        positions: None,
        use_cuda_graph: false,
        decode_plan_info: Some(decode_plan_info.clone()),
        prefill_plan_info: None,
        mla_decode_plan_info: None,
        mla_prefill_plan_info: None,
    };

    let clones = (0..64)
        .map(|_| {
            (
                metadata.indptr_host.clone(),
                metadata
                    .last_len_host
                    .as_ref()
                    .expect("fixture has last lengths")
                    .clone(),
                metadata
                    .kv_len_arr_host
                    .as_ref()
                    .expect("fixture has KV lengths")
                    .clone(),
                metadata
                    .decode_plan_info
                    .as_ref()
                    .expect("fixture has decode plan")
                    .clone(),
            )
        })
        .collect::<Vec<_>>();

    for (indptr, last_len, kv_len, plan) in &clones {
        assert!(Arc::ptr_eq(indptr, &indptr_host));
        assert!(Arc::ptr_eq(last_len, &last_len_host));
        assert!(Arc::ptr_eq(kv_len, &kv_len_arr_host));
        assert!(Arc::ptr_eq(plan, &decode_plan_info));
    }
    assert_eq!(Arc::strong_count(&indptr_host), 66);
    assert_eq!(Arc::strong_count(&last_len_host), 66);
    assert_eq!(Arc::strong_count(&kv_len_arr_host), 66);
    assert_eq!(Arc::strong_count(&decode_plan_info), 66);
    Ok(())
}

struct Run {
    key_bytes: Vec<u8>,
    value_bytes: Vec<u8>,
    output: Vec<f32>,
}

fn zero_cache(device: &Device) -> Result<Tensor> {
    Tensor::zeros((BLOCKS, PAGE_SIZE, HEADS, HEAD_DIM), DType::U8, device)
}

fn qkv(device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    let query = (0..QUERY_LEN * HEAD_DIM)
        .map(|index| bf16::from_f32(((index % HEAD_DIM) as f32 + 1.0) / HEAD_DIM as f32))
        .collect::<Vec<_>>();
    let key = (0..QUERY_LEN)
        .flat_map(|token| std::iter::repeat_n(bf16::from_f32((token + 1) as f32), HEAD_DIM))
        .collect::<Vec<_>>();
    let value = (0..QUERY_LEN * HEAD_DIM)
        .map(|index| bf16::from_f32((index + 1) as f32 / 32.0))
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_vec(query, (QUERY_LEN, HEADS, HEAD_DIM), device)?,
        Tensor::from_vec(key, (QUERY_LEN, HEADS, HEAD_DIM), device)?,
        Tensor::from_vec(value, (QUERY_LEN, HEADS, HEAD_DIM), device)?,
    ))
}

fn metadata(
    device: &Device,
    slots: &[i64],
    context_len: u32,
    flashinfer: bool,
) -> Result<InputMetadata> {
    let query_len = slots.len();
    let indices_host = vec![1_u32, 3];
    let indptr_host = vec![0_u32, 2];
    let last_len_host = vec![(context_len - 1) % PAGE_SIZE as u32 + 1];
    let qo_indptr_host = vec![0_u32, query_len as u32];
    let flashinfer_metadata = if flashinfer {
        Some(FlashInferMetadata {
            indptr: Tensor::from_vec(indptr_host.clone(), 2, device)?,
            indptr_host: Arc::new(indptr_host.clone()),
            indices: Tensor::from_vec(indices_host.clone(), 2, device)?,
            last_len: Tensor::from_vec(last_len_host.clone(), 1, device)?,
            last_len_host: Some(Arc::new(last_len_host)),
            kv_len_arr_host: Some(Arc::new(vec![context_len])),
            total_num_rows: Some(query_len as u32),
            window_left: WINDOW_LEFT,
            batch_indices: None,
            positions: None,
            use_cuda_graph: false,
            decode_plan_info: None,
            prefill_plan_info: Some(Arc::new(prefill_plan(
                device,
                &qo_indptr_host,
                &indptr_host,
                &[context_len],
                query_len as u32,
                1,
                HEADS,
                HEADS,
                HEAD_DIM,
                PAGE_SIZE,
                DType::BF16,
                Some(WINDOW_LEFT),
                Some(DType::U8),
                false,
            )?)),
            mla_decode_plan_info: None,
            mla_prefill_plan_info: None,
        })
    } else {
        None
    };

    Ok(InputMetadata {
        is_prefill: true,
        is_mla: false,
        sequence_ids: None,
        mamba_slot_mapping: None,
        slot_mapping: Tensor::from_vec(slots.to_vec(), slots.len(), device)?,
        block_tables: Some(Tensor::from_vec(indices_host, (1, 2), device)?),
        context_lens: Some(Tensor::from_vec(vec![context_len], 1, device)?),
        cu_seqlens_q: Some(Tensor::from_vec(qo_indptr_host, 2, device)?),
        cu_seqlens_k: Some(Tensor::from_vec(vec![0_u32, context_len], 2, device)?),
        max_seqlen_q: query_len,
        max_seqlen_k: context_len as usize,
        max_context_len: context_len as usize,
        seqlens: Some(vec![query_len as u32]),
        flashinfer_metadata,
        is_mtp_verify: false,
    })
}

fn forward(
    attention: &PagedAttention,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    metadata: &InputMetadata,
) -> Result<Tensor> {
    attention.forward(
        query,
        key,
        value,
        None,
        Some(key_cache.clone()),
        Some(value_cache.clone()),
        metadata,
        None,
    )
}

fn run_one_shot(device: &Device, flashinfer: bool) -> Result<Run> {
    let (query, key, value) = qkv(device)?;
    let key_cache = zero_cache(device)?;
    let value_cache = zero_cache(device)?;
    let attention = PagedAttention::new(
        HEADS,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Some(HEADS),
        None,
        device.clone(),
        None,
        true,
    )?;
    let output = forward(
        &attention,
        &query,
        &key,
        &value,
        &key_cache,
        &value_cache,
        &metadata(device, &[63, 96, 97], 34, flashinfer)?,
    )?;
    device.synchronize()?;
    Ok(Run {
        key_bytes: key_cache.flatten_all()?.to_vec1()?,
        value_bytes: value_cache.flatten_all()?.to_vec1()?,
        output: output.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
    })
}

fn run_chunked(device: &Device) -> Result<Run> {
    let (query, key, value) = qkv(device)?;
    let key_cache = zero_cache(device)?;
    let value_cache = zero_cache(device)?;
    let attention = PagedAttention::new(
        HEADS,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Some(HEADS),
        None,
        device.clone(),
        None,
        true,
    )?;
    let first = forward(
        &attention,
        &query.narrow(0, 0, 2)?,
        &key.narrow(0, 0, 2)?,
        &value.narrow(0, 0, 2)?,
        &key_cache,
        &value_cache,
        &metadata(device, &[63, 96], 33, true)?,
    )?;
    let second = forward(
        &attention,
        &query.narrow(0, 2, 1)?,
        &key.narrow(0, 2, 1)?,
        &value.narrow(0, 2, 1)?,
        &key_cache,
        &value_cache,
        &metadata(device, &[97], 34, true)?,
    )?;
    let output = Tensor::cat(&[&first, &second], 0)?;
    device.synchronize()?;
    Ok(Run {
        key_bytes: key_cache.flatten_all()?.to_vec1()?,
        value_bytes: value_cache.flatten_all()?.to_vec1()?,
        output: output.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
    })
}

fn changed_slots(bytes: &[u8]) -> Vec<usize> {
    bytes
        .chunks_exact(HEAD_DIM)
        .enumerate()
        .filter_map(|(slot, values)| values.iter().any(|value| *value != 0).then_some(slot))
        .collect()
}

fn assert_output_parity(reference: &[f32], candidate: &[f32]) {
    assert_eq!(candidate.len(), reference.len());
    assert!(reference.iter().all(|value| value.is_finite()));
    assert!(candidate.iter().all(|value| value.is_finite()));
    for (index, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
        let tolerance = 0.05 + 0.05 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "output mismatch at {index}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
    for (expected, actual) in reference
        .chunks_exact(HEAD_DIM)
        .zip(candidate.chunks_exact(HEAD_DIM))
    {
        let expected_argmax = expected
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index);
        let actual_argmax = actual
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index);
        assert_eq!(
            actual_argmax, expected_argmax,
            "greedy argmax must be exact"
        );
    }
}

#[test]
fn flashinfer_eligible_path_matches_native_slot_mapped_cache_bytes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let native = run_one_shot(&device, false)?;
    let optimized = run_one_shot(&device, true)?;
    let expected_slots = vec![63, 96, 97];

    assert_eq!(changed_slots(&native.key_bytes), expected_slots);
    assert_eq!(changed_slots(&native.value_bytes), expected_slots);
    assert_eq!(
        changed_slots(&optimized.key_bytes),
        expected_slots,
        "FlashInfer-eligible PagedAttention path omitted key slots"
    );
    assert_eq!(
        changed_slots(&optimized.value_bytes),
        expected_slots,
        "FlashInfer-eligible PagedAttention path omitted value slots"
    );
    assert_eq!(
        optimized.key_bytes, native.key_bytes,
        "key cache must match exact native bytes"
    );
    assert_eq!(
        optimized.value_bytes, native.value_bytes,
        "value cache must match exact native bytes"
    );
    Ok(())
}

#[test]
fn flashinfer_attention_matches_native_finite_output_and_argmax() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let native = run_one_shot(&device, false)?;
    let optimized = run_one_shot(&device, true)?;
    assert_output_parity(&native.output, &optimized.output);
    Ok(())
}

#[test]
fn flashinfer_multi_chunk_matches_one_shot_cache_and_output() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let one_shot = run_one_shot(&device, true)?;
    let chunked = run_chunked(&device)?;
    assert_eq!(
        chunked.key_bytes, one_shot.key_bytes,
        "multi-chunk key cache must match exact one-shot bytes"
    );
    assert_eq!(
        chunked.value_bytes, one_shot.value_bytes,
        "multi-chunk value cache must match exact one-shot bytes"
    );
    assert_output_parity(&one_shot.output, &chunked.output);
    Ok(())
}
