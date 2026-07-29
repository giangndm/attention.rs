#![cfg(all(feature = "cuda", feature = "flashinfer"))]

use attention_rs::flashinfer::{
    append_kv_cache, decode_plan, decode_with_plan_into, decode_with_plan_shared,
    FlashInferDecodeOutput,
};
use attention_rs::{FlashInferMetadata, InputMetadata, PagedAttention};
use candle_core::{DType, Device, Result, Tensor};
use half::bf16;
use std::sync::Arc;

const PAGE_SIZE: usize = 32;
const HEAD_DIM: usize = 128;
const TARGET_Q_HEADS: usize = 64;
const TARGET_KV_HEADS: usize = 8;

struct DecodeFixture {
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    indices: Tensor,
    indptr: Tensor,
    last_len: Tensor,
    plan: Arc<Vec<i64>>,
}

impl DecodeFixture {
    fn new(device: &Device, query_scale: f32) -> Result<Self> {
        let query = Tensor::from_vec(
            (0..HEAD_DIM)
                .map(|index| bf16::from_f32(query_scale * (index + 1) as f32 / HEAD_DIM as f32))
                .collect::<Vec<_>>(),
            (1, 1, HEAD_DIM),
            device,
        )?;
        let key_cache = Tensor::from_vec(
            (0..PAGE_SIZE * HEAD_DIM)
                .map(|index| bf16::from_f32((index % HEAD_DIM + 1) as f32 / HEAD_DIM as f32))
                .collect::<Vec<_>>(),
            (1, PAGE_SIZE, 1, HEAD_DIM),
            device,
        )?;
        let value_cache = Tensor::from_vec(
            (0..PAGE_SIZE * HEAD_DIM)
                .map(|index| bf16::from_f32((index + 1) as f32 / 64.0))
                .collect::<Vec<_>>(),
            (1, PAGE_SIZE, 1, HEAD_DIM),
            device,
        )?;
        let indices = Tensor::from_vec(vec![u32::MAX, 0], 2, device)?.narrow(0, 1, 1)?;
        let indptr = Tensor::from_vec(vec![0_u32, 1], 2, device)?;
        let last_len = Tensor::from_vec(vec![17_u32], 1, device)?;
        let plan = Arc::new(decode_plan(
            device,
            DType::BF16,
            DType::BF16,
            &[0, 1],
            Some(&[17]),
            Some(&[17]),
            1,
            1,
            1,
            HEAD_DIM,
            PAGE_SIZE,
            false,
        )?);
        Ok(Self {
            query,
            key_cache,
            value_cache,
            indices,
            indptr,
            last_len,
            plan,
        })
    }

    fn allocating(&self) -> Result<Tensor> {
        decode_with_plan_shared(
            &self.query,
            &self.key_cache,
            &self.value_cache,
            None,
            None,
            &self.indices,
            &self.indptr,
            &self.last_len,
            PAGE_SIZE,
            1,
            1,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
            self.plan.clone(),
            false,
            None,
            None,
        )
    }

    fn write(&self, output: &mut FlashInferDecodeOutput) -> Result<()> {
        decode_with_plan_into(
            output,
            &self.query,
            &self.key_cache,
            &self.value_cache,
            None,
            None,
            &self.indices,
            &self.indptr,
            &self.last_len,
            PAGE_SIZE,
            1,
            1,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
            self.plan.clone(),
            false,
            None,
            None,
        )
    }
}

struct TargetFp8Fixture {
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    indices: Tensor,
    indptr: Tensor,
    last_len: Tensor,
    plan: Arc<Vec<i64>>,
}

impl TargetFp8Fixture {
    fn new(device: &Device, query_scale: f32, context_len: usize) -> Result<Self> {
        let query = Tensor::from_vec(
            (0..TARGET_Q_HEADS * HEAD_DIM)
                .map(|index| {
                    bf16::from_f32(query_scale * (index % HEAD_DIM + 1) as f32 / HEAD_DIM as f32)
                })
                .collect::<Vec<_>>(),
            (1, TARGET_Q_HEADS, HEAD_DIM),
            device,
        )?;
        let key_cache =
            Tensor::zeros((2, PAGE_SIZE, TARGET_KV_HEADS, HEAD_DIM), DType::U8, device)?;
        let value_cache =
            Tensor::zeros((2, PAGE_SIZE, TARGET_KV_HEADS, HEAD_DIM), DType::U8, device)?;
        let k_scale = Tensor::ones(TARGET_KV_HEADS, DType::F32, device)?;
        let v_scale = Tensor::ones(TARGET_KV_HEADS, DType::F32, device)?;
        let indices = Tensor::from_vec(vec![u32::MAX, 1], 2, device)?.narrow(0, 1, 1)?;
        let indptr = Tensor::from_vec(vec![0_u32, 1], 2, device)?;
        let last_len = Tensor::from_vec(vec![context_len as u32], 1, device)?;
        populate_fp8_cache(
            device,
            &key_cache,
            &value_cache,
            &k_scale,
            &v_scale,
            &indices,
            &indptr,
            context_len,
        )?;
        let plan = Arc::new(decode_plan(
            device,
            DType::U8,
            DType::BF16,
            &[0, 1],
            Some(&[context_len as u32]),
            Some(&[context_len as u32]),
            1,
            TARGET_Q_HEADS,
            TARGET_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            false,
        )?);
        Ok(Self {
            query,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            indices,
            indptr,
            last_len,
            plan,
        })
    }

    fn decode(&self, output: Option<&mut FlashInferDecodeOutput>) -> Result<Tensor> {
        if let Some(output) = output {
            decode_with_plan_into(
                output,
                &self.query,
                &self.key_cache,
                &self.value_cache,
                Some(&self.k_scale),
                Some(&self.v_scale),
                &self.indices,
                &self.indptr,
                &self.last_len,
                PAGE_SIZE,
                TARGET_Q_HEADS,
                TARGET_KV_HEADS,
                HEAD_DIM,
                1.0 / (HEAD_DIM as f32).sqrt(),
                self.plan.clone(),
                false,
                None,
                None,
            )?;
            return Ok(output.as_tensor()?.clone());
        }
        decode_with_plan_shared(
            &self.query,
            &self.key_cache,
            &self.value_cache,
            Some(&self.k_scale),
            Some(&self.v_scale),
            &self.indices,
            &self.indptr,
            &self.last_len,
            PAGE_SIZE,
            TARGET_Q_HEADS,
            TARGET_KV_HEADS,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
            self.plan.clone(),
            false,
            None,
            None,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn populate_fp8_cache(
    device: &Device,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: &Tensor,
    v_scale: &Tensor,
    indices: &Tensor,
    indptr: &Tensor,
    context_len: usize,
) -> Result<()> {
    if context_len == 0 {
        return Ok(());
    }
    let key = Tensor::from_vec(
        (0..context_len * TARGET_KV_HEADS * HEAD_DIM)
            .map(|index| bf16::from_f32((index % HEAD_DIM + 1) as f32 / HEAD_DIM as f32))
            .collect::<Vec<_>>(),
        (context_len, TARGET_KV_HEADS, HEAD_DIM),
        device,
    )?;
    let value = Tensor::from_vec(
        (0..context_len * TARGET_KV_HEADS * HEAD_DIM)
            .map(|index| bf16::from_f32((index % 251 + 1) as f32 / 32.0))
            .collect::<Vec<_>>(),
        (context_len, TARGET_KV_HEADS, HEAD_DIM),
        device,
    )?;
    let last_len = Tensor::from_vec(vec![context_len as u32], 1, device)?;
    let batch_indices = Tensor::zeros(context_len, DType::U32, device)?;
    let positions = Tensor::from_vec(
        (0..context_len as u32).collect::<Vec<_>>(),
        context_len,
        device,
    )?;
    append_kv_cache(
        &key,
        &value,
        key_cache,
        value_cache,
        Some(k_scale),
        Some(v_scale),
        indices,
        indptr,
        &last_len,
        Some(&batch_indices),
        Some(&positions),
    )
}

#[test]
fn new_decode_output_is_not_observable_before_a_successful_write() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let output = FlashInferDecodeOutput::new(&device, DType::BF16, 1, 1, 128)?;

    let error = output
        .as_tensor()
        .expect_err("uninitialized decode output must remain inaccessible");
    assert!(error.to_string().contains("not initialized"));
    Ok(())
}

#[test]
fn caller_owned_output_matches_allocating_decode_with_offset_indices_and_partial_page() -> Result<()>
{
    let device = Device::new_cuda(0)?;
    let fixture = DecodeFixture::new(&device, 1.0)?;
    let expected = fixture.allocating()?;
    let mut output = FlashInferDecodeOutput::new(&device, DType::BF16, 1, 1, HEAD_DIM)?;

    fixture.write(&mut output)?;
    device.synchronize()?;

    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = output
        .as_tensor()?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 0.02,
            "actual={actual} expected={expected}"
        );
    }
    Ok(())
}

#[test]
fn sequential_writes_preserve_immediately_enqueued_consumers() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let first = DecodeFixture::new(&device, 1.0)?;
    let second = DecodeFixture::new(&device, 0.5)?;
    let first_expected = first.allocating()?.affine(2.0, 1.0)?;
    let second_expected = second.allocating()?.affine(3.0, -1.0)?;
    let mut output = FlashInferDecodeOutput::new(&device, DType::BF16, 1, 1, HEAD_DIM)?;

    first.write(&mut output)?;
    let first_consumed = output.as_tensor()?.affine(2.0, 1.0)?;
    second.write(&mut output)?;
    let second_consumed = output.as_tensor()?.affine(3.0, -1.0)?;
    device.synchronize()?;

    let first_delta = first_consumed
        .sub(&first_expected)?
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    let second_delta = second_consumed
        .sub(&second_expected)?
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        first_delta <= 0.02,
        "first immediate consumer delta={first_delta}"
    );
    assert!(
        second_delta <= 0.02,
        "second immediate consumer delta={second_delta}"
    );
    Ok(())
}

#[test]
fn decode_into_rejects_cuda_graph_without_exposing_output() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let fixture = DecodeFixture::new(&device, 1.0)?;
    let mut output = FlashInferDecodeOutput::new(&device, DType::BF16, 1, 1, HEAD_DIM)?;

    let error = decode_with_plan_into(
        &mut output,
        &fixture.query,
        &fixture.key_cache,
        &fixture.value_cache,
        None,
        None,
        &fixture.indices,
        &fixture.indptr,
        &fixture.last_len,
        PAGE_SIZE,
        1,
        1,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        fixture.plan,
        true,
        None,
        None,
    )
    .expect_err("caller-owned decode output must reject CUDA Graph mode");
    assert!(error.to_string().contains("CUDA Graph"));
    assert!(output.as_tensor().is_err());
    Ok(())
}

#[test]
fn failed_attempt_invalidates_a_previously_initialized_output() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let fixture = DecodeFixture::new(&device, 1.0)?;
    let mut output = FlashInferDecodeOutput::new(&device, DType::BF16, 1, 1, HEAD_DIM)?;
    fixture.write(&mut output)?;
    assert!(output.as_tensor().is_ok());

    let error = decode_with_plan_into(
        &mut output,
        &fixture.query,
        &fixture.key_cache,
        &fixture.value_cache,
        None,
        None,
        &fixture.indices,
        &fixture.indptr,
        &fixture.last_len,
        PAGE_SIZE,
        1,
        1,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        fixture.plan,
        true,
        None,
        None,
    )
    .expect_err("second graph-mode attempt must fail");
    assert!(error.to_string().contains("CUDA Graph"));
    assert!(output.as_tensor().is_err());
    Ok(())
}

#[test]
fn paged_attention_exposes_a_caller_owned_decode_entry_point() {
    let _method = PagedAttention::forward_flashinfer_decode_into;
}

#[test]
fn target_fp8_decode_reuses_output_for_sequential_immediate_consumers() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let first = TargetFp8Fixture::new(&device, 1.0, 17)?;
    let second = TargetFp8Fixture::new(&device, 0.5, 17)?;
    let first_expected = first.decode(None)?.affine(2.0, 1.0)?;
    let second_expected = second.decode(None)?.affine(3.0, -1.0)?;
    let mut output =
        FlashInferDecodeOutput::new(&device, DType::BF16, 1, TARGET_Q_HEADS, HEAD_DIM)?;

    let first_consumed = first.decode(Some(&mut output))?.affine(2.0, 1.0)?;
    let second_consumed = second.decode(Some(&mut output))?.affine(3.0, -1.0)?;
    device.synchronize()?;

    let first_delta = first_consumed
        .sub(&first_expected)?
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    let second_delta = second_consumed
        .sub(&second_expected)?
        .abs()?
        .to_dtype(DType::F32)?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        first_delta <= 0.02,
        "first FP8 consumer delta={first_delta}"
    );
    assert!(
        second_delta <= 0.02,
        "second FP8 consumer delta={second_delta}"
    );
    assert!(output
        .as_tensor()?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn paged_attention_runs_target_fp8_cache_write_and_decode_into() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let fixture = TargetFp8Fixture::new(&device, 1.0, 16)?;
    let context_len = 17_u32;
    let last_len = Tensor::from_vec(vec![context_len], 1, &device)?;
    let plan = Arc::new(decode_plan(
        &device,
        DType::U8,
        DType::BF16,
        &[0, 1],
        Some(&[context_len]),
        Some(&[context_len]),
        1,
        TARGET_Q_HEADS,
        TARGET_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        false,
    )?);
    let mut metadata = InputMetadata {
        is_prefill: false,
        is_mla: false,
        sequence_ids: None,
        mamba_slot_mapping: None,
        slot_mapping: Tensor::from_vec(vec![48_i64], 1, &device)?,
        block_tables: Some(fixture.indices.reshape((1, 1))?),
        context_lens: Some(Tensor::from_vec(vec![context_len], 1, &device)?),
        cu_seqlens_q: None,
        cu_seqlens_k: None,
        max_seqlen_q: 1,
        max_seqlen_k: context_len as usize,
        max_context_len: context_len as usize,
        seqlens: Some(vec![1]),
        flashinfer_metadata: Some(FlashInferMetadata {
            indptr: fixture.indptr.clone(),
            indptr_host: Arc::new(vec![0, 1]),
            indices: fixture.indices.clone(),
            last_len,
            last_len_host: Some(Arc::new(vec![context_len])),
            kv_len_arr_host: Some(Arc::new(vec![context_len])),
            total_num_rows: Some(1),
            window_left: -1,
            batch_indices: None,
            positions: None,
            use_cuda_graph: true,
            decode_plan_info: Some(plan),
            prefill_plan_info: None,
            mla_decode_plan_info: None,
            mla_prefill_plan_info: None,
        }),
        is_mtp_verify: false,
    };
    let key = Tensor::from_vec(
        vec![bf16::from_f32(0.5); TARGET_KV_HEADS * HEAD_DIM],
        (1, TARGET_KV_HEADS, HEAD_DIM),
        &device,
    )?;
    let value = Tensor::from_vec(
        vec![bf16::from_f32(2.0); TARGET_KV_HEADS * HEAD_DIM],
        (1, TARGET_KV_HEADS, HEAD_DIM),
        &device,
    )?;
    let attention = PagedAttention::new(
        TARGET_Q_HEADS,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        Some(TARGET_KV_HEADS),
        None,
        device.clone(),
        None,
        true,
    )?;
    let mut output =
        FlashInferDecodeOutput::new(&device, DType::BF16, 1, TARGET_Q_HEADS, HEAD_DIM)?;

    let key_before = fixture.key_cache.flatten_all()?.to_vec1::<u8>()?;
    let value_before = fixture.value_cache.flatten_all()?.to_vec1::<u8>()?;
    let graph_error = attention
        .forward_flashinfer_decode_into(
            &mut output,
            &fixture.query,
            &key,
            &value,
            None,
            Some(fixture.key_cache.clone()),
            Some(fixture.value_cache.clone()),
            &metadata,
            None,
        )
        .expect_err("PagedAttention caller-owned output must reject graph mode before cache write");
    assert!(graph_error.to_string().contains("CUDA Graph"));
    assert!(output.as_tensor().is_err());
    assert_eq!(
        fixture.key_cache.flatten_all()?.to_vec1::<u8>()?,
        key_before
    );
    assert_eq!(
        fixture.value_cache.flatten_all()?.to_vec1::<u8>()?,
        value_before
    );
    metadata
        .flashinfer_metadata
        .as_mut()
        .expect("fixture has FlashInfer metadata")
        .use_cuda_graph = false;

    attention.forward_flashinfer_decode_into(
        &mut output,
        &fixture.query,
        &key,
        &value,
        None,
        Some(fixture.key_cache.clone()),
        Some(fixture.value_cache.clone()),
        &metadata,
        None,
    )?;
    device.synchronize()?;

    assert!(output
        .as_tensor()?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|value| value.is_finite()));
    let written_value = fixture
        .value_cache
        .narrow(0, 1, 1)?
        .narrow(1, 16, 1)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    assert!(written_value.iter().any(|value| *value != 0));
    Ok(())
}
