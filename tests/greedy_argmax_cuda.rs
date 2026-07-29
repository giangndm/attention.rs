#![cfg(feature = "cuda")]

use attention_rs::sampler::Sampler;
use candle_core::{DType, Device, Result, Tensor, D};

const VOCAB: usize = 151_936;

fn cuda() -> Result<Option<Device>> {
    if candle_core::utils::cuda_is_available() {
        Ok(Some(Device::new_cuda(0)?))
    } else {
        Ok(None)
    }
}

fn logits_with_peaks(batch: usize, peaks: &[&[(usize, f32)]]) -> Vec<f32> {
    let mut values = vec![f32::NEG_INFINITY; batch * VOCAB];
    for (row, row_peaks) in peaks.iter().enumerate() {
        for &(index, value) in *row_peaks {
            values[row * VOCAB + index] = value;
        }
    }
    values
}

#[test]
fn greedy_matches_candle_tie_order_and_infinities() -> Result<()> {
    let Some(device) = cuda()? else { return Ok(()) };
    let sampler = Sampler::new();

    let cases: &[(&[(usize, f32)], u32)] = &[
        (&[(41, 7.0), (42, 7.0)], 42),
        (&[(11, 5.0), (1036, 5.0)], 1036),
        (&[(1, 6.0), (1024, 6.0)], 1024),
        (&[(1, 8.0), (1025, 8.0)], 1),
        (&[(9, f32::INFINITY), (1032, f32::INFINITY)], 1032),
    ];

    for &(peaks, expected) in cases {
        let logits = Tensor::from_vec(logits_with_peaks(1, &[peaks]), (1, VOCAB), &device)?;
        let candle = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        assert_eq!(sampler.greedy_cuda(&logits)?, candle);
        assert_eq!(candle, vec![expected]);
    }

    let all_negative_infinity =
        Tensor::from_vec(vec![f32::NEG_INFINITY; VOCAB], (1, VOCAB), &device)?;
    let candle = all_negative_infinity.argmax(D::Minus1)?.to_vec1::<u32>()?;
    assert_eq!(sampler.greedy_cuda(&all_negative_infinity)?, candle);
    assert_eq!(candle, vec![0]);
    Ok(())
}

#[test]
fn greedy_handles_random_rows_batches_and_row_permutations() -> Result<()> {
    let Some(device) = cuda()? else { return Ok(()) };
    let sampler = Sampler::new();
    let mut rows = vec![vec![0f32; VOCAB]; 3];
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for row in &mut rows {
        for value in row {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *value = ((state >> 40) as i32 - (1 << 23)) as f32;
        }
    }
    rows[0][17] = 20_000_000.0;
    rows[1][88_888] = 21_000_000.0;
    rows[2][151_935] = 22_000_000.0;

    for batch in 1..=3 {
        let values = rows[..batch].concat();
        let logits = Tensor::from_vec(values, (batch, VOCAB), &device)?;
        let candle = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        assert_eq!(sampler.greedy_cuda(&logits)?, candle);
        assert_eq!(candle, [17, 88_888, 151_935][..batch]);
    }

    let permuted = Tensor::from_vec(
        [rows[2].as_slice(), rows[0].as_slice(), rows[1].as_slice()].concat(),
        (3, VOCAB),
        &device,
    )?;
    let candle = permuted.argmax(D::Minus1)?.to_vec1::<u32>()?;
    assert_eq!(sampler.greedy_cuda(&permuted)?, candle);
    assert_eq!(candle, vec![151_935, 17, 88_888]);
    Ok(())
}

#[test]
fn greedy_honors_contiguous_layout_start_offset() -> Result<()> {
    let Some(device) = cuda()? else { return Ok(()) };
    let sampler = Sampler::new();
    let values = logits_with_peaks(2, &[&[(3, 10.0)], &[(123_456, 11.0)]]);
    let backing = Tensor::from_vec(values, (2, VOCAB), &device)?;
    let offset_view = backing.narrow(0, 1, 1)?;

    assert!(offset_view.is_contiguous());
    let candle = offset_view.argmax(D::Minus1)?.to_vec1::<u32>()?;
    assert_eq!(sampler.greedy_cuda(&offset_view)?, candle);
    assert_eq!(candle, vec![123_456]);
    Ok(())
}

#[test]
fn greedy_rejects_ineligible_tensors() -> Result<()> {
    let Some(device) = cuda()? else { return Ok(()) };
    let sampler = Sampler::new();

    let rank_one = Tensor::zeros(VOCAB, DType::F32, &device)?;
    assert!(sampler.greedy_cuda(&rank_one).is_err());

    let rank_three = Tensor::zeros((1, 1, VOCAB), DType::F32, &device)?;
    assert!(sampler.greedy_cuda(&rank_three).is_err());

    let wrong_width = Tensor::zeros((1, VOCAB - 1), DType::F32, &device)?;
    assert!(sampler.greedy_cuda(&wrong_width).is_err());

    let empty_vocab = Tensor::zeros((1, 0), DType::F32, &device)?;
    assert!(sampler.greedy_cuda(&empty_vocab).is_err());

    let empty_batch = Tensor::zeros((0, VOCAB), DType::F32, &device)?;
    assert!(sampler.greedy_cuda(&empty_batch).is_err());

    let wrong_dtype = Tensor::zeros((1, VOCAB), DType::F16, &device)?;
    assert!(sampler.greedy_cuda(&wrong_dtype).is_err());

    let non_contiguous = Tensor::zeros((VOCAB, 2), DType::F32, &device)?.t()?;
    assert_eq!(non_contiguous.dims(), &[2, VOCAB]);
    assert!(!non_contiguous.is_contiguous());
    assert!(sampler.greedy_cuda(&non_contiguous).is_err());

    let cpu = Tensor::zeros((1, VOCAB), DType::F32, &Device::Cpu)?;
    assert!(sampler.greedy_cuda(&cpu).is_err());
    Ok(())
}
