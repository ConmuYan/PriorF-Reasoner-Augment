from __future__ import annotations

import math

import pytest
import torch

from llm.hidden_state_pooling import pool_last_valid_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_left_pad(hidden_vals, pad_lens, dtype=torch.float32, device="cpu"):
    """Construct a left-padded batch from per-row valid token values.

    Args:
        hidden_vals: list of ``torch.Tensor`` of shape ``[valid_len, H]``.
        pad_lens: list of ints, minimum pad length per row.
    Returns:
        (hidden_states [B, T, H], attention_mask [B, T])
    """
    H = hidden_vals[0].shape[1]
    rows = []
    masks = []
    max_len = max(v.shape[0] + p for v, p in zip(hidden_vals, pad_lens))
    for v, p in zip(hidden_vals, pad_lens):
        valid_len = v.shape[0]
        total_pad = max_len - valid_len
        left_pad = torch.zeros(total_pad, H, dtype=dtype, device=device)
        row = torch.cat([left_pad, v.to(dtype).to(device)], dim=0)
        mask = torch.cat(
            [
                torch.zeros(total_pad, dtype=torch.long, device=device),
                torch.ones(valid_len, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        rows.append(row)
        masks.append(mask)
    return torch.stack(rows, dim=0), torch.stack(masks, dim=0)


def _make_right_pad(hidden_vals, pad_lens, dtype=torch.float32, device="cpu"):
    """Construct a right-padded batch from per-row valid token values."""
    H = hidden_vals[0].shape[1]
    rows = []
    masks = []
    max_len = max(v.shape[0] + p for v, p in zip(hidden_vals, pad_lens))
    for v, p in zip(hidden_vals, pad_lens):
        valid_len = v.shape[0]
        total_pad = max_len - valid_len
        right_pad = torch.zeros(total_pad, H, dtype=dtype, device=device)
        row = torch.cat([v.to(dtype).to(device), right_pad], dim=0)
        mask = torch.cat(
            [
                torch.ones(valid_len, dtype=torch.long, device=device),
                torch.zeros(total_pad, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        rows.append(row)
        masks.append(mask)
    return torch.stack(rows, dim=0), torch.stack(masks, dim=0)


# ---------------------------------------------------------------------------
# 1. Single sequence, no pad, arbitrary T
# ---------------------------------------------------------------------------
def test_single_no_pad():
    T, H = 5, 4
    hs = torch.randn(1, T, H)
    mask = torch.ones(1, T, dtype=torch.long)
    out = pool_last_valid_token(hs, mask)
    assert out.shape == (1, H)
    assert torch.equal(out[0], hs[0, -1, :])


# ---------------------------------------------------------------------------
# 2. Single sequence, T=1, mask=[[1]]
# ---------------------------------------------------------------------------
def test_single_t1():
    hs = torch.randn(1, 1, 3)
    mask = torch.ones(1, 1, dtype=torch.long)
    out = pool_last_valid_token(hs, mask)
    assert out.shape == (1, 3)
    assert torch.equal(out[0], hs[0, 0, :])


# ---------------------------------------------------------------------------
# 3. Multi batch, mixed left-pad lengths
# ---------------------------------------------------------------------------
def test_mixed_left_pad():
    H = 3
    v0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    v1 = torch.tensor([[10.0, 11.0, 12.0]])
    hs, mask = _make_left_pad([v0, v1], [2, 0])
    out = pool_last_valid_token(hs, mask)
    assert out.shape == (2, H)
    assert torch.equal(out[0], v0[-1])
    assert torch.equal(out[1], v1[-1])


# ---------------------------------------------------------------------------
# 4. Multi batch, mixed right-pad lengths
# ---------------------------------------------------------------------------
def test_mixed_right_pad():
    H = 3
    v0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v1 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
    hs, mask = _make_right_pad([v0, v1], [1, 2])
    out = pool_last_valid_token(hs, mask)
    assert out.shape == (2, H)
    assert torch.equal(out[0], v0[-1])
    assert torch.equal(out[1], v1[-1])


# ---------------------------------------------------------------------------
# 5. No-pad batch (mask all 1)
# ---------------------------------------------------------------------------
def test_batch_all_ones():
    B, T, H = 3, 7, 5
    hs = torch.randn(B, T, H)
    mask = torch.ones(B, T, dtype=torch.long)
    out = pool_last_valid_token(hs, mask)
    assert out.shape == (B, H)
    assert torch.equal(out, hs[:, -1, :])


# ---------------------------------------------------------------------------
# 6. Padding-side invariance (core defence)
# ---------------------------------------------------------------------------
def test_padding_side_invariance():
    H = 4
    v0 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    v1 = torch.tensor([[9.0, 10.0, 11.0, 12.0]])

    # Common sequence length for both padding sides.
    max_len = 6

    # --- Left-pad batch: valid tokens at the END, pad at the START ---
    hs_l = torch.randn(2, max_len, H)
    # Row 0: valid_len=2, left-pad = max_len - 2 = 4
    hs_l[0, 4:4 + v0.shape[0]] = v0
    # Row 1: valid_len=1, left-pad = max_len - 1 = 5
    hs_l[1, 5:5 + v1.shape[0]] = v1
    mask_l = torch.zeros(2, max_len, dtype=torch.long)
    mask_l[0, 4:4 + v0.shape[0]] = 1
    mask_l[1, 5:5 + v1.shape[0]] = 1

    # --- Right-pad batch: valid tokens at the START, pad at the END ---
    hs_r = torch.randn(2, max_len, H)
    hs_r[0, 0:v0.shape[0]] = v0
    hs_r[1, 0:v1.shape[0]] = v1
    mask_r = torch.zeros(2, max_len, dtype=torch.long)
    mask_r[0, 0:v0.shape[0]] = 1
    mask_r[1, 0:v1.shape[0]] = 1

    out_l = pool_last_valid_token(hs_l, mask_l)
    out_r = pool_last_valid_token(hs_r, mask_r)
    assert torch.equal(out_l, out_r)
    assert torch.equal(out_l[0], v0[-1])
    assert torch.equal(out_l[1], v1[-1])


# ---------------------------------------------------------------------------
# 7. All-pad row -> ValueError with row index
# ---------------------------------------------------------------------------
def test_all_pad_row():
    hs = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.long)
    mask[1] = 0
    with pytest.raises(ValueError, match=r"row.*1"):
        pool_last_valid_token(hs, mask)


# ---------------------------------------------------------------------------
# 8. Non-contiguous mask (e.g. [1,0,1,1]) -> ValueError with row index
# ---------------------------------------------------------------------------
def test_non_contiguous_mask():
    hs = torch.randn(1, 4, 3)
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"row.*0"):
        pool_last_valid_token(hs, mask)


# ---------------------------------------------------------------------------
# 9. Mask contains values outside {0,1} -> ValueError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_mask", [
    torch.tensor([[0, 2, 1, 0]], dtype=torch.long),
    torch.tensor([[0.0, 0.5, 1.0, 0.0]], dtype=torch.float32),
])
def test_mask_values_outside_zero_one(bad_mask):
    hs = torch.randn(1, 4, 3)
    with pytest.raises(ValueError):
        pool_last_valid_token(hs, bad_mask)


# ---------------------------------------------------------------------------
# 10. hidden_states non-floating-point -> TypeError
# ---------------------------------------------------------------------------
def test_hidden_states_non_float():
    hs = torch.randint(0, 10, (1, 3, 4), dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.long)
    with pytest.raises(TypeError):
        pool_last_valid_token(hs, mask)


# ---------------------------------------------------------------------------
# 11. Wrong dimensionality -> ValueError
# ---------------------------------------------------------------------------
def test_wrong_dims():
    hs_2d = torch.randn(3, 4)
    mask_1d = torch.ones(4, dtype=torch.long)
    hs_ok = torch.randn(1, 4, 3)
    mask_ok = torch.ones(1, 4, dtype=torch.long)

    with pytest.raises(ValueError):
        pool_last_valid_token(hs_2d, mask_ok)
    with pytest.raises(ValueError):
        pool_last_valid_token(hs_ok, mask_1d)


# ---------------------------------------------------------------------------
# 12. Batch or seq dim mismatch -> ValueError
# ---------------------------------------------------------------------------
def test_shape_mismatch():
    hs = torch.randn(2, 5, 3)
    mask = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        pool_last_valid_token(hs, mask)


# ---------------------------------------------------------------------------
# 13. Device mismatch -> ValueError
# ---------------------------------------------------------------------------
def test_device_mismatch():
    # Only test if CPU vs meta is possible without a real GPU
    hs = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.long)
    # Move mask to meta device
    mask_meta = mask.to("meta")
    with pytest.raises(ValueError, match=r"Device mismatch"):
        pool_last_valid_token(hs, mask_meta)


# ---------------------------------------------------------------------------
# 14. NaN / Inf in hidden_states -> ValueError
# ---------------------------------------------------------------------------
def test_nan_inf():
    hs = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.long)

    hs_nan = hs.clone()
    hs_nan[0, 1, 2] = float("nan")
    with pytest.raises(ValueError, match=r"NaN"):
        pool_last_valid_token(hs_nan, mask)

    hs_inf = hs.clone()
    hs_inf[0, 1, 2] = float("inf")
    with pytest.raises(ValueError, match=r"Inf"):
        pool_last_valid_token(hs_inf, mask)

    hs_ninf = hs.clone()
    hs_ninf[0, 1, 2] = float("-inf")
    with pytest.raises(ValueError, match=r"Inf"):
        pool_last_valid_token(hs_ninf, mask)


# ---------------------------------------------------------------------------
# 15. dtype / device fidelity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_fidelity(dtype):
    # Skip bfloat16 if unsupported on this platform
    try:
        hs = torch.randn(2, 4, 3, dtype=dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} not supported on this platform")
    mask = torch.ones(2, 4, dtype=torch.long)
    out = pool_last_valid_token(hs, mask)
    assert out.dtype == dtype
    assert out.device == hs.device


# ---------------------------------------------------------------------------
# 16. Autograd preserved
# ---------------------------------------------------------------------------
def test_autograd():
    hs = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.long)
    out = pool_last_valid_token(hs, mask)
    out.sum().backward()
    assert hs.grad is not None
    assert hs.grad.shape == hs.shape

    # Selected positions should have non-zero gradients
    # Since out = hs[b, last_idx, :], grad at selected positions should be 1.0
    # because sum() derivative is 1 everywhere.
    for b in range(2):
        last_idx = (mask[b] == 1).nonzero(as_tuple=False).max().item()
        assert torch.all(hs.grad[b, last_idx, :] == 1.0)
        # Other positions should be 0
        if last_idx > 0:
            assert torch.all(hs.grad[b, :last_idx, :] == 0.0)
        if last_idx < 4:
            assert torch.all(hs.grad[b, last_idx + 1 :, :] == 0.0)


# ---------------------------------------------------------------------------
# 17. Determinism
# ---------------------------------------------------------------------------
def test_determinism():
    hs = torch.randn(2, 5, 3)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.long)
    out1 = pool_last_valid_token(hs, mask)
    out2 = pool_last_valid_token(hs, mask)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# 18. Module exports only pool_last_valid_token
# ---------------------------------------------------------------------------
def test_module_exports():
    import llm.hidden_state_pooling as m

    assert m.__all__ == ("pool_last_valid_token",)
    assert hasattr(m, "pool_last_valid_token")
    assert not hasattr(m, "pool_mean_token")
    assert not hasattr(m, "pool_first_token")
    assert not hasattr(m, "pool_cls_token")
    assert not hasattr(m, "pool_max_token")
    assert not hasattr(m, "mean_pooling")
    assert not hasattr(m, "max_pooling")
