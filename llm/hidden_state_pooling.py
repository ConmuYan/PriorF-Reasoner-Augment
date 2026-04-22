from __future__ import annotations

__all__ = ("pool_last_valid_token",)


def pool_last_valid_token(
    hidden_states: "torch.Tensor",
    attention_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Extract the hidden state of the last valid token per batch row.

    For each batch row ``b``, this function finds the greatest absolute position
    ``t*`` where ``attention_mask[b, t*] == 1`` and returns
    ``hidden_states[b, t*, :]``.  The result shape is ``[B, H]``.

    The algorithm does **not** depend on ``padding_side``.  It infers the last
    valid token purely from the binary ``attention_mask`` using absolute
    position indexing.

    Fail-closed validation order (each guard halts execution before any
    computation if violated):

    1. ``hidden_states`` and ``attention_mask`` must be ``torch.Tensor``
       instances.
    2. ``hidden_states`` must be 3-D ``[B, T, H]`` and ``attention_mask``
       must be 2-D ``[B, T]``.
    3. The first two dimensions must match exactly:
       ``hidden_states.shape[:2] == attention_mask.shape``.
    4. ``B >= 1``, ``T >= 1``, ``H >= 1``.
    5. ``hidden_states.device == attention_mask.device``.
    6. ``hidden_states.dtype`` must be a floating-point dtype.
    7. ``hidden_states`` must not contain ``NaN`` or ``Inf``.
    8. ``attention_mask`` values must be a strict subset of ``{0, 1}``.
       ``torch.bool`` and integer dtypes are accepted and treated as-is.
       Float dtypes are accepted only if every element is exactly ``0.0``
       or ``1.0``.
    9. Each row of ``attention_mask`` must be in contiguous form:
       ``0^a 1^b`` (left-pad, ``a >= 0, b >= 1``),
       ``1^b 0^a`` (right-pad, ``a >= 0, b >= 1``),
       or all ``1``s.  If any row violates this, a ``ValueError`` is raised
       containing the violating row index.
    10. No row may be all-pad (sum ``== 0``).  If found, a ``ValueError`` is
        raised containing the violating row index(s).

    Absolute position indexing (Path A):

    .. code-block:: python

        positions = torch.arange(T, device=hidden_states.device,
                                 dtype=torch.long).unsqueeze(0).expand(B, T)
        masked_positions = positions.masked_fill(attention_mask == 0, -1)
        last_idx = masked_positions.argmax(dim=1)
        pooled = hidden_states.gather(
            1, last_idx.view(B, 1, 1).expand(-1, 1, H)
        ).squeeze(1)

    ``argmax`` on ``masked_positions`` returns the **first** index of the
    maximum value.  Because valid positions are strictly increasing
    ``0, 1, ..., T-1`` and invalid positions are filled with ``-1``, the
    maximum value in any valid row is the largest valid position itself,
    which occurs exactly once at the **last** valid token.  Therefore
    ``argmax`` yields the last valid token index regardless of whether the
    row is left-padded, right-padded, or unpadded.

    Autograd is preserved: the returned tensor retains the computation
    graph of ``hidden_states``.

    Args:
        hidden_states: ``[B, T, H]`` float tensor.
        attention_mask: ``[B, T]`` binary tensor (bool, integer, or exact
            float ``0.0``/``1.0``).

    Returns:
        ``torch.Tensor`` of shape ``[B, H]``, dtype matching
        ``hidden_states.dtype``, device matching ``hidden_states.device``,
        with autograd attached.
    """
    import torch

    # 1. Type check
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("hidden_states must be a torch.Tensor")
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a torch.Tensor")

    # 2. Dimension check
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be 3-D [B, T, H], got dim={hidden_states.dim()}"
        )
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2-D [B, T], got dim={attention_mask.dim()}"
        )

    B, T, H = hidden_states.shape

    # 3. Shape alignment
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"Shape mismatch: hidden_states[:2] {tuple(hidden_states.shape[:2])} "
            f"vs attention_mask {tuple(attention_mask.shape)}"
        )

    # 4. Non-empty dimensions
    if B < 1 or T < 1 or H < 1:
        raise ValueError(
            f"All dimensions must be >= 1, got B={B}, T={T}, H={H}"
        )

    # 5. Device consistency
    if hidden_states.device != attention_mask.device:
        raise ValueError(
            f"Device mismatch: hidden_states on {hidden_states.device}, "
            f"attention_mask on {attention_mask.device}"
        )

    # 6. Floating-point dtype
    if not hidden_states.dtype.is_floating_point:
        raise TypeError(
            f"hidden_states must have a floating-point dtype, got {hidden_states.dtype}"
        )

    # 7. NaN / Inf check
    if torch.isnan(hidden_states).any():
        raise ValueError("hidden_states contains NaN")
    if torch.isinf(hidden_states).any():
        raise ValueError("hidden_states contains Inf")

    # 8. Value domain check: strict subset of {0, 1}
    # Normalize to a boolean-compatible comparison without bool conversion.
    if attention_mask.dtype == torch.bool:
        # bool is already a valid binary mask
        pass
    elif attention_mask.dtype.is_floating_point:
        if not torch.all(
            (attention_mask == 0.0) | (attention_mask == 1.0)
        ):
            raise ValueError(
                "attention_mask values must be exactly 0.0 or 1.0 for float dtype"
            )
    else:
        # Integer types
        if not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError(
                "attention_mask values must be 0 or 1"
            )

    # 9. Contiguous mask pattern per row
    # Normalize to long for diff computation.
    mask_long = attention_mask.to(torch.long)
    for b in range(B):
        row = mask_long[b]
        diffs = row[1:] - row[:-1]
        # Valid contiguous patterns have at most one transition:
        # 0->1 (left pad) or 1->0 (right pad) or none (all 1s or all 0s).
        # All 0s is handled by guard 10.
        transitions = torch.count_nonzero(diffs != 0).item()
        if transitions > 1:
            raise ValueError(
                f"attention_mask row {b} is not contiguous: "
                f"found {transitions} transitions"
            )
        # Additionally guard against the illegal "1s, 0s, 1s" pattern that
        # transitions==1 cannot catch when the first/last values are both 1.
        # transitions==1 with first==1 and last==1 means [1,0...,1] which is illegal.
        # But the diff-based check above correctly counts this as 2 transitions.
        # Wait: [1,0,1] -> diffs = [-1, 1], transitions = 2. Correct.
        # [0,1,0] -> diffs = [1, -1], transitions = 2. Correct.
        # [0,0,1,1,0,0] -> diffs = [0,1,0,-1,0], nonzeros at idx 1 and 3 -> 2 transitions. Correct.
        # [1,1,0,0] -> diffs = [0,-1,0], transitions = 1. Correct.
        # [0,0,1,1] -> diffs = [0,1,0], transitions = 1. Correct.
        # So transitions <= 1 is necessary and sufficient.
        if transitions > 1:
            raise ValueError(
                f"attention_mask row {b} is not contiguous"
            )
        # Also, if transitions == 1, verify direction is valid (0->1 or 1->0)
        if transitions == 1:
            first_val = row[0].item()
            if first_val not in (0, 1):
                # Should not reach here because of guard 8
                raise ValueError(f"attention_mask row {b} has invalid start value")

    # 10. No all-pad rows
    per_row_sum = mask_long.sum(dim=1)
    all_pad_mask = per_row_sum == 0
    if all_pad_mask.any():
        bad_indices = torch.nonzero(all_pad_mask, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"attention_mask rows with all zeros (all-pad): {bad_indices}"
        )

    # --- Canonical absolute-position indexing (Path A) ---
    positions = torch.arange(
        T, device=hidden_states.device, dtype=torch.long
    ).unsqueeze(0).expand(B, T)
    # Build a binary bool mask for positions where attention_mask == 1
    # attention_mask may be bool/int/float; compare against 0 safely.
    mask_eq_1 = attention_mask == 1
    if attention_mask.dtype.is_floating_point:
        mask_eq_1 = attention_mask == 1.0
    masked_positions = positions.masked_fill(~mask_eq_1, -1)
    last_idx = masked_positions.argmax(dim=1)

    # Gather
    last_idx_unsqueezed = last_idx.view(B, 1, 1).expand(-1, 1, H)
    pooled = hidden_states.gather(1, last_idx_unsqueezed).squeeze(1)

    return pooled
