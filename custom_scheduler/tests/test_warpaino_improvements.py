"""Comprehensive unit and functional tests for WarpAINO improvements and bug fixes.

Tests:
1. Low-precision (bfloat16, float16) handling with stochastic rounding and Kahan summation on CUDA.
2. Non-contiguous FP32 parameters (channels_last format) writeback in native and foreach modes.
3. Spectral meta-update mathematical gradients vs Autograd on CUDA.
4. Lazy initialization of warp states when meta_lr changes from 0 to >0.
5. Lazy initialization of CAME confidence buffers without corrupting innovation moments.
6. Foreach vs native step numerical equivalence.
7. Torch compile compatibility (compile_step=True).
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from LoraEasyCustomOptimizer.warpaino import (
        WarpAINO,
        _spectral_meta_left_core,
        _spectral_meta_bilateral_core,
        _spectral_log_gain,
        _spectral_mirror,
        _spectral_expand_rfft_gradient,
    )
except ImportError:
    import importlib.util

    _module_path = os.path.join(
        os.path.dirname(__file__), "..", "LoraEasyCustomOptimizer", "warpaino.py"
    )
    _spec = importlib.util.spec_from_file_location("warpaino_standalone", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    WarpAINO = _module.WarpAINO
    _spectral_meta_left_core = _module._spectral_meta_left_core
    _spectral_meta_bilateral_core = _module._spectral_meta_bilateral_core
    _spectral_log_gain = _module._spectral_log_gain
    _spectral_mirror = _module._spectral_mirror
    _spectral_expand_rfft_gradient = _module._spectral_expand_rfft_gradient

DEVICE = "cuda"


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def test_bfloat16_stochastic_rounding():
    """Verify bfloat16 parameters with stochastic_fp=True step without assertion error."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(32, 32, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    p.grad = torch.randn_like(p)
    p_orig = p.clone()

    opt = WarpAINO([p], lr=0.01, stochastic_fp=True, kahan_sum=False)
    opt.step()

    assert not torch.equal(p, p_orig)
    assert p.dtype == torch.bfloat16
    assert torch.isfinite(p).all()


def test_bfloat16_kahan_sum():
    """Verify bfloat16 parameters with kahan_sum=True maintain error compensation buffer."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(16, 16, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    p.grad = torch.randn_like(p)

    opt = WarpAINO([p], lr=0.01, kahan_sum=True)
    opt.step()

    state = opt.state[p]
    assert "param_compensation" in state
    comp = state["param_compensation"]
    assert comp.dtype == torch.float32
    assert comp.shape == p.shape
    assert torch.isfinite(comp).all()


def test_float16_kahan_sum():
    """Verify float16 parameters step cleanly and maintain FP32 compensation."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(16, 16, device=DEVICE, dtype=torch.float16, requires_grad=True)
    p.grad = torch.randn_like(p)
    p_orig = p.clone()

    opt = WarpAINO([p], lr=0.01, kahan_sum=True)
    opt.step()

    assert not torch.equal(p, p_orig)
    assert p.dtype == torch.float16


@pytest.mark.parametrize("foreach", [False, True])
def test_channels_last_fp32_writeback(foreach):
    """Verify non-contiguous 4D (channels_last) FP32 parameters are updated in both modes."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(8, 8, 3, 3, device=DEVICE, dtype=torch.float32).to(
        memory_format=torch.channels_last
    )
    p.requires_grad = True
    assert not p.is_contiguous()

    p.grad = torch.randn_like(p)
    p_orig = p.clone()

    opt = WarpAINO([p], lr=0.01, foreach=foreach)
    opt.step()

    diff = (p - p_orig).abs().max().item()
    assert diff > 0.0, f"channels_last parameter was silently not updated (diff={diff})"


def test_spectral_meta_left_grad_matches_autograd():
    """Verify unilateral spectral meta gradient exactly matches autograd across all frequencies."""
    _requires_cuda()
    torch.manual_seed(42)
    prev = torch.randn(8, 16, device=DEVICE, dtype=torch.float64)
    current = torch.randn(8, 16, device=DEVICE, dtype=torch.float64)
    log_left = torch.randn(8, device=DEVICE, dtype=torch.float64) * 0.1
    log_left = 0.5 * (log_left + _spectral_mirror(log_left))
    log_left.requires_grad = True

    left_gain = (0.5 * (log_left + _spectral_mirror(log_left))).exp()
    spectrum = torch.fft.fft(prev, dim=0, norm="ortho")
    warped = torch.fft.ifft(spectrum * left_gain[:, None], dim=0, norm="ortho").real
    loss = 0.5 * (warped - current).pow(2).sum() / prev.norm().square()
    loss.backward()

    with torch.no_grad():
        left_gain_f = (
            0.5 * (log_left + _spectral_mirror(log_left))
        )[: log_left.shape[0] // 2 + 1].exp()
        prev_spectrum = torch.fft.rfft(prev, dim=0, norm="ortho")
        current_spectrum = torch.fft.rfft(current, dim=0, norm="ortho")
        residual = left_gain_f[:, None] * prev_spectrum - current_spectrum
        grad_positive = left_gain_f * torch.real(
            torch.conj(residual) * prev_spectrum
        ).sum(dim=1)
        grad_positive = grad_positive / prev.norm().square()
        grad_analytic = _spectral_expand_rfft_gradient(grad_positive, log_left.shape[0])

    max_diff = (grad_analytic - log_left.grad).abs().max().item()
    assert max_diff < 1e-15, f"Unilateral spectral grad mismatch: {max_diff}"


def test_spectral_meta_bilateral_grad_matches_autograd():
    """Verify bilateral spectral meta gradients match autograd across both dimensions."""
    _requires_cuda()
    torch.manual_seed(42)
    prev = torch.randn(8, 16, device=DEVICE, dtype=torch.float64)
    current = torch.randn(8, 16, device=DEVICE, dtype=torch.float64)
    log_left = torch.randn(8, device=DEVICE, dtype=torch.float64) * 0.1
    log_left = 0.5 * (log_left + _spectral_mirror(log_left))
    log_left.requires_grad = True
    log_right = torch.randn(16, device=DEVICE, dtype=torch.float64) * 0.1
    log_right = 0.5 * (log_right + _spectral_mirror(log_right))
    log_right.requires_grad = True

    left_gain = (0.5 * (log_left + _spectral_mirror(log_left))).exp()
    right_gain = (0.5 * (log_right + _spectral_mirror(log_right))).exp()
    spectrum = torch.fft.fft2(prev, dim=(0, 1), norm="ortho")
    gain = left_gain[:, None] * right_gain[None, :]
    warped = torch.fft.ifft2(spectrum * gain, dim=(0, 1), norm="ortho").real
    loss = 0.5 * (warped - current).pow(2).sum() / prev.norm().square()
    loss.backward()

    # Verify updated core implementation without artificial doubling
    updated_left, updated_right = _spectral_meta_bilateral_core(
        log_left.detach().clone(),
        log_right.detach().clone(),
        prev,
        current,
        meta_lr=1.0,
        meta_wd=0.0,
        spectral_log_bound=10.0,
    )
    grad_left_impl = log_left.detach() - updated_left
    grad_right_impl = log_right.detach() - updated_right

    max_diff_left = (grad_left_impl - log_left.grad).abs().max().item()
    max_diff_right = (grad_right_impl - log_right.grad).abs().max().item()
    assert max_diff_left < 1e-15, f"Bilateral left grad mismatch: {max_diff_left}"
    assert max_diff_right < 1e-15, f"Bilateral right grad mismatch: {max_diff_right}"


def test_lazy_init_meta_lr_transition():
    """Verify warp states are cleanly lazily initialized when meta_lr is enabled post step 0."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(16, 16, device=DEVICE, requires_grad=True)
    p.grad = torch.randn_like(p)

    opt = WarpAINO([p], lr=0.01, meta_lr=0.0, warp_mode="spectral")
    opt.step()

    state = opt.state[p]
    assert "spectral_log_left" not in state
    assert "warp" not in state

    # User turns on warp dynamically
    opt.param_groups[0]["meta_lr"] = 0.05
    p.grad = torch.randn_like(p)
    opt.step()

    assert "spectral_log_left" in state
    assert state["spectral_log_left"].shape == (16,)


def test_came_confidence_dynamic_enable():
    """Verify dynamically enabling came_confidence does not corrupt innovation tracking."""
    _requires_cuda()
    torch.manual_seed(42)
    p = torch.randn(16, 16, device=DEVICE, requires_grad=True)
    p.grad = torch.randn_like(p)

    opt = WarpAINO([p], lr=0.01, came_confidence=False)
    opt.step()

    state = opt.state[p]
    sq_row_before = state["exp_avg_sq_row"].clone()
    sq_col_before = state["exp_avg_sq_col"].clone()
    assert "exp_avg_res_row" not in state

    # Dynamically enable came_confidence
    opt.param_groups[0]["came_confidence"] = True
    p.grad = torch.randn_like(p)
    opt.step()

    assert "exp_avg_res_row" in state
    assert "exp_avg_res_col" in state
    assert state["exp_avg_res_row"] is not state["exp_avg_sq_row"]
    assert state["exp_avg_res_col"] is not state["exp_avg_sq_col"]


def test_foreach_vs_native_equivalence():
    """Verify foreach and native step paths produce identical updates."""
    _requires_cuda()
    torch.manual_seed(42)
    p_native_2d = torch.randn(16, 16, device=DEVICE, requires_grad=True)
    p_native_1d = torch.randn(16, device=DEVICE, requires_grad=True)
    p_foreach_2d = p_native_2d.clone().detach().requires_grad_(True)
    p_foreach_1d = p_native_1d.clone().detach().requires_grad_(True)

    g_2d = torch.randn_like(p_native_2d)
    g_1d = torch.randn_like(p_native_1d)

    p_native_2d.grad = g_2d.clone()
    p_native_1d.grad = g_1d.clone()
    p_foreach_2d.grad = g_2d.clone()
    p_foreach_1d.grad = g_1d.clone()

    opt_native = WarpAINO([p_native_2d, p_native_1d], lr=0.01, foreach=False)
    opt_foreach = WarpAINO([p_foreach_2d, p_foreach_1d], lr=0.01, foreach=True)

    opt_native.step()
    opt_foreach.step()

    torch.testing.assert_close(p_native_2d, p_foreach_2d)
    torch.testing.assert_close(p_native_1d, p_foreach_1d)


def test_compile_step_execution():
    """Verify compile_step=True runs without errors on 2D and 1D parameters."""
    _requires_cuda()
    torch.manual_seed(42)
    p2d = torch.randn(16, 16, device=DEVICE, requires_grad=True)
    p1d = torch.randn(16, device=DEVICE, requires_grad=True)
    p2d.grad = torch.randn_like(p2d)
    p1d.grad = torch.randn_like(p1d)

    opt = WarpAINO([p2d, p1d], lr=0.01, compile_step=True)
    opt.step()

    assert torch.isfinite(p2d).all()
    assert torch.isfinite(p1d).all()
