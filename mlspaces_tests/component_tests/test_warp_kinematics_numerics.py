"""Asset-free regression tests pinning the warp IK numerics and array-transfer contract.

Background
----------
warp 1.15.0 added an element-size equality check to ``wp.copy``
(``warp/_src/context.py``: ``if type_size_in_bytes(src.dtype) != type_size_in_bytes(dest.dtype):
raise RuntimeError("Incompatible array data types")``). warp <= 1.14 only compared *total*
byte counts.

``wp.from_numpy(arr)`` without an explicit ``dtype`` infers a **vector** dtype for a 2D array:
``(B, N) float32`` becomes ``B`` elements of ``vec{N}f`` (element size ``4*N`` bytes), not
``B*N`` float32 elements. Copying that into a ``wp.array2d(dtype=float32)`` used to work by
accident and now raises. These tests lock in the explicit-dtype call convention and the
numerical behaviour of the solver kernels so a future warp bump cannot silently reopen either.

Everything here runs without robot meshes, the asset cache, or a GPU.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from molmo_spaces.kinematics.parallel import warp_kinematics as wk

_WARP_DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

_SOURCE = Path(wk.__file__).read_text()


# --- Wrapper kernels exercising the repo's own @wp.func / @wp.kernel objects ---


@wp.kernel
def _solve_batch(
    H: wp.array(dtype=wk.mat66f),
    b: wp.array(dtype=wk.vec6f),
    out: wp.array(dtype=wk.vec6f),
):
    i = wp.tid()
    out[i] = wk.cholesky_solve6(H[i], b[i])


def _run_solve(H32: np.ndarray, b32: np.ndarray, device: str) -> np.ndarray:
    n = H32.shape[0]
    Hw = wp.array(H32, dtype=wk.mat66f, device=device)
    bw = wp.array(b32, dtype=wk.vec6f, device=device)
    ow = wp.zeros(n, dtype=wk.vec6f, device=device)
    wp.launch(_solve_batch, dim=n, inputs=[Hw, bw], outputs=[ow], device=device)
    wp.synchronize()
    return ow.numpy()


# --- cholesky_solve6 ---


@pytest.mark.parametrize("device", _WARP_DEVICES)
def test_cholesky_solve6_exact_on_power_of_two_diagonal(device):
    """Analytic, bit-exact pin: for a diagonal H with power-of-two entries the float32
    Cholesky solve is exactly representable, so the result must match bit-for-bit on
    every device and every warp version. Any change to the accumulator type or to what
    the kernel actually computes shows up here immediately."""
    diag = np.array([4.0, 16.0, 0.25, 1.0, 64.0, 0.0625], dtype=np.float32)
    H = np.diag(diag)[None].astype(np.float32)
    b = np.array([[1.0, -2.0, 3.0, 0.5, -8.0, 0.125]], dtype=np.float32)

    x = _run_solve(H, b, device)

    np.testing.assert_array_equal(x, b / diag)


@pytest.mark.parametrize("device", _WARP_DEVICES)
def test_cholesky_solve6_matches_numpy_well_conditioned(device):
    """Well-conditioned SPD systems must be solved to float32 accuracy.

    Measured margin: max relative error is ~4e-7 (pure float32 round-off); the 1e-5
    threshold is ~25x looser, so it tolerates FMA contraction differences between CPU
    and CUDA but not an actual numerical regression."""
    rng = np.random.default_rng(1234)
    n = 512
    A = rng.normal(size=(n, 6, 6))
    H64 = np.einsum("nab,ncb->nac", A, A) + 6.0 * np.eye(6)[None]
    b64 = rng.normal(size=(n, 6))
    x_ref = np.linalg.solve(H64, b64[..., None])[..., 0]

    x = _run_solve(H64.astype(np.float32), b64.astype(np.float32), device)

    assert np.isfinite(x).all()
    rel = np.linalg.norm(x - x_ref, axis=1) / np.linalg.norm(x_ref, axis=1)
    assert rel.max() < 1e-5, f"max relative error {rel.max():.3e}"


@pytest.mark.parametrize("device", _WARP_DEVICES)
def test_cholesky_solve6_residual_in_production_regime(device):
    """H = J J^T + damping*I with the solver's default damping (1e-12) and a 6x7 arm
    Jacobian. This is near-singular by construction, so the solution itself is not
    well determined -- but the *residual* ||Hx - b|| must stay small and finite."""
    rng = np.random.default_rng(7)
    n = 512
    J = rng.normal(size=(n, 6, 7))
    H64 = np.einsum("nak,nbk->nab", J, J) + 1e-12 * np.eye(6)[None]
    b64 = rng.normal(0.0, 0.1, size=(n, 6))

    x = _run_solve(H64.astype(np.float32), b64.astype(np.float32), device)

    assert np.isfinite(x).all()
    resid = np.einsum("nab,nb->na", H64, x.astype(np.float64)) - b64
    assert np.abs(resid).max() < 1e-2, f"max residual {np.abs(resid).max():.3e}"


# --- lm_step ---


@pytest.mark.parametrize("device", _WARP_DEVICES)
def test_lm_step_matches_numpy(device):
    """One Levenberg-Marquardt step must reproduce q_dot = J^T (J J^T + lambda I)^-1 e
    and qpos += q_dot * dt."""
    rng = np.random.default_rng(99)
    n, nv, damping, dt = 256, 7, 1e-3, 0.5

    J = rng.normal(size=(n, 6, nv)).astype(np.float32)
    jacp = np.ascontiguousarray(J[:, :3, :])
    jacr = np.ascontiguousarray(J[:, 3:, :])
    pos_err = rng.normal(0.0, 0.05, size=(n, 3)).astype(np.float32)
    rot_err = rng.normal(0.0, 0.05, size=(n, 3)).astype(np.float32)
    q0 = rng.normal(0.0, 0.3, size=(n, nv)).astype(np.float32)

    qpos = wp.array(q0.copy(), dtype=wp.float32, device=device)
    q_dot = wp.zeros((n, nv), dtype=wp.float32, device=device)
    wp.launch(
        wk.lm_step,
        dim=n,
        inputs=[
            wp.array(jacp, dtype=wp.float32, device=device),
            wp.array(jacr, dtype=wp.float32, device=device),
            wp.array(pos_err, dtype=wp.vec3f, device=device),
            wp.array(rot_err, dtype=wp.vec3f, device=device),
            wp.array(np.array([damping], dtype=np.float32), dtype=wp.float32, device=device),
            wp.array(np.array([dt], dtype=np.float32), dtype=wp.float32, device=device),
            nv,
        ],
        outputs=[qpos, q_dot],
        device=device,
    )
    wp.synchronize()
    q_dot_out, qpos_out = q_dot.numpy(), qpos.numpy()

    J64 = J.astype(np.float64)
    err64 = np.concatenate([pos_err, rot_err], axis=1).astype(np.float64)
    H64 = np.einsum("nak,nbk->nab", J64, J64) + damping * np.eye(6)[None]
    x64 = np.linalg.solve(H64, err64[..., None])[..., 0]
    q_dot_ref = np.einsum("nak,na->nk", J64, x64)

    assert np.isfinite(q_dot_out).all()
    np.testing.assert_allclose(q_dot_out, q_dot_ref, atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(
        qpos_out, q0.astype(np.float64) + q_dot_out.astype(np.float64) * dt, atol=1e-6
    )


@pytest.mark.parametrize("device", _WARP_DEVICES)
def test_mask_jacobian_is_exact(device):
    """Column masking is a multiply by 0.0 or 1.0 and must be bit-exact."""
    rng = np.random.default_rng(5)
    n, nv = 32, 7
    J = rng.normal(size=(n, 3, nv)).astype(np.float32)
    mask = (rng.random((n, nv)) > 0.5).astype(np.int32)

    Jw = wp.array(J.copy(), dtype=wp.float32, device=device)
    wp.launch(
        wk.mask_jacobian,
        dim=n,
        inputs=[Jw, wp.array(mask, dtype=wp.int32, device=device), nv],
        device=device,
    )
    wp.synchronize()

    np.testing.assert_array_equal(Jw.numpy(), J * mask[:, None, :])


# --- wp.from_numpy / wp.copy transfer contract ---


@pytest.mark.parametrize("device", _WARP_DEVICES)
@pytest.mark.parametrize(("np_dtype", "wp_dtype"), [(np.float32, wp.float32), (np.int32, wp.int32)])
def test_from_numpy_with_explicit_dtype_copies_into_2d_array(device, np_dtype, wp_dtype):
    """The exact transfer shape used by SimpleWarpKinematics for qpos and the Jacobian
    mask. With an explicit dtype the warp array keeps the numpy 2D shape and scalar
    element type, so wp.copy into a wp.array2d succeeds and preserves values."""
    batch, n = 4, 13
    arr = np.arange(batch * n, dtype=np_dtype).reshape(batch, n)

    src = wp.from_numpy(arr, dtype=wp_dtype, device=device)
    assert src.shape == (batch, n)
    assert wp.types.type_size_in_bytes(src.dtype) == np.dtype(np_dtype).itemsize

    dest = wp.zeros((batch, n), dtype=wp_dtype, device=device)
    wp.copy(dest, src)
    np.testing.assert_array_equal(dest.numpy(), arr)


def test_from_numpy_without_dtype_infers_a_vector_dtype():
    """Documents the trap. wp.from_numpy on a 2D array with no dtype collapses the
    trailing axis into a vector element, so the array no longer matches a
    wp.array2d(dtype=<scalar>) destination. Stable across warp 1.14 / 1.15 / 1.17."""
    batch, n = 4, 13
    arr = np.zeros((batch, n), dtype=np.float32)

    inferred = wp.from_numpy(arr, device="cpu")

    assert inferred.shape == (batch,)
    assert wp.types.type_size_in_bytes(inferred.dtype) == 4 * n


def test_all_from_numpy_calls_in_warp_kinematics_pass_an_explicit_dtype():
    """Guard against reintroducing the bug at a new call site.

    Every wp.from_numpy(...) in warp_kinematics.py must pass dtype=. Without it the
    inferred dtype depends on the input's ndim and warp >= 1.15 rejects the subsequent
    wp.copy with 'Incompatible array data types'."""
    calls = re.findall(r"wp\.from_numpy\((?:[^()]|\([^()]*\))*\)", _SOURCE, flags=re.DOTALL)
    assert calls, "expected wp.from_numpy call sites in warp_kinematics.py"
    missing = [c for c in calls if "dtype=" not in c]
    assert not missing, f"wp.from_numpy call(s) without an explicit dtype: {missing}"


def test_no_bare_float_builtin_inside_warp_kernels():
    """Kernel code must use wp.float32(...) rather than the Python float(...) builtin.

    Inside a kernel, float(x) is emitted as C++ ``wp::float(x)`` and only compiles
    because warp's generated module header contains ``#define float(x) cast_float(x)``.
    That is a preprocessor hack, not part of warp's public API. wp.float32(...) is
    explicit and produces identical machine code."""
    kernel_region = _SOURCE.split("class SimpleWarpKinematics", 1)[0]
    bare = re.findall(r"(?<![\w.])float\(", kernel_region)
    assert not bare, f"{len(bare)} bare float(...) call(s) remain in warp kernel code"
