import numpy as np
import pytest

from gagavatar_web import covariances_from_scale_rotation, decode_bundle, encode_bundle, hash_fill


def test_bundle_round_trip_keeps_arrays_aligned():
    rng = np.random.default_rng(0)
    arrays = {
        "pos": rng.normal(size=(7, 3)).astype(np.float32),
        "opacity": rng.uniform(size=7).astype(np.float32),
        "feat3": rng.uniform(size=(7, 3)).astype(np.float16),
    }
    meta = {"count": 7, "head_count": 2, "tan_fov": 1 / 12, "feature_channels": 3}
    data = encode_bundle(meta, arrays)
    assert data[:4] == b"GSB1"
    header_bytes = int.from_bytes(data[4:8], "little")
    assert (16 + header_bytes) % 16 == 0
    back_meta, back = decode_bundle(data)
    assert back_meta == meta
    for name, a in arrays.items():
        np.testing.assert_array_equal(back[name], a)


def test_bundle_rejects_other_files():
    with pytest.raises(ValueError, match="not a GSB1"):
        decode_bundle(b"GSB2" + bytes(12))


def test_hash_fill_matches_the_javascript_values():
    # packages/gagavatar-web/test/cpu.test.ts pins the same values.
    assert [f"{v:.6f}" for v in hash_fill(8, 3, 1.0)[:4]] == ["-0.159008", "-0.320069", "-0.061774", "-0.906509"]
    assert np.abs(hash_fill(10000, 0, 2.0)).max() <= 2.0


def test_covariances_from_scale_rotation():
    # A quarter turn about z swaps the x and y scales.
    s = np.array([[0.1, 0.2, 0.3]])
    q = np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    np.testing.assert_allclose(covariances_from_scale_rotation(s, q)[0], [0.04, 0, 0, 0.01, 0, 0.09], atol=1e-7)
    # Taken as given, the doubled quaternion makes the matrix [[-3, -4, 0], [4, -3, 0], [0, 0, 1]].
    np.testing.assert_allclose(covariances_from_scale_rotation(s, 2 * q)[0], [0.73, 0.36, 0, 0.52, 0, 0.09], atol=1e-6)
    np.testing.assert_allclose(covariances_from_scale_rotation(s, 2 * q, normalize=True), covariances_from_scale_rotation(s, q), rtol=1e-6)
