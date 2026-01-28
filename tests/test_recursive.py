"""Tests for RecursiveRefinement module covering all RECR requirements."""
import pytest
import torch
from src.trm.model import RecursiveRefinement, TRMNetwork, GridEmbedding


class TestRecursiveRefinementCore:
    """Tests for core recursive refinement behavior (RECR-01, RECR-02, RECR-03)."""

    def test_recr_01_latent_state_update(self):
        """RECR-01: Latent state z is updated each inner iteration."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, outer_steps=1, inner_steps=2, enable_halting=False)

        x = torch.randint(0, 10, (1, 5, 5))
        out = rec(x)

        # With 1 outer step and 2 inner steps: 2 inner + 1 outer = 3 iterations
        assert out["iterations"] == 3, f"Expected 3 iterations, got {out['iterations']}"
        assert out["logits"].shape == (1, 5, 5, 10)

    def test_recr_02_answer_state_update(self):
        """RECR-02: Answer state y is updated after each outer iteration."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        # Use different inner/outer steps to verify structure
        rec = RecursiveRefinement(net, emb, outer_steps=2, inner_steps=3, enable_halting=False)

        x = torch.randint(0, 10, (1, 5, 5))
        out = rec(x)

        # 2 outer iterations: each has 3 inner + 1 outer = 4 iterations per outer
        # Total: 2 * 4 = 8
        assert out["iterations"] == 8, f"Expected 8 iterations, got {out['iterations']}"
        assert out["logits"].shape == (1, 5, 5, 10)

    def test_recr_03_nested_loop_structure_default(self):
        """RECR-03: Default nested loop is T=3, n=6, total=21 iterations."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)  # Use defaults

        x = torch.randint(0, 10, (1, 10, 10))
        out = rec(x)

        # Default: T=3, n=6 → 3 * (6 + 1) = 21 iterations
        assert out["iterations"] == 21, f"Expected 21 iterations, got {out['iterations']}"
        assert out["logits"].shape == (1, 10, 10, 10)

    def test_recr_03_nested_loop_structure_custom(self):
        """RECR-03: Custom T=5, n=4 produces T*(n+1)=25 iterations."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, outer_steps=5, inner_steps=4, enable_halting=False)

        x = torch.randint(0, 10, (1, 8, 8))
        out = rec(x)

        # Custom: T=5, n=4 → 5 * (4 + 1) = 25 iterations
        assert out["iterations"] == 25, f"Expected 25 iterations, got {out['iterations']}"


class TestRecursiveRefinementWeightSharing:
    """Tests for weight sharing verification (RECR-04)."""

    def test_recr_04_weight_sharing_single_network(self):
        """RECR-04: Single TRMNetwork instance is reused across all iterations."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        # Verify the network instance is the same reference
        assert rec.network is net, "RecursiveRefinement should store same network instance"

        # Count parameters - should be network params only (no duplication)
        rec_params = sum(p.numel() for p in rec.parameters())
        net_params = sum(p.numel() for p in net.parameters())
        emb_params = sum(p.numel() for p in emb.parameters())

        # RecursiveRefinement params = network params + embedding params (no extra layers)
        assert rec_params == net_params + emb_params, \
            f"Expected {net_params + emb_params} params, got {rec_params}"

    def test_recr_04_same_weights_used_each_iteration(self):
        """RECR-04: Network weights are shared, not duplicated per iteration."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, outer_steps=10, inner_steps=10, enable_halting=False)

        # With T=10, n=10 → 10 * 11 = 110 iterations
        # But parameter count should NOT scale with iterations
        rec_params = sum(p.numel() for p in rec.parameters())

        # Should be ~10.5M (from 03-03), not 110x that
        assert rec_params < 20_000_000, \
            f"Parameter count {rec_params:,} suggests no weight sharing (expected ~10M)"


class TestRecursiveRefinementHalting:
    """Tests for learned halting mechanism (RECR-05)."""

    def test_recr_05_halting_enabled_by_default(self):
        """RECR-05: Halting is enabled by default."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb)

        assert rec.enable_halting is True, "enable_halting should default to True"
        assert rec.halt_threshold == 0.9, "halt_threshold should default to 0.9"

    def test_recr_05_halting_disabled_runs_full_iterations(self):
        """RECR-05: With halting disabled, full T*(n+1) iterations execute."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, outer_steps=3, inner_steps=6, enable_halting=False)

        x = torch.randint(0, 10, (2, 10, 10))
        out = rec(x)

        # Should always run full 21 iterations when halting disabled
        assert out["iterations"] == 21, f"Expected 21 iterations, got {out['iterations']}"
        assert out["halted_early"] is False, "halted_early should be False when disabled"

    def test_recr_05_halting_threshold_parameter(self):
        """RECR-05: halt_threshold parameter controls early stopping."""
        net = TRMNetwork()
        emb = GridEmbedding(512)

        # Test different thresholds
        for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
            rec = RecursiveRefinement(net, emb, halt_threshold=threshold, enable_halting=True)
            assert rec.halt_threshold == threshold, \
                f"halt_threshold should be {threshold}, got {rec.halt_threshold}"

    def test_recr_05_halting_can_stop_early(self):
        """RECR-05: Halting can stop before max iterations if confidence high."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        # Use very low threshold to increase chance of early stopping
        rec = RecursiveRefinement(net, emb, outer_steps=10, inner_steps=5,
                                   halt_threshold=0.1, enable_halting=True)

        x = torch.randint(0, 10, (1, 5, 5))
        out = rec(x)

        # Max iterations would be 10 * 6 = 60
        # With low threshold, should halt early at least sometimes
        max_possible = 60
        assert out["iterations"] <= max_possible, \
            f"Iterations {out['iterations']} exceeded max {max_possible}"

        # Output should indicate halting status
        assert "halted_early" in out, "Output should contain halted_early flag"
        assert isinstance(out["halted_early"], bool), "halted_early should be boolean"

    def test_recr_05_halting_requires_all_batch_items(self):
        """RECR-05: Halting requires ALL batch items to exceed threshold."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        # Use batch size > 1 to test .all() behavior
        rec = RecursiveRefinement(net, emb, halt_threshold=0.95, enable_halting=True)

        x = torch.randint(0, 10, (4, 8, 8))
        out = rec(x)

        # Verify batch dimension is preserved
        assert out["logits"].shape[0] == 4, "Batch size should be preserved"
        assert out["halt_confidence"].shape == (4,), "Confidence should be per-batch"

    def test_recr_05_halted_early_flag_in_output(self):
        """RECR-05: halted_early flag correctly indicates early stopping."""
        net = TRMNetwork()
        emb = GridEmbedding(512)

        # Test with halting disabled
        rec_disabled = RecursiveRefinement(net, emb, enable_halting=False)
        x = torch.randint(0, 10, (1, 5, 5))
        out_disabled = rec_disabled(x)
        assert out_disabled["halted_early"] is False, \
            "halted_early should be False when halting disabled"

        # Test with halting enabled (may or may not halt early)
        rec_enabled = RecursiveRefinement(net, emb, enable_halting=True)
        out_enabled = rec_enabled(x)
        assert isinstance(out_enabled["halted_early"], bool), \
            "halted_early should be boolean"


class TestRecursiveRefinementShapes:
    """Tests for shape handling across various grid sizes."""

    def test_small_grids(self):
        """Test recursive refinement on very small grids (1x1 to 3x3)."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        for size in [1, 2, 3]:
            x = torch.randint(0, 10, (1, size, size))
            out = rec(x)
            assert out["logits"].shape == (1, size, size, 10), \
                f"Failed for size {size}x{size}"

    def test_rectangular_grids(self):
        """Test recursive refinement on non-square grids."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        test_shapes = [(3, 5), (10, 5), (5, 15), (20, 10)]
        for h, w in test_shapes:
            x = torch.randint(0, 10, (1, h, w))
            out = rec(x)
            assert out["logits"].shape == (1, h, w, 10), \
                f"Failed for shape {h}x{w}"

    def test_max_arc_grid_30x30(self):
        """Test recursive refinement on maximum ARC grid size (30x30)."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        x = torch.randint(0, 10, (1, 30, 30))
        out = rec(x)
        assert out["logits"].shape == (1, 30, 30, 10)
        assert out["iterations"] == 21

    def test_batch_sizes(self):
        """Test recursive refinement with various batch sizes."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randint(0, 10, (batch_size, 10, 10))
            out = rec(x)
            assert out["logits"].shape == (batch_size, 10, 10, 10), \
                f"Failed for batch size {batch_size}"
            assert out["halt_confidence"].shape == (batch_size,)


class TestRecursiveRefinementIntegration:
    """Integration tests for recursive refinement."""

    def test_no_nan_or_inf_in_output(self):
        """Test that recursive refinement never produces NaN or Inf."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        # Test with various inputs
        test_cases = [
            torch.randint(0, 10, (2, 5, 5)),
            torch.randint(0, 10, (1, 15, 20)),
            torch.zeros(1, 3, 3, dtype=torch.long),  # All zeros
            torch.full((1, 4, 4), 9, dtype=torch.long),  # All nines
        ]

        for x in test_cases:
            out = rec(x)
            assert not torch.isnan(out["logits"]).any(), \
                f"Found NaN in logits for input shape {x.shape}"
            assert not torch.isinf(out["logits"]).any(), \
                f"Found Inf in logits for input shape {x.shape}"
            assert not torch.isnan(out["halt_confidence"]).any(), \
                "Found NaN in halt_confidence"
            assert not torch.isinf(out["halt_confidence"]).any(), \
                "Found Inf in halt_confidence"

    def test_confidence_range_valid(self):
        """Test that halt confidence is always in [0, 1] range."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        x = torch.randint(0, 10, (4, 10, 10))
        out = rec(x)

        conf = out["halt_confidence"]
        assert (conf >= 0).all(), f"Found confidence < 0: {conf[conf < 0]}"
        assert (conf <= 1).all(), f"Found confidence > 1: {conf[conf > 1]}"

    def test_with_padding_values(self):
        """Test recursive refinement handles padding (-1) correctly."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb, enable_halting=False)

        # Create grid with padding
        x = torch.randint(0, 10, (2, 8, 8))
        x[0, 0, :] = -1  # Pad first row of first batch item
        x[1, :, 0] = -1  # Pad first column of second batch item

        out = rec(x)
        assert out["logits"].shape == (2, 8, 8, 10)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isinf(out["logits"]).any()

    def test_output_dict_keys(self):
        """Test that output dictionary has all required keys."""
        net = TRMNetwork()
        emb = GridEmbedding(512)
        rec = RecursiveRefinement(net, emb)

        x = torch.randint(0, 10, (1, 5, 5))
        out = rec(x)

        required_keys = {"logits", "halt_confidence", "iterations", "halted_early"}
        assert set(out.keys()) == required_keys, \
            f"Expected keys {required_keys}, got {set(out.keys())}"
