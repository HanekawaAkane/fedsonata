import unittest
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from wire_codec import encode_coordinates, encode_raw, decode_message, acknowledge_coordinates, _pack_bits, _unpack_bits, _frame, WireController
from mec.error_feedback import FactorResidualState


class WireTests(unittest.TestCase):
    def setUp(self):
        self.a = {"test": torch.tensor([[1., -.3, 0., .1, 4., -2., .7, -.5]])}
        self.b = {"test": torch.tensor([[.2], [0.], [-1.], [3.]])}

    def test_raw_message_exact_roundtrip_and_crc(self):
        blob = encode_raw(self.a, self.b)
        a, b, _ = decode_message(blob, self.a, self.b)
        torch.testing.assert_close(a["test"], self.a["test"], rtol=0, atol=0)
        torch.testing.assert_close(b["test"], self.b["test"], rtol=0, atol=0)
        broken = bytearray(blob)
        broken[-6] ^= 1
        with self.assertRaises(ValueError):
            decode_message(bytes(broken), self.a, self.b)

    def test_packed_values(self):
        for width in range(1, 9):
            values = np.arange(51, dtype=np.uint32) % (1 << width)
            np.testing.assert_array_equal(values, _unpack_bits(_pack_bits(values, width), len(values), width))

    def test_real_bytes_and_decode_for_all_codecs(self):
        a = {"test": torch.linspace(-2, 3, 400).reshape(4, 100)}
        b = {"test": torch.linspace(-1, .2, 300).reshape(100, 3)}
        for codec in ["topk", "stc", "qsgd"]:
            blob = encode_coordinates(a, b, codec, 700, np.random.default_rng(42))
            self.assertLessEqual(len(blob), 700)
            da, db, header = decode_message(blob, a, b)
            self.assertEqual(da["test"].shape, a["test"].shape)
            self.assertEqual(db["test"].shape, b["test"].shape)
            self.assertTrue(torch.isfinite(da["test"]).all())
            if codec == "stc":
                self.assertLessEqual(len(torch.unique(da["test"])), 3)

    def test_qsgd_zero_and_unbiased_rounding(self):
        zeros = {"test": torch.zeros(2, 2)}
        blob = encode_coordinates(zeros, {}, "qsgd", 200, np.random.default_rng(0))
        decoded, _, _ = decode_message(blob, zeros, {})
        self.assertEqual(torch.count_nonzero(decoded["test"]).item(), 0)
        x = {"test": torch.tensor([[.17, -.31, .51, 0.]])}
        # Choose a small transport budget forcing low precision, where stochastic
        # expectation matters; verify a Monte Carlo mean against the actual input.
        budget = len(_frame({"codec": "qsgd", "items": [["A", "test", [1, 4]]],
                            "n": 4, "width": 2, "levels": 1}, b"")) + 5
        sums = np.zeros(4)
        rng = np.random.default_rng(23)
        for _ in range(4000):
            blob = encode_coordinates(x, {}, "qsgd", budget, rng)
            d, _, header = decode_message(blob, x, {})
            self.assertEqual(header["width"], 2)
            sums += d["test"].numpy().ravel()
        np.testing.assert_allclose(sums / 4000, x["test"].numpy().ravel(), atol=.035, rtol=0)

    def test_residual_depends_on_delivery_and_decoded_update(self):
        for arrived in [False, True]:
            state = FactorResidualState(decay=.5)
            decoded = {"test": torch.zeros_like(self.a["test"])}
            decoded["test"][0, 0] = 1
            acknowledge_coordinates(state, 0, self.a, self.b, decoded, {}, arrived)
            expected = self.a["test"] - decoded["test"] if arrived else self.a["test"]
            torch.testing.assert_close(state.residual_a[0]["test"], expected)
            # Corrected increments must also include the retained coordinate error.
            corrected_a, _, _ = state.corrected(0, self.a, self.b)
            torch.testing.assert_close(corrected_a["test"], self.a["test"] + .5 * expected)

    def test_random_factor_control_fills_feasible_slots_reproducibly(self):
        a = {str(i): torch.arange(1024, dtype=torch.float32).reshape(8, 128) for i in range(6)}
        b = {str(i): torch.arange(1024, dtype=torch.float32).reshape(128, 8) for i in range(6)}
        choices = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temp:
                args = SimpleNamespace(wire_codec="random_rift", wire_replay=None, mec_output_dir=temp, seed=42)
                controller = WireController(args)
                controller.round_record = {"clients": {}}
                selection = controller.select(round_num=1, client_id=0, a_after=a, b_after=b,
                    delta_a=a, delta_b=b, budget_ratio=.35, reliability=1, reliability_floor=.25, quantum_bytes=4096)
                self.assertLessEqual(selection.selected_bytes, selection.budget_bytes)
                self.assertEqual(len(selection.updates_A) + len(selection.updates_B), 4)
                choices.append(selection.choices)
        self.assertEqual(choices[0], choices[1])


if __name__ == "__main__":
    unittest.main()
