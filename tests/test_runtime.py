"""Real generation through a tiny random local model; no checkpoint download."""
import importlib.util
import threading
import unittest

from ntweak.engine import Runtime, find_backbone

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(HAS_RUNTIME, "Install inference dependencies to run local generation integration tests")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        words = ["[UNK]", "[PAD]", "[EOS]", "system", "user", "assistant", "hello", "world", "one", "two", "three", "four"]
        raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant "
        torch.manual_seed(12)
        config = LlamaConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                             eos_token_id=2, pad_token_id=1)
        rt = Runtime()
        rt.model = LlamaForCausalLM(config).eval()
        rt.tokenizer = tok
        rt.backbone = find_backbone(rt.model)
        rt.model_info = {"id": "test", "repo": "test/tiny-llama", "path": "unused"}
        rt.device, rt.state = "cpu", "ready"
        self.rt = rt
        self.body = {"messages": [{"role": "user", "content": "hello world"}], "recipe": {
            "model": "test/tiny-llama", "neurons": [{"layer": 0, "neuron": 1, "dose": 0.8}],
            "sampling": {"temperature": 0, "max_new_tokens": 6}}, "max_context": 256}

    def test_real_generation_emits_calibration_and_completion_and_removes_hooks(self):
        events = []
        self.rt.chat(self.body, threading.Event(), events.append)
        calibration = next(x for x in events if x["type"] == "calibration")
        self.assertEqual(calibration["neurons"][0]["dose"], 0.8)
        self.assertGreater(calibration["neurons"][0]["residual_median"], 0)
        self.assertEqual(events[-1]["type"], "done")
        self.assertGreater(events[-1]["output_tokens"], 0)
        self.assertEqual(self.rt.state, "ready")
        self.assertFalse(self.rt.backbone.layers[0].mlp.down_proj._forward_pre_hooks)
        self.assertFalse(self.rt.backbone.layers[0]._forward_hooks)

    def test_baseline_has_no_steering_and_is_repeatable(self):
        results = []
        for _ in range(2):
            events = []
            self.rt.chat({**self.body, "steering": False}, threading.Event(), events.append)
            self.assertEqual(next(x for x in events if x["type"] == "calibration")["neurons"], [])
            results.append("".join(x["text"] for x in events if x["type"] == "token"))
        self.assertEqual(*results)

    def test_stop_during_calibration_cleans_up_and_releases_model_lock(self):
        from ntweak.models import Cancelled
        event = threading.Event()
        def stop_on_phase(value):
            if value["type"] == "phase":
                event.set()
        with self.assertRaises(Cancelled):
            self.rt.chat(self.body, event, stop_on_phase)
        self.assertEqual(self.rt.state, "ready")
        self.assertFalse(self.rt.lock.locked())
        self.assertFalse(self.rt.backbone.layers[0]._forward_hooks)

    def test_wrong_checkpoint_is_rejected_before_hook_installation(self):
        self.body["recipe"]["model"] = "Qwen/Qwen3.5-4B"
        with self.assertRaisesRegex(ValueError, "targets"):
            self.rt.chat(self.body, threading.Event(), lambda _: None)
        self.assertFalse(self.rt.lock.locked())


if __name__ == "__main__":
    unittest.main()
