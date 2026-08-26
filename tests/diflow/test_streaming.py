# Copyright 2026 The DiFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download-free regression tests for DiFlow's additions to upstream diffusers.

Run with: python -m unittest discover -s tests/diflow -v
"""

import unittest
from unittest.mock import patch

import torch

from diffusers import (
    ControlNetModel,
    FluxControlNetModel,
    FluxTransformer2DModel,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    UNet2DConditionModel,
)
from diffusers.models.modeling_utils import apply_lora_scale_to_generator, get_lazy_tensor


def flux_config(guidance=False):
    return {
        "in_channels": 4,
        "num_layers": 2,
        "num_single_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 12,
        "pooled_projection_dim": 8,
        "axes_dims_rope": (2, 2, 4),
        "guidance_embeds": guidance,
    }


def flux_inputs(guidance=False):
    result = {
        "hidden_states": torch.randn(2, 4, 4),
        "encoder_hidden_states": torch.randn(2, 3, 12),
        "pooled_projections": torch.randn(2, 8),
        "timestep": torch.tensor([0.1, 0.3]),
        "img_ids": torch.zeros(4, 3),
        "txt_ids": torch.zeros(3, 3),
    }
    if guidance:
        result["guidance"] = torch.tensor([3.5, 3.5])
    return result


def sd3_config():
    return {
        "sample_size": 8,
        "patch_size": 2,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 12,
        "caption_projection_dim": 16,
        "pooled_projection_dim": 8,
        "pos_embed_max_size": 8,
    }


def sd3_inputs():
    return {
        "hidden_states": torch.randn(2, 4, 8, 8),
        "encoder_hidden_states": torch.randn(2, 3, 12),
        "pooled_projections": torch.randn(2, 8),
        "timestep": torch.tensor([100.0, 300.0]),
    }


def unet_config(sdxl=False):
    result = {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": (16, 32),
        "layers_per_block": 1,
        "norm_num_groups": 8,
        "cross_attention_dim": 12,
        "attention_head_dim": 2,
        "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
    }
    if sdxl:
        result.update(
            addition_embed_type="text_time", addition_time_embed_dim=2, projection_class_embeddings_input_dim=20
        )
    return result


def unet_inputs(sdxl=False):
    result = {
        "sample": torch.randn(2, 4, 8, 8),
        "timestep": torch.tensor(100),
        "encoder_hidden_states": torch.randn(2, 3, 12),
    }
    if sdxl:
        result["added_cond_kwargs"] = {"text_embeds": torch.randn(2, 8), "time_ids": torch.ones(2, 6)}
    return result


def nonzero_projections(modules):
    # ControlNet projections default to zero; randomize so parity tests exercise the computation.
    for module in modules:
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.03)


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def assertTensorEqual(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_get_lazy_tensor(self):
        tensor = torch.ones(2)
        self.assertIs(get_lazy_tensor(tensor), tensor)
        self.assertIs(get_lazy_tensor(lambda: tensor), tensor)

    @torch.no_grad()
    def test_flux_stream_parity_and_deferred_order(self):
        for guidance in (False, True):
            with self.subTest(guidance=guidance):
                model = FluxTransformer2DModel(**flux_config(guidance)).eval()
                inputs = flux_inputs(guidance)
                self.assertTensorEqual(model.stream_forward(**inputs), model(**inputs).sample)
                residuals = [torch.randn(2, 4, 16) * 0.1 for _ in range(2)]
                singles = [torch.randn(2, 4, 16) * 0.1 for _ in range(2)]
                expected = model(
                    **inputs, controlnet_block_samples=residuals, controlnet_single_block_samples=singles
                ).sample
                events = []
                hooks = [
                    block.register_forward_hook(lambda *args, i=i: events.append(("block", i)))
                    for i, block in enumerate(model.transformer_blocks)
                ]

                def deferred(i):
                    events.append(("residual", i))
                    return residuals[i]

                try:
                    actual = model.stream_forward(
                        **inputs,
                        controlnet_block_samples=[lambda i=i: deferred(i) for i in range(2)],
                        controlnet_single_block_samples=[lambda i=i: singles[i] for i in range(2)],
                    )
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertTensorEqual(actual, expected)
                self.assertEqual(events, [("block", 0), ("residual", 0), ("block", 1), ("residual", 1)])
                expected = model(
                    **inputs, controlnet_block_samples=residuals[:1], controlnet_blocks_repeat=True
                ).sample
                self.assertTensorEqual(
                    model.stream_forward(
                        **inputs, controlnet_block_samples=[lambda: residuals[0]], controlnet_blocks_repeat=True
                    ),
                    expected,
                )

    @torch.no_grad()
    def test_sd3_stream_parity(self):
        model = SD3Transformer2DModel(**sd3_config()).eval()
        inputs = sd3_inputs()
        residuals = [torch.randn(2, 16, 16) * 0.1 for _ in range(2)]
        for skip_layers in (None, [0]):
            with self.subTest(skip_layers=skip_layers):
                expected = model(**inputs, block_controlnet_hidden_states=residuals, skip_layers=skip_layers).sample
                actual = model.stream_forward(
                    **inputs,
                    block_controlnet_hidden_states=[lambda x=x: x for x in residuals],
                    skip_layers=skip_layers,
                )
                self.assertTensorEqual(actual, expected)

    @torch.no_grad()
    def test_flux_controlnet_generator(self):
        for union in (False, True):
            with self.subTest(union=union):
                config = flux_config(True)
                config["num_single_layers"] = 0  # DiFlow's existing double-block-only contract.
                if union:
                    config["num_mode"] = 2
                model = FluxControlNetModel(**config).eval()
                nonzero_projections(model.controlnet_blocks)
                inputs = flux_inputs(True)
                inputs.update(controlnet_cond=torch.randn(2, 4, 4), conditioning_scale=0.6)
                if union:
                    inputs["controlnet_mode"] = torch.zeros(2, 1, dtype=torch.long)
                    inputs["txt_ids"] = inputs["txt_ids"].unsqueeze(0)
                    inputs["img_ids"] = inputs["img_ids"].unsqueeze(0)
                expected = model(**inputs).controlnet_block_samples
                actual = list(model.yield_controlnet_block_samples(**inputs))
                self.assertEqual(len(actual), len(expected))
                for i, item in enumerate(actual):
                    self.assertEqual(list(item), [f"control_block_sample_{i}"])
                    self.assertTensorEqual(item[f"control_block_sample_{i}"], expected[i])

    @torch.no_grad()
    def test_sd3_controlnet_generator(self):
        model = SD3ControlNetModel(**sd3_config()).eval()
        nonzero_projections(model.controlnet_blocks)
        inputs = sd3_inputs()
        inputs.update(controlnet_cond=torch.randn(2, 4, 8, 8), conditioning_scale=0.6)
        expected = model(**inputs).controlnet_block_samples
        actual = list(model.yield_controlnet_block_samples(**inputs))
        self.assertEqual(len(actual), len(expected))
        for i, item in enumerate(actual):
            self.assertTensorEqual(item[f"control_block_sample_{i}"], expected[i])

    @torch.no_grad()
    def test_unet_and_controlnet_streaming(self):
        for sdxl in (False, True):
            with self.subTest(sdxl=sdxl):
                model = UNet2DConditionModel(**unet_config(sdxl)).eval()
                controlnet = ControlNetModel.from_unet(model, conditioning_embedding_out_channels=(4, 8)).eval()
                nonzero_projections([*controlnet.controlnet_down_blocks, controlnet.controlnet_mid_block])
                inputs = unet_inputs(sdxl)
                self.assertTensorEqual(model.stream_forward(**inputs), model(**inputs).sample)
                for guess_mode, pooled in ((False, False), (True, False), (True, True)):
                    with self.subTest(guess_mode=guess_mode, pooled=pooled):
                        controlnet.register_to_config(global_pool_conditions=pooled)
                        controls = dict(
                            **inputs,
                            controlnet_cond=torch.randn(2, 3, 16, 16),
                            conditioning_scale=0.6,
                            guess_mode=guess_mode,
                        )
                        down, mid = controlnet(**controls, return_dict=False)
                        actual = list(controlnet.yield_control_block_samples(**controls))
                        self.assertEqual(len(actual), len(down) + 1)
                        for i, expected in enumerate(down):
                            self.assertTensorEqual(actual[i][f"down_block_res_sample_{i}"], expected)
                        self.assertTensorEqual(actual[-1]["mid_block_res_sample"], mid)
                        expected = model(
                            **inputs, down_block_additional_residuals=down, mid_block_additional_residual=mid
                        ).sample
                        actual = model.stream_forward(
                            **inputs,
                            down_block_additional_residuals=[lambda x=x: x for x in down],
                            mid_block_additional_residual=lambda: mid,
                        )
                        self.assertTensorEqual(actual, expected)

    def test_gradient_checkpointing_still_works(self):
        for cls, config, inputs in (
            (FluxTransformer2DModel, flux_config(), flux_inputs()),
            (SD3Transformer2DModel, sd3_config(), sd3_inputs()),
        ):
            with self.subTest(model=cls.__name__):
                model = cls(**config).train()
                model.enable_gradient_checkpointing()
                inputs["hidden_states"].requires_grad_()
                expected = model(**inputs).sample
                expected.sum().backward()
                expected_grad = inputs["hidden_states"].grad.detach().clone()
                model.zero_grad(set_to_none=True)
                inputs["hidden_states"].grad = None
                actual = model.stream_forward(**inputs)
                actual.sum().backward()
                self.assertTensorEqual(actual, expected)
                self.assertTensorEqual(inputs["hidden_states"].grad, expected_grad)

    @torch.no_grad()
    def test_lora_scale_parity(self):
        from peft import LoraConfig

        model = FluxTransformer2DModel(**flux_config()).eval()
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, init_lora_weights=False, target_modules=["to_q", "to_v"]))
        inputs = flux_inputs()
        kwargs = {"scale": 0.4}
        expected = model(**inputs, joint_attention_kwargs=kwargs).sample
        actual = model.stream_forward(**inputs, joint_attention_kwargs=kwargs)
        self.assertTensorEqual(actual, expected)
        self.assertEqual(kwargs, {"scale": 0.4})

    @torch.no_grad()
    def test_controlnet_generator_lora_parity(self):
        from peft import LoraConfig

        flux_options = flux_config()
        flux_options["num_single_layers"] = 0
        for cls, config, inputs in (
            (FluxControlNetModel, flux_options, flux_inputs()),
            (SD3ControlNetModel, sd3_config(), sd3_inputs()),
        ):
            with self.subTest(model=cls.__name__):
                model = cls(**config).eval()
                nonzero_projections(model.controlnet_blocks)
                model.add_adapter(
                    LoraConfig(r=2, lora_alpha=2, init_lora_weights=False, target_modules=["to_q", "to_v"])
                )
                inputs["controlnet_cond"] = torch.randn_like(inputs["hidden_states"])
                for scale in (0.0, 0.4, 1.0):
                    with self.subTest(scale=scale):
                        kwargs = {"scale": scale}
                        expected = model(**inputs, joint_attention_kwargs=kwargs).controlnet_block_samples
                        actual = list(model.yield_controlnet_block_samples(**inputs, joint_attention_kwargs=kwargs))
                        self.assertEqual(len(actual), len(expected))
                        for i, sample in enumerate(expected):
                            self.assertTensorEqual(actual[i][f"control_block_sample_{i}"], sample)
                        self.assertEqual(kwargs, {"scale": scale})

    def test_generator_lora_restored_between_yields_and_on_error(self):
        class Model:
            scale = 1.0
            fail = False

            @apply_lora_scale_to_generator("cross_attention_kwargs")
            def samples(self, cross_attention_kwargs=None):
                assert "scale" not in cross_attention_kwargs
                yield self.scale
                if self.fail:
                    raise RuntimeError("expected failure")
                yield self.scale

        def scale(model, value):
            model.scale *= value

        def unscale(model, value):
            model.scale /= value

        with (
            patch("diffusers.utils.USE_PEFT_BACKEND", True),
            patch("diffusers.utils.peft_utils.scale_lora_layers", scale),
            patch("diffusers.utils.peft_utils.unscale_lora_layers", unscale),
        ):
            model = Model()
            kwargs = {"scale": 0.4}
            generator = model.samples(kwargs)  # Preserve positional attention kwargs too.
            self.assertEqual(next(generator), 0.4)
            self.assertEqual(model.scale, 1.0)
            generator.close()
            self.assertEqual(model.scale, 1.0)
            self.assertEqual(kwargs, {"scale": 0.4})
            self.assertEqual(list(model.samples(kwargs)), [0.4, 0.4])
            model.fail = True
            generator = model.samples(kwargs)
            next(generator)
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                next(generator)
            self.assertEqual(model.scale, 1.0)


if __name__ == "__main__":
    unittest.main()
