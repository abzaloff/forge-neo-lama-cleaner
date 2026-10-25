import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'lama_cleaner_masked_content_sctipt.py'


class BatchLamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = {name: sys.modules.get(name) for name in
                        ('modules', 'modules.scripts', 'modules.processing', 'gradio')}
        modules = types.ModuleType('modules')
        modules.__path__ = []
        modules.scripts = types.ModuleType('modules.scripts')
        modules.scripts.Script = type('Script', (), {})
        modules.scripts.AlwaysVisible = object()
        modules.scripts.script_callbacks = types.SimpleNamespace(on_after_component=lambda callback: None)
        processing = types.ModuleType('modules.processing')
        processing.StableDiffusionProcessingImg2Img = type('StableDiffusionProcessingImg2Img', (), {})
        processing.create_binary_mask = lambda mask: mask.convert('L')
        gradio = types.ModuleType('gradio')
        gradio.Checkbox = lambda **kwargs: types.SimpleNamespace(**kwargs)
        gradio.Number = type('Number', (), {})
        sys.modules.update({'modules': modules, 'modules.scripts': modules.scripts,
                            'modules.processing': processing, 'gradio': gradio})
        spec = importlib.util.spec_from_file_location('batch_lama_script_test', SCRIPT)
        cls.script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.script_module)

    @classmethod
    def tearDownClass(cls):
        for name, previous in cls.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def setUp(self):
        self.script = self.script_module.Script()
        self.inpaint = Mock(side_effect=lambda image, mask, *args: image.copy())
        self.script_module.get_runtime_functions = Mock(return_value=(
            self.inpaint, lambda p: 'upscaler', lambda p: 512,
        ))
        self.script_module.NEW_ELEMENT_INDEX = 4

    def processing(self, mask, *, batch=True, batch_size=2, fill=0):
        image = Image.new('RGB', (12, 12), 'white')
        return types.SimpleNamespace(
            mask=None if batch else mask,
            image_mask=mask,
            init_images=[image] * batch_size,
            inpainting_fill=fill,
            inpaint_full_res=False,
            inpaint_full_res_padding=32,
            inpainting_mask_invert=0,
            mask_blur=4,
            extra_generation_params={},
        )

    def test_batch_control_is_injected_next_to_mask_directory(self):
        self.script.after_component(types.SimpleNamespace(elem_id='mode_img2img'))
        selected_tab = self.script_module.gr.Number()
        selected_tab.elem_id = None
        self.script.after_component(selected_tab)
        self.script.after_component(types.SimpleNamespace(elem_id='img2img_batch_inpaint_mask_dir'))
        control, tab = self.script.ui(True)
        self.assertIs(control, self.script.batch_lama)
        self.assertIs(tab, selected_tab)
        self.assertEqual(control.elem_id, 'img2img_batch_lama_cleaner')
        self.assertIsNone(self.script.ui(False))

    def test_each_batch_image_uses_its_current_mask_and_all_samples_get_result(self):
        first_mask = Image.new('L', (6, 5), 0)  # Different aspect ratio from the image.
        first_mask.putpixel((0, 0), 255)
        p = self.processing(first_mask)
        self.script.before_process(p, True, 5)
        self.assertEqual(self.inpaint.call_args.args[1].size, (12, 12))
        self.assertEqual(self.inpaint.call_args.args[1].getpixel((0, 0)), 255)
        self.assertIs(p.image_mask, self.inpaint.call_args.args[1])
        self.assertIs(p.init_images[0], p.init_images[1])
        self.assertEqual(p.inpainting_fill, 1)

        second_mask = Image.new('L', (12, 12), 0)
        second_mask.putpixel((11, 11), 255)
        p.image_mask = second_mask
        p.init_images = [Image.new('RGB', (12, 12), 'black')] * 2
        p.mask = object()  # Forge sets a latent mask after the first p.init().
        self.script.before_process(p, True, 5)
        self.assertEqual(self.inpaint.call_count, 2)
        self.assertEqual(self.inpaint.call_args.args[1].getpixel((11, 11)), 255)
        self.assertEqual(self.inpaint.call_args.args[1].getpixel((0, 0)), 0)

        third_mask = Image.new('L', (12, 12), 0)
        third_mask.putpixel((6, 6), 255)
        p.image_mask = third_mask
        p.init_images = [Image.new('RGB', (12, 12), 'blue')] * 2
        self.script.before_process(p, True, 5)
        self.assertEqual(self.inpaint.call_count, 3)
        self.assertEqual(self.inpaint.call_args.args[1].getpixel((6, 6)), 255)
        self.assertEqual(self.script_module.get_runtime_functions.call_args.kwargs,
                         {'reload_modules': False})

    def test_batch_requires_enable_while_regular_inpaint_keeps_masked_content_choice(self):
        mask = Image.new('L', (12, 12), 255)
        batch = self.processing(mask, batch=True, fill=4)
        self.script.before_process(batch, False, 5)
        self.inpaint.assert_not_called()

        regular = self.processing(mask, batch=False, fill=4)
        self.script.before_process(regular, False, 2)
        self.inpaint.assert_called_once()
        self.assertEqual(self.script_module.get_runtime_functions.call_args.kwargs,
                         {'reload_modules': True})

    def test_failed_batch_inpaint_does_not_keep_previous_image_metadata(self):
        p = self.processing(Image.new('L', (12, 12), 255))
        p.extra_generation_params['Masked content'] = 'lama cleaner'
        self.inpaint.side_effect = RuntimeError('model failed')
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            self.script.before_process(p, True, 5)
        self.assertNotIn('Masked content', p.extra_generation_params)


if __name__ == '__main__':
    unittest.main()
