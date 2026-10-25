import modules.scripts
import importlib
import gradio as gr
from PIL import Image
from modules.processing import StableDiffusionProcessingImg2Img, create_binary_mask


INPAINTING_FILL_ELEMENTS = ['img2img_inpainting_fill', 'replacer_inpainting_fill']


def get_runtime_functions(reload_modules=True):
    from lama_cleaner_masked_content import tools, model, inpaint, options

    if reload_modules:
        importlib.reload(tools)
        importlib.reload(model)
        importlib.reload(inpaint)
        importlib.reload(options)

    return inpaint.lamaInpaint, options.getLamaUpscaler, options.getResolution



class Script(modules.scripts.Script):
    def title(self):
        return "Lama-cleaner-masked-content"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if component.elem_id == 'mode_img2img':
            self.expect_selected_tab = True
        elif getattr(self, 'expect_selected_tab', False) and isinstance(component, gr.Number):
            self.selected_tab = component
            self.expect_selected_tab = False
        if component.elem_id == 'img2img_batch_inpaint_mask_dir':
            self.batch_lama = gr.Checkbox(
                label='Use Lama Cleaner in Batch',
                value=False,
                elem_id='img2img_batch_lama_cleaner',
                info='Match each image with a mask of the same filename in the mask directory.',
            )

    def ui(self, is_img2img):
        if not is_img2img:
            return None
        checkbox = getattr(self, 'batch_lama', None)
        if checkbox is None:
            checkbox = gr.Checkbox(label='Use Lama Cleaner in Batch', value=False,
                                   elem_id='img2img_batch_lama_cleaner')
        selected_tab = getattr(self, 'selected_tab', None)
        return [checkbox, selected_tab] if selected_tab is not None else [checkbox]

    def before_process(self, p: StableDiffusionProcessingImg2Img, batch_lama=False, selected_tab=None, *args):
        if not hasattr(p, 'inpainting_fill'):
            return

        # Forge reuses p for the whole directory. During p.init(), p.mask
        # becomes a latent tensor, so the selected UI tab identifies Batch.
        is_batch = selected_tab == 5 if selected_tab is not None else bool(batch_lama)
        use_batch_lama = is_batch and bool(batch_lama)
        if is_batch and not use_batch_lama:
            return
        if not is_batch and (NEW_ELEMENT_INDEX is None or p.inpainting_fill != NEW_ELEMENT_INDEX):
            return
        if use_batch_lama:
            p.extra_generation_params.pop("Masked content", None)
        if getattr(p, 'image_mask', None) is None or not p.init_images:
            return

        padding = None
        if p.inpaint_full_res:
            padding = p.inpaint_full_res_padding
        lamaInpaint, getLamaUpscaler, getResolution = get_runtime_functions(reload_modules=not use_batch_lama)
        mask = p.image_mask
        if use_batch_lama:
            mask = create_binary_mask(mask)
            if mask.size != p.init_images[0].size:
                mask = mask.resize(p.init_images[0].size, Image.Resampling.NEAREST)
            # Forge's inpaint stage must receive the same aligned mask as Lama.
            p.image_mask = mask
        result = lamaInpaint(p.init_images[0], mask,
                    p.inpainting_mask_invert, getLamaUpscaler(p), padding, getResolution(p), p.mask_blur)
        p.init_images = [result] * len(p.init_images)
        p.extra_generation_params["Masked content"] = "lama cleaner"
        p.inpainting_fill = 1 # original


NEW_ELEMENT_INDEX = None

def addIntoMaskedContent(component, **kwargs):
    elem_id = kwargs.get('elem_id', None)
    if elem_id not in INPAINTING_FILL_ELEMENTS:
        return
    newElement = ('lama cleaner', 'lama cleaner')
    if newElement not in component.choices:
        component.choices.append(newElement)
    global NEW_ELEMENT_INDEX
    NEW_ELEMENT_INDEX = component.choices.index(newElement)


modules.scripts.script_callbacks.on_after_component(addIntoMaskedContent)
