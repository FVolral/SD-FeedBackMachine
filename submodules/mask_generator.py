"""
    Will generate mask for inpaiting
"""
import cairo
import numpy as np
import opensimplex
from PIL import Image, ImageFilter, ImageOps

try:
    from .utils import normalize, convert_from_np_to_image
    from .perlin_function import perlin, lerp, fade, gradient, gen_perlin_noise
except:
    from utils import normalize, convert_from_np_to_image
    from perlin_function import perlin, lerp, fade, gradient, gen_perlin_noise




def gen_mask(ctx, frame_no, w, h, mode):
    if mode == 'fft_data':
        pass
    elif mode == 'test':
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0,) # explicitly draw white background
        ctx.rectangle(0, 0, w/2, h/2)
        ctx.fill()
    elif mode == 'tunnel':
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0,) # explicitly draw white background

        p_z = (frame_no % 48) / 48.0
        p_z = p_z * 50
        n_tunnel = 16
        w_step = w / n_tunnel * 0.5
        h_step = h / n_tunnel * 0.5

        line_width = (w_step / 2) * 0.8
        ctx.set_line_width(line_width)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        for i in range(16):
            ctx.move_to(0 + w_step * i + p_z, h_step * i + p_z)
            ctx.line_to(w - w_step * i - p_z, h_step * i + p_z)
            ctx.line_to(w - w_step * i - p_z, h - h_step * i - p_z)
            ctx.line_to(w_step * i + p_z, h - h_step * i - p_z)
            ctx.close_path()
            ctx.stroke()


def get_mask(frame_no, w, h, mode, blur_fact):
    if mode in ['test', 'fft_data', 'tunnel']:
        surface = cairo.ImageSurface(cairo.FORMAT_A8, w, h,)
        ctx = cairo.Context(surface)

        gen_mask(ctx, frame_no, w, h, mode)

        image_mask = Image.frombuffer(mode = 'L', size = (w, h), data = surface.get_data(),)

        # image_mask= ImageOps.invert(image_mask)
    elif mode in ['perlin']:
        image_mask = gen_perlin_noise(w, h)
    elif mode in ['open_simplex']:
        pass

    blur_filter = ImageFilter.GaussianBlur(blur_fact)
    # image_mask = image_mask.filter(blur_filter)
    # image_mask = normalize(image_mask)
    image_mask = ImageOps.autocontrast(image_mask, cutoff=2)


    image_mask.save('test_mask_gen_mask.png') # debug
    return image_mask




# get_mask(0, 960, 540, 'test', 200)
# get_mask(0, 960, 540, 'tunnel', 3)

