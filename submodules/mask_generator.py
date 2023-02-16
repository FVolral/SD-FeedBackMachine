"""
    Will generate mask for inpaiting
"""
import math
import cairo
import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    from .utils import normalize, convert_from_np_to_image
    from .perlin_function import perlin, lerp, fade, gradient, gen_perlin_noise
except:
    from utils import normalize, convert_from_np_to_image
    from perlin_function import perlin, lerp, fade, gradient, gen_perlin_noise



def path_regular_polygone(ctx, x_center, y_center, radius_h, radius_v, theta, edges=3):
    """Trace the path of the regular polygone in Cairo ctx.

    @param ctx: CairoContext
    @param x: float, x center
    @param x: float, y center
    @param radius: float, dist from center
    @param theta: angle rotation in degree
    @param edges: number of edge
    """
    step = 360 / edges
    theta = theta * math.pi / 180
    radians = theta
    x = x_center + math.cos(radians) * radius_h
    y = y_center + math.sin(radians) * radius_v

    ctx.move_to(x, y)
    for i in range(1, edges):
        radians = ((i * step)) * math.pi / 180 + theta
        x = x_center + math.cos(radians) * radius_h
        y = y_center + math.sin(radians) * radius_v
        ctx.line_to(x, y)

    ctx.close_path()

def gen_mask(ctx, frame_no, w, h, mode):
    w2 = w/2
    h2 = h/2

    if mode == 'fft_data':
        pass
    elif mode == 'test':
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0,) # explicitly draw white background
        ctx.rectangle(0, 0, w/2, h/2)
        ctx.fill()
    elif mode == 'tunnel':
        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0,) # explicitly draw white background

        p_z = (frame_no % 48) / 48.0
        # p_z = p_z ** 10

        n_tunnel = 32
        w_step = w / n_tunnel * 1
        h_step = h / n_tunnel * 1

        line_width = (w_step / 2) * 0.5
        ctx.set_line_width(line_width)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        for i in range(0, n_tunnel):
            radius_h = i * w_step + p_z * w_step
            radius_v = i * h_step + p_z * h_step
            radius_h = radius_h * 1
            radius_v = radius_v * 1
            path_regular_polygone(ctx, w2, h2, radius_h, radius_v, 45, 4 )
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

    blur_filter = ImageFilter.BoxBlur(radius=blur_fact)
    image_mask = image_mask.filter(blur_filter)
    image_mask = normalize(image_mask)
    image_mask = ImageOps.autocontrast(image_mask, cutoff=2)


    image_mask.save(f"test_mask_gen_mask_{frame_no % 48}.png") # debug
    return image_mask



if __name__ == "__main__":
    for i in range(48):
        get_mask(i, 960, 540, 'tunnel', 0)

