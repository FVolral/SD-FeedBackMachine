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


cairo_mode = ['croix', 'fft_data', 'tunnel', 'tunnel_2']
noise_mode = ['perlin']

mask_modes = cairo_mode + noise_mode


def white(ctx):
    ctx.set_source_rgb(1.0, 1.0, 1.0)

def black(ctx):
    ctx.set_source_rgb(0.0, 0.0, 0.0)

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
    elif mode == 'croix':
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        line_width = w / 10
        ctx.set_line_width(line_width)
        ctx.move_to(0, 0)
        ctx.line_to(0, h)
        ctx.line_to(w, h)
        ctx.line_to(w, 0)
        ctx.close_path()
        ctx.stroke()
        ctx.move_to(0, 0)
        ctx.line_to(w, h)
        ctx.stroke()
        ctx.move_to(w, 0)
        ctx.line_to(0, h)
        ctx.stroke()
    elif mode == 'inverseur_1':
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        line_width = w / 10
        ctx.set_line_width(line_width)

    elif mode == 'tunnel':
        p_z = (frame_no % 48) / 48.0
        # p_z = p_z ** 10

        n_tunnel = 32
        w_step = w / n_tunnel * 1
        h_step = h / n_tunnel * 1

        line_width = (w_step / 2) * 0.5
        ctx.set_line_width(line_width)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
        for i in range(0, n_tunnel):
            radius_h = i * w_step + p_z * w_step
            radius_v = i * h_step + p_z * h_step
            radius_h = radius_h * 1
            radius_v = radius_v * 1
            path_regular_polygone(ctx, w2, h2, radius_h, radius_v, 45, 4 )
            ctx.stroke()
    elif mode == 'tunnel_2':
        white(ctx)

        p_z = (frame_no % 48) / 48.0
        # p_z = p_z ** 10

        n_tunnel = 32
        step = w / n_tunnel * 1
        aspect = w/h

        # ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        for i in range(n_tunnel, 0, -1 ):
            if i % 2 == 0:
                ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)
            else:
                ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            radius = 5  + i * step + p_z * step

            path_regular_polygone(ctx, w2, h2, radius, radius / aspect, 45, 4)

            #ctx.fill_preserve()
            ctx.fill()
            #ctx.stroke()



def get_mask(frame_no, w, h, mode, blur_fact):
    if mode in cairo_mode :
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h,)
        ctx = cairo.Context(surface)

        gen_mask(ctx, frame_no, w, h, mode)

        image_mask = Image.frombuffer(mode = 'RGBA', size = (w, h), data = surface.get_data())
        image_mask = image_mask.convert('L')
        # image_mask= ImageOps.invert(image_mask)
    elif mode in noise_mode:
        image_mask = gen_perlin_noise(w, h)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # blur_filter = ImageFilter.BoxBlur(radius=blur_fact)
    # image_mask = image_mask.filter(blur_filter)
    # image_mask = normalize(image_mask)
    # image_mask = ImageOps.autocontrast(image_mask, cutoff=2)


    image_mask.save(f"test_mask_gen_mask_{frame_no % 48}.png") # debug
    return image_mask



if __name__ == "__main__":
    for i in range(48):
        get_mask(i, 960, 540, 'croix', 0)

