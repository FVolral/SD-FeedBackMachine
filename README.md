Animation Script v0.6
Inspired by Deforum Notebook
Must have ffmpeg installed in path.
Poor img2img implentation, will trash images that aren't moving.

Explanation of settings:
Video formats:
 Create GIF, webM or MP4 file from the series of images. Regardless, .bat files will be created with the right options to make the videos at a later time.

Total Animation Length (s):
 Total number of seconds to create. Will create fps frames for every second, as you'd expect.
Framerate:
 Frames per second to generate.

Denoising Strength:
 Initial denoising strength value, overrides the value above which is a bit strong for a default. Can be overridden later by keyframes.
Denoising Decay:
 Experimental option to enable a half-life decay on the denoising strength. Its value is halved every second.

Zoom Factor (scale/s):
 Zoom in (>1) or out (<1), at this rate per second. E.g. 2.0 will double size (and crop) every second. Can be overridden later by keyframes.
X Pixel Shift (pixels/s):
 Shift the image right (+) or left (-) in pixels per second. Can be overridden later by keyframes.
Y Pixel Shift (pixels/s):
 Shift the image down (+) or up (-) in pixels per second. Can be overridden later by keyframes.

Templates: Can be used with keyframes. if not used, img2img prompts or keyframes will be used.
Positive Prompts:
 A template for positive prompts that will be added to every keyframe prompt. Allows you to apply a style to an animation.
Negative Prompts:
 A template for negative prompts that will be added to every keyframe prompt. Allows you to apply a style to an animation.

Keyframes:
Format: Time (s) | Desnoise | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Positive Prompts | Negative Prompts
E.g. Doesn't much, zoom in, move around, zoom out again. expects animation length to be greater than 25s. Prompts blank, templates or the img2img values will be used.
  0 | 0.4 | 2.0 |    0 |    0 | | |
  5 | 0.4 | 1.0 |  250 |    0 | | |
 10 | 0.4 | 1.0 |    0 |  250 | | |
 15 | 0.4 | 1.0 | -250 |    0 | | |
 20 | 0.4 | 1.0 |    0 | -250 | | |
 25 | 0.4 | 0.5 |    0 |    0 | | |
