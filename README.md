# Animation Script
A basic img2img script that will dump frames and build a video file. Suitable for creating interesting zoom in warping movies, but not too much else at this time.
The basic idea is to story board some kinf od animation with changes in prompts, translation etc at a low framerate until you get it roughly right. Then bump up the framerate for a final render, it should play out roughly the same, just more detail.

Inspired by Deforum Notebook
Must have ffmpeg installed in path.
This suffers from img2img embossing, if the image is static for too long. I would have to look at someone else's implemetation to figure out why and don't want to steal their code.

## Explanation of settings:
### Video formats:
 Create GIF, webM or MP4 file from the series of images. Regardless, .bat files will be created with the right options to make the videos at a later time.

## Total Animation Length (s):
 Total number of seconds to create. Will create fps frames for every second, as you'd expect.
### Framerate:
 Frames per second to generate.

## Denoising Strength:
 Initial denoising strength value, overrides the value above which is a bit strong for a default. Will be overridden by keyframes when they are hit.
 Note that denoising is not scaled by fps, like other parameters are.
### Denoising Decay:
 Experimental option to enable a half-life decay on the denoising strength. Its value is halved every second. Not that useful because of img2img embossing.

### Zoom Factor (scale/s):
 Initial zoom in (>1) or out (<1), at this rate per second. E.g. 2.0 will double size (and crop) every second. Will be overridden by keyframes when they are hit.
### X Pixel Shift (pixels/s):
 Shift the image right (+) or left (-) in pixels per second. Will be overridden by keyframes when they are hit.
### Y Pixel Shift (pixels/s):
 Shift the image down (+) or up (-) in pixels per second. Will be overridden by keyframes when they are hit.

## Templates:
 Provide common positive and negative prompts for each keyframe below, save typing them out over and over. They will only be applied when a keyframe is hit. The prompts in the keyframes will be appended to these and sent for processing until the next keyframe that has a prompt.

## Keyframes:
Format: Time (s) | Desnoise | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Positive Prompts | Negative Prompts | Seed
A list of parameter changes to be applied at the specified time. 
