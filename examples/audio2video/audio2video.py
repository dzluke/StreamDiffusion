from pathlib import Path
import torch
import math
import librosa
# from pydub import AudioSegment
# import generatee
from subprocess import run
import time

import utils.bending
import utils.bending as util

import os
import sys
from typing import Literal, Dict, Optional

import fire

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


# set constants
TXT2IMG = True
SAMPLING_RATE = 44100
IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./video_outputs")
util.set_sampling_rate(SAMPLING_RATE)

# initialize paths
OUTPUT_VIDEO_PATH.mkdir(exist_ok=True)
IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
util.clear_dir(IMAGE_STORAGE_PATH)


"""
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """

output = IMAGE_STORAGE_PATH
model_id_or_path = "runwayml/stable-diffusion-v1-5"
lora_dict = None
prompt = "a floating orb"
width = 512
height = 512
frame_buffer_size = 1
acceleration = "xformers"
seed = 46

os.makedirs(output, exist_ok=True)

bend = utils.bending.add_full(1)

stream = StreamDiffusionWrapper(
    model_id_or_path=model_id_or_path,
    lora_dict=lora_dict,
    t_index_list=[0, 16, 32, 45],
    frame_buffer_size=frame_buffer_size,
    width=width,
    height=height,
    warmup=10,
    acceleration=acceleration,
    mode="txt2img",
    use_denoising_batch=False,
    cfg_type="none",
    seed=seed,
    bending_fn=bend
)


def txt2img(wrapper, prompt, noise, bending_fn):
    wrapper.prepare(
        prompt=prompt,
        num_inference_steps=50,
        bending_fn=bending_fn,
        input_noise=noise
    )

    count = len(list(output.iterdir()))
    output_images = wrapper()
    output_images.save(os.path.join(output, f"{count:05}.png"))
    # for i, output_image in enumerate(output_images):
    #     output_image.save(os.path.join(output, f"{count + i:05}.png"))


# helper functions
tic = time.time()

# take input from command line args
args = util.run_argparse()

audio_path = Path(args.audio)
audio_path = Path("C:\\Users\dzluk\StreamDiffusion\inputs\hits.wav")

# load input audio
audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)

# calculate number of frames needed
frame_length = 1. / args.fps  # length of each frame in seconds
num_frames = int(audio.size // (frame_length * SAMPLING_RATE)) + 1
print(f">>> Generating {num_frames} frames")

# the user inputs a visual aesthetic in the form of image(s), video(s), or text



# Adding sin wave noise
walk_length = 0.5  # set to 2 for 2pi walk
noise = torch.empty((1, 4, stream.stream.latent_height, stream.stream.latent_width), dtype=torch.float64)
walk_noise_x = torch.distributions.normal.Normal(0, 1).sample(noise.shape).double()
walk_noise_y = torch.distributions.normal.Normal(0, 1).sample(noise.shape).double()

walk_scale_x = torch.cos(torch.linspace(0, walk_length, num_frames) * math.pi).double()
walk_scale_y = torch.sin(torch.linspace(0, walk_length, num_frames) * math.pi).double()
noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)
batched_noise = noise_x + noise_y

# if the user input is text
prompt = "3D mesh geometry"
layer = 1
bend = util.add_full
bend_function_name = bend.__name__
audio_feature = util.rms
audio_feature_name = audio_feature.__name__

# generate frames
for i in range(num_frames):
    slice_start = int(i * frame_length * SAMPLING_RATE)
    slice_end = int((i + 1) * frame_length * SAMPLING_RATE)
    audio_slice = audio[slice_start:slice_end]

    # bend is a function that defines how to apply network bending given a latent tensor and audio
    audio_feature = audio_feature(audio_slice, SAMPLING_RATE) * 10
    print(">>> Audio Feature Value:", audio_feature)
    bend = bend(audio_feature)

    if TXT2IMG:
        txt2img(stream, prompt, batched_noise[i], bend)
    else:
        pass
        # args.seed += 1
        # generate.img2img(model, encoding, init_img_path, IMAGE_STORAGE_PATH, bend, layer, args)

    # every 10 seconds, create an in progress video
    if i % (10 * args.fps) == 0:
        ffmpeg_command = ["ffmpeg",
                          "-y",  # automatically overwrite if output exists
                          "-framerate", str(args.fps),  # set framerate
                          "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                          "-i", str(audio_path),  # set audio path
                          "-vcodec", "libx264",
                          "-pix_fmt", "yuv420p",
                          "in_progress.mp4"]
        run(ffmpeg_command)


video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{args.fps}fps.mp4"
counter = 1
while video_name.exists():
    video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{args.fps}fps{counter}.mp4"
    counter += 1

# turn images into video
ffmpeg_command = ["ffmpeg",
                  "-y",  # automatically overwrite if output exists
                  "-framerate", str(args.fps),  # set framerate
                  "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                  "-i", str(audio_path),  # set audio path
                  "-vcodec", "libx264",
                  "-pix_fmt", "yuv420p",
                  str(video_name)]
run(ffmpeg_command)

print(">>> Saved video as", video_name)
print(">>> Generated {} images".format(num_frames))
print(">>> Took", util.time_string(time.time() - tic))
print(">>> Avg time per frame:", (time.time() - tic) / num_frames)
print("Done.")
