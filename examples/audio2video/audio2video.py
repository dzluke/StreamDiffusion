from pathlib import Path
import torch
import math
import librosa
# from pydub import AudioSegment
from subprocess import run
import time


import os
# import sys
# from typing import Literal, Dict, Optional
from PIL import Image
import numpy as np
import random
# import fire

import utils.bending as util
from utils.wrapper import StreamDiffusionWrapper



# set constants
FPS = 30
TXT2IMG = True
SAMPLING_RATE = 44100
IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./video_outputs")
util.set_sampling_rate(SAMPLING_RATE)

bending_functions_range = {
    util.add_full: (0, 5),
    util.multiply: (0, 10),  # the min could be 1 or 0
    util.add_sparse: (),
    util.add_noise: (0, 3),
    util.subtract_full: (0, 5),
    util.threshold: (0, 4),
    util.soft_threshold: (0, 2),
    util.soft_threshold2: (0, 2),
    util.inversion: (0, 5),
    util.inversion2: (),
    util.log: (0, 10),
    util.power: (),
    util.rotate_z: (0, 2 * math.pi),
    util.rotate_x: (0, 2 * math.pi),
    util.rotate_y: (0, 2 * math.pi),
    util.rotate_y2: (0, 2 * math.pi),
    util.reflect: (),
    util.hadamard1: (),
    util.hadamard2: (),
    # util.gradient: (),
    # util.dilation: (),
    # util.erosion: (),
    # util.sobel: (),
    util.absolute: ()
}


def generate_video(audio_path, prompt_file_path, layer, bend_function, audio_feature, smoothing_fn, seed):
    bend_function_name = bend_function.__name__
    audio_feature_name = audio_feature.__name__

    # initialize paths
    OUTPUT_VIDEO_PATH.mkdir(exist_ok=True)
    IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
    util.clear_dir(IMAGE_STORAGE_PATH)

    prompt = ""  # set this if NOT using txt2img
    negative_prompt = "low quality, text, words, letters"
    output = IMAGE_STORAGE_PATH
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    lora_dict = None
    width = 512
    height = 512
    frame_buffer_size = 1  # batch size
    acceleration = "xformers"
    guidance_scale = 1.2
    do_classifier_free_guidance = True if guidance_scale > 1.0 else False
    t_index_list = [0, 16, 32, 45]  # I think that the number of layers equals the len of this list

    def txt2img(wrapper, prompt, noise, bending_fn):
        wrapper.prepare(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            bending_fn=bending_fn,
            input_noise=noise
        )

        count = len(list(output.iterdir()))
        output_images = wrapper()
        output_images.save(os.path.join(output, f"{count:05}.png"))


    def encoding2img(wrapper, encoding, noise, bending_fn):
        wrapper.prepare(
            prompt_encoding=encoding,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            bending_fn=bending_fn,
            input_noise=noise
        )
        count = len(list(output.iterdir()))
        output_images = wrapper()
        output_images.save(os.path.join(output, f"{count:05}.png"))


    def encode_prompt(wrapper, prompt):
        encoder_output = wrapper.stream.pipe.encode_prompt(
            prompt=prompt,
            device=wrapper.stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        prompt_embeds = encoder_output[0].repeat(wrapper.stream.batch_size, 1, 1)
        return prompt_embeds


    wrapper = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,  # the length of this list is the number of denoising steps
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=False,
        cfg_type="none",
        seed=seed,
        bending_fn=bend_function
    )


    if not TXT2IMG:
        # generate first frame using txt2img
        txt2img(wrapper, prompt, None, None)
        # create img2img stream
        wrapper = StreamDiffusionWrapper(
                model_id_or_path=model_id_or_path,
                lora_dict=lora_dict,
                t_index_list=[22, 32, 45],
                frame_buffer_size=1,  # this is batch size
                width=width,
                height=height,
                warmup=10,
                acceleration=acceleration,
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="self",
                seed=seed,
        )


    # helper functions
    tic = time.time()

    # seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # read prompt file with format:
    # 0:00 : First prompt
    # 0:10 : Second prompt
    # etc...
    with open(prompt_file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        prompts = [line.split(":")[-1] for line in lines]
        times = [float(line.split(":")[0]) * 60 + float(line.split(":")[1]) for line in lines]
        prompt_times = [int(time * FPS) for time in times]
    num_prompts = len(prompts)

    # load input audio
    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)

    # calculate number of frames needed
    frame_length = 1. / FPS  # length of each frame in seconds
    num_frames = int(audio.size // (frame_length * SAMPLING_RATE)) + 1
    print(f">>> Generating {num_frames} frames")

    # Adding sin wave noise
    walk_length = 2  # set to 2 for 2pi walk
    noise = torch.empty((1, 4, wrapper.stream.latent_height, wrapper.stream.latent_width), dtype=torch.float64)
    walk_noise_x = torch.normal(mean=0, std=1, size=noise.shape, dtype=torch.float64)
    walk_noise_y = torch.normal(mean=0, std=1, size=noise.shape, dtype=torch.float64)
    walk_scale_x = torch.cos(torch.linspace(0, walk_length, num_frames) * math.pi).double()
    walk_scale_y = torch.sin(torch.linspace(0, walk_length, num_frames) * math.pi).double()
    noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
    noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)
    batched_noise = noise_x + noise_y

    # calculate audio features
    audio_features = []
    for i in range(0, num_frames):
        slice_start = int(i * frame_length * SAMPLING_RATE)
        slice_end = int((i + 1) * frame_length * SAMPLING_RATE)
        audio_slice = audio[slice_start:slice_end]
        value = audio_feature(audio_slice, SAMPLING_RATE)
        audio_features.append(value)

    # apply smoothing to audio features
    if smoothing_fn is not None:
        audio_features = smoothing_fn(audio_features)

    # calculate min/max for scaling
    audio_features_max = max(audio_features)
    audio_features_min = min(audio_features)
    bending_function_min, bending_function_max = bending_functions_range[bend_function]
    # rescale audio features to the range that the bending fn expects
    audio_features = [util.scale_range(x, audio_features_min, audio_features_max, bending_function_min, bending_function_max) for x in audio_features]

    # do prompt interpolations
    prompt_embeddings = []
    for i in range(num_prompts - 1):
        start_prompt, end_prompt = prompts[i], prompts[i + 1]
        start_frame, end_frame = prompt_times[i], prompt_times[i + 1]

        start_embedding = encode_prompt(wrapper, start_prompt)
        end_embedding = encode_prompt(wrapper, end_prompt)

        embeddings = np.linspace(start_embedding.cpu(), end_embedding.cpu(), num=(end_frame - start_frame))
        prompt_embeddings.append(embeddings)
    prompt_embeddings = np.vstack(prompt_embeddings)
    prompt_embeddings = torch.from_numpy(prompt_embeddings).to(wrapper.stream.device)
    # padding = num_frames - prompt_times[-1]
    # if padding > 0:
    #     prompt_embeddings = torch.nn.functional.pad(prompt_embeddings, (0, padding), mode='replicate')


    # generate frames
    prompt_embedding = prompt_embeddings[0]
    for i in range(num_frames):
        print(f"Frame {i} / {num_frames}")

        # bend is a function that defines how to apply network bending given a latent tensor and audio
        audio_feature_value = audio_features[i]
        print(">>> Audio Feature Value:", audio_feature_value)
        bend = bend_function(audio_feature_value)

        if TXT2IMG:
            try:
                prompt_embedding = prompt_embeddings[i]
            except IndexError:
                # the length of prompt_embeddings is shorter than num_frames, so just use the previous prompt_embedding
                pass
            encoding2img(wrapper, prompt_embedding, batched_noise[i], bend)
        else:
            if i == 0: continue
            guidance_scale = 0.5
            delta = 0.5
            wrapper.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            # randomness = np.random.rand(512, 512) * 255
            # input_image = Image.fromarray(randomness, mode='L')
            input_image = Image.open(os.path.join(output, f"{i-1:05}.png"))
            image_tensor = wrapper.preprocess_image(input_image)
            output_image = wrapper(image=image_tensor)
            output_image.save(os.path.join(output, f"{i:05}.png"))

        # every 10 seconds, create an in progress video
        # if i % (10 * fps) == 0:
        #     ffmpeg_command = ["ffmpeg",
        #                       "-y",  # automatically overwrite if output exists
        #                       "-framerate", str(fps),  # set framerate
        #                       "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
        #                       "-i", str(audio_path),  # set audio path
        #                       "-vcodec", "libx264",
        #                       "-pix_fmt", "yuv420p",
        #                       "in_progress.mp4"]
        #     run(ffmpeg_command)


    video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{FPS}fps_seed{seed}-StreamDiffusion.mp4"
    counter = 1
    while video_name.exists():
        video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{FPS}fps_seed{seed}-StreamDiffusion{counter}.mp4"
        counter += 1

    # turn images into video
    ffmpeg_command = ["ffmpeg",
                      "-y",  # automatically overwrite if output exists
                      "-framerate", str(FPS),  # set framerate
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
