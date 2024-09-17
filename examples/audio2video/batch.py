"""
Run many audio2video generations in a row
"""

from pathlib import Path
import utils.bending as util
import audio2video

# User Input:
# audio_path = Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-drone-v2.wav")
prompt_file_path = Path("C:\\Users\dzluk\StreamDiffusion\inputs\prompts.txt")


inputs = []
# inputs.append(
#     {
#          "audio": Path(""),
#          "prompt": Path(""),
#          "layer": 1,
#          "bend": util,
#          "feature": util,
#          "smoothing": None,
#          "seed": 46,
#     }
# )
inputs.append(
    {
         "audio": Path("C:\\Users\dzluk\StreamDiffusion\inputs\Breathing-drone-v2.wav"),
         "prompt": prompt_file_path,
         "layer": 1,
         "bend": util.rotate_x,
         "feature": util.centroid,
         "smoothing": None,
         "seed": 48,
    }
)



# def median_filtering(data):
#     return medfilt(data, kernel_size=9)
#
#
# def gaussian_filtering(data):
#     return gaussian_filter1d(data, sigma=3, radius=4)
#
#
# def log_smoothing(data):
#     return [math.log(x) if x > 0 else 0 for x in data]


for input in inputs:
    audio2video.generate_video(
                          input["audio"],
                          input["prompt"],
                          input["layer"],
                          input["bend"],
                          input["feature"],
                          input["smoothing"],
                          input["seed"]
    )
