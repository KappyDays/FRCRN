import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='speech_frcrn_ans_cirm_16k') # model='damo/speech_frcrn_ans_cirm_16k'


wav_path = "../../datasets/VoxSRC/Full_Data/voxconverse/DEV402/wav/"
output_path = "se_result/VoxSRC_DEV402_SE_FRCRN/"


file_list = os.listdir(wav_path)
for file in file_list:
    result = ans(wav_path+file, output_path=output_path+file)
