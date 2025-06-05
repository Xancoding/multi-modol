import numpy as np
import config
import librosa
import pickle
import utils
import os


def Wav2Segments(dataDir, labelDir):
    def SlidingWindows2Segments(CryList, raw_audio, sample_rate, slidingLengthWindows, step, start=0):
        slidingLengthWindows = int(slidingLengthWindows * sample_rate)  # number of frame
        step = int(step * sample_rate)  # number of step

        sample = {
            "data": [],
            "label": []
        }
        while start < len(raw_audio):
            end = int(start + slidingLengthWindows)
            # Data Acq
            if end > len(raw_audio):
                data = raw_audio[start: -1]
                padding = np.zeros((end-len(raw_audio)+1,))
                data = np.concatenate((data, padding))
            else:
                data = raw_audio[start: end]

            # 标签设置（Cry=1, No-Cry=0）
            label = 0  # 默认非哭声
            frameWork = [i for i in range(start, end)]
            if len(list(set(frameWork).intersection(set(CryList)))) != 0:
                label = 1  # 如果与哭声区间有交集，标记为哭声

            # 存储
            sample["data"].append(data)
            sample["label"].append(label)

            start += step

        sample["data"] = np.array(sample["data"])
        sample["label"] = np.array(sample["label"])

        return sample


    raw_audio, sample_rate = librosa.load(path=dataDir, sr=config.audioSampleRate)

    # Label from file
    CryList = None
    f = open(f"{labelDir}", "r")
    res = f.readlines()
    for cur in res:
        cur = cur.strip("\n")
        lStart, lEnd, lSet = cur.split("\t")

        lStart = int(float(lStart) * config.audioSampleRate)
        lEnd = int(float(lEnd) * config.audioSampleRate)

        if int(lSet) == 1:
            if CryList is None:
                CryList = np.array([i for i in range(lStart, lEnd)])
            else:
                CryList = np.concatenate((CryList, np.array([i for i in range(lStart, lEnd)])))
    if CryList is None:
        CryList = []

    # Sliding windows to traim audio
    sample = SlidingWindows2Segments(CryList, raw_audio, sample_rate, config.slidingWindows, config.step, start=0)

    # Save
    saveDir = dataDir.split(".")[0] + ".dat"
    file = open(saveDir, 'wb')
    pickle.dump(sample, file)
    file.close()

    return sample


def NICUWav2Segments(dataDir, labelDir):
    # Load and preprocess audio
    raw_audio, sr = librosa.load(dataDir, sr=config.audioSampleRate)
    min_t, max_t = utils.get_valid_time_range(labelDir)
    raw_audio = utils.crop_data_by_time(raw_audio, sr, min_t, max_t)

    # Process cry segments (vectorized)
    cry_ranges = []
    if os.path.exists(labelDir):
        with open(labelDir) as f:
            for line in filter(None, map(str.strip, f)):
                parts = line.replace('\t', ' ').split()
                if len(parts) == 3 and int(parts[2]) == 1:
                    start, end = (int((float(t)-min_t)*sr) for t in parts[:2])
                    cry_ranges.append((start, end))

    # Generate valid window indices (discard incomplete segments)
    win_size, step_size = utils.calculate_window_params(sr, config.slidingWindows, config.step)
    valid_indices = np.arange(0, len(raw_audio) - win_size + 1, step_size)
    
    # Vectorized window extraction (automatically discards incomplete segments)
    windows = np.stack([raw_audio[i:i+win_size] for i in valid_indices])
    
    # Vectorized label generation
    cry_mask = np.zeros(len(raw_audio), dtype=bool)
    for start, end in cry_ranges:
        cry_mask[start:end] = True
    labels = np.array([cry_mask[i:i+win_size].any() for i in valid_indices], dtype=int)

    return {"data": windows, "label": labels}


def main():
    dataDir = f"Data\\Crying_Github.wav"
    labelDir = f"Data\\Crying_Github.txt"
    sample = Wav2Segments(dataDir, labelDir)
    print(sample)


if __name__ == '__main__':
    main()






