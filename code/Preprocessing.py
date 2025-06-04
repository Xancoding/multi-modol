import numpy as np
import config
import librosa
import pickle


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
    def SlidingWindows2Segments(CryList, raw_audio, sample_rate, slidingLengthWindows, step, start=0):
        slidingLengthWindows = int(slidingLengthWindows * sample_rate)  # number of frames
        step = int(step * sample_rate)  # number of step frames

        sample = {
            "data": [],
            "label": []
        }
        while start < len(raw_audio):
            end = int(start + slidingLengthWindows)
            # Data Acquisition
            if end > len(raw_audio):
                data = raw_audio[start: -1]
                padding = np.zeros((end-len(raw_audio)+1,))
                data = np.concatenate((data, padding))
            else:
                data = raw_audio[start: end]

            # Label setting (Cry=1, No-Cry=0)
            label = 0  # Default non-cry
            frameWork = [i for i in range(start, end)]
            if len(list(set(frameWork).intersection(set(CryList)))) != 0:
                label = 1  # If intersects with cry intervals, mark as cry

            # Store
            sample["data"].append(data)
            sample["label"].append(label)

            start += step

        sample["data"] = np.array(sample["data"])
        sample["label"] = np.array(sample["label"])

        return sample

    # Load audio file
    raw_audio, sample_rate = librosa.load(path=dataDir, sr=config.audioSampleRate)

    # Read label file to find min_start_time and max_end_time
    min_start_time = float('inf')
    max_end_time = 0.0
    label_intervals = []

    f = open(f"{labelDir}", "r")
    res = f.readlines()
    res = [line for line in res if line.strip()] # 去除空行
    for cur in res:
        cur = cur.strip("\n")

        parts = cur.split("\t")
        if len(parts) != 3:
            print(f"\n错误：在文件 {labelDir} 发现格式问题")
            print(parts)
            print

        # lStart, lEnd, lSet = cur.split("\t")
        lStart, lEnd, lSet = cur.replace('\t', ' ').split()
        lStart = float(lStart)
        lEnd = float(lEnd)
        lSet = int(lSet)

        # Update min and max times
        if lStart < min_start_time:
            min_start_time = lStart
        if lEnd > max_end_time:
            max_end_time = lEnd

        label_intervals.append((lStart, lEnd, lSet))
    f.close()

    # Calculate offset in samples
    offset_samples = int(min_start_time * config.audioSampleRate)
    max_end_samples = int(max_end_time * config.audioSampleRate)

    # Crop audio to only the labeled portion
    raw_audio = raw_audio[offset_samples:max_end_samples]

    # Process label file with offset (adjust to start from 0)
    CryList = None
    for lStart, lEnd, lSet in label_intervals:
        # Adjust timestamps relative to min_start_time
        lStart = float(lStart) - min_start_time
        lEnd = float(lEnd) - min_start_time

        lStart = int(lStart * config.audioSampleRate)
        lEnd = int(lEnd * config.audioSampleRate)

        if lSet == 1:
            if CryList is None:
                CryList = np.array([i for i in range(lStart, lEnd)])
            else:
                CryList = np.concatenate((CryList, np.array([i for i in range(lStart, lEnd)])))
    if CryList is None:
        CryList = []

    # Sliding windows to train audio (now only within labeled range)
    sample = SlidingWindows2Segments(CryList, raw_audio, sample_rate, config.slidingWindows, config.step, start=0)

    return sample


def main():
    dataDir = f"Data\\Crying_Github.wav"
    labelDir = f"Data\\Crying_Github.txt"
    sample = Wav2Segments(dataDir, labelDir)
    print(sample)


if __name__ == '__main__':
    main()






