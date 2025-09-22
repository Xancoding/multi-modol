from Feature_Extraction import acoustic_features_and_spectrogram
from Preprocessing import Wav2Segments
from Model.model import CNN_Model, evaluate, DatasetLoad, modelMetrics
from torch.utils.data import DataLoader
import torch
import config
import warnings


import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*NaN values found in confusion matrix.*")


def main():
    # Wav to segments
    # dataDir = f"Data/No_Crying_Github.wav"
    # labelDir = f"Data/No_Crying_Github.txt"
    prefix = "/data/Leo/mm/data/" + config.data + "/"
    folder_path = prefix + "Label/"
    infants = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.txt')]
    # infants = ["50cm3.05kg", "52cm3.4kg"]
    for infant in infants:
        with open(config.data + '.csv', 'a', encoding='utf-8') as f:
            f.write(infant + '\t') 
        dataDir = prefix + "data/" + infant + ".wav"
        labelDir = prefix + "Label/" + infant + ".txt"
        sample = Wav2Segments(dataDir, labelDir)

        # Feature extract
        acoustic_feature, spectrogram_feature = acoustic_features_and_spectrogram(sample["data"])
        mean_train_data = 0.02303917053151151
        std_train_data = 0.053298024206659946
        test_data = (spectrogram_feature - mean_train_data) / std_train_data

        test_dataloader = DataLoader(
            dataset=DatasetLoad(test_data, sample["label"]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Model load
        device = "cuda"
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        model = CNN_Model().to(device)
        model.load_state_dict(torch.load(f"./Model/Weight/Model_0_with_SAF.pkl", weights_only=True))

        evaluate(model, test_dataloader, device)

        with open(config.data + '.csv', 'a', encoding='utf-8') as f:
            f.write('\n')
        # _, acc, recall, precision, f1, FPR, confu_ma = \
        #     evaluate(model, test_dataloader, device)
        # print("acc, recall, precision, f1, FPR, confu_ma")
        # print(acc, recall, precision, f1, FPR, confu_ma)



if __name__ == '__main__':
    main()








