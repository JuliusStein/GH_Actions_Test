import os
import sys

#USAGE 'python renameFilesCogweb.py FILENAME' will rename using dictionary below
#Expects the .md files that were changed as command line arguments
#For use in Github Actions Workflow

labels = {
    "AudioSignalBasics_STUDENT.ipynb": "01_AudioSignalBasics.ipynb",
    "WorkingWithMic_STUDENT.ipynb": "02_WorkingWithMic.ipynb",
    "AnalogToDigital_STUDENT.ipynb": "03_AnalogToDigital.ipynb",
    "BasicsOfDFT_STUDENT.ipynb": "04_BasicsOfDFT.ipynb",
    "DFTOfVariousSignals_STUDENT.ipynb": "05_DFTOfVariousSignals.ipynb",
    "ApplicationsOfDFTs_STUDENT.ipynb": "06_.ApplicationsOfDFTsipynb",
    "Spectrogram_STUDENT.ipynb": "07_Spectrogram.ipynb",
    "spectrogram_STUDENT.ipynb": "07_Spectrogram.ipynb",
    "PeakFinding_STUDENT.ipynb": "08_PeakFinding.ipynb",

    "Data_Exploration_STUDENT.ipynb": "01_Data_Exploration.ipynb",
    "Autodiff_and_Grad_Descent_STUDENT.ipynb": "02_Autodiff_and_Grad_Descent.ipynb",
    "Linear_Regression_Exercise_STUDENT.ipynb": "03_Linear_Regression_Exercise.ipynb",
    "UniversalFunctionApprox_STUDENT.ipynb": "04_UniversalFunctionApprox.ipynb",
    "TendrilClassification_STUDENT.ipynb": "05_TendrilClassification.ipynb",
    "TendrilClassificationMyNN_STUDENT.ipynb": "06_TendrilClassificationMyNN.ipynb",
    "Cifar10MyGrad_STUDENT.ipynb": "07_Cifar10MyGrad.ipynb",
    "WritingCNNOperations_STUDENT.ipynb": "08_WritingCNNOperations.ipynb",
    "MyGradMnist_STUDENT.ipynb": "09_MyGradMnist.ipynb",

    "LanguageModels_STUDENT.ipynb": "01_LanguageModels.ipynb",
    "BagOfWords_STUDENT.ipynb": "02_BagOfWords.ipynb",
    "FunWithWordEmbeddings_STUDENT.ipynb": "03_FunWithWordEmbeddings.ipynb",
    "LinearAutoencoder_STUDENT.ipynb": "04_LinearAutoencoder.ipynb",
    "AutoencoderWordEmbeddings_STUDENT.ipynb": "05_AutoencoderWordEmbeddings.ipynb",
    "MyGradSimpleCellRNN_STUDENT.ipynb": "06_MyGradSimpleCellRNN.ipynb",
    "NotebookOfFailure_STUDENT.ipynb": "07_NotebookOfFailure.ipynb",
    "Seq2SeqModels_STUDENT.ipynb": "08_Seq2SeqModels.ipynb",
    "Transformers_STUDENT.ipynb": "09_Transformers.ipynb",
}



def rename_cogfile(filename):
    if "Audio" in filename:
        path = "test_outputs/Audio/"
    elif "Video" in filename:
        path = "test_outputs/Video/"
    elif "Language" in filename:
        path = "test_outputs/Language/"
    else:
        return

    #Could take path to student ipynb file, but checks for .md and renames to match cogbooks
    file = (filename.rsplit('/',1))[1]
    if ".md" in file:
        file = file[:-3]+"_STUDENT.ipynb"

    try:
        newFile = labels[file]
    except:
        newFile = file[:-14] + ".ipynb"

    os.rename(path+file, path+newFile)

if __name__ == "__main__":
    f = str(sys.argv[1])
    rename_cogfile(f)
