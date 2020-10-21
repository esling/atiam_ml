import librosa as librosa
import numpy as np
import os as os

#
# Main function for importing a dataset
#
def import_dataset(classPath, typeD):
    # Type of files admitted
    wavExt = ['wav', 'wave']
    audioExt = wavExt + ['mp3', 'au', 'aiff', 'ogg']
    if (typeD == "classification"):
        # Listing of classes in the classification problem
        classesPaths = []
        # List classes first
        for item in os.listdir(classPath):
            if os.path.isdir(os.path.join(classPath, item)):
                classesPaths.append(item)
        # Set of classes data
        classData = {}
        # Number of classes
        nbClasses = len(classesPaths)
        # Names of the classes
        classData["name"] = []
        # Number of files
        classData["nb_files"] = np.zeros((len(classesPaths)))
        # Filenames for each class
        classData["filenames"] = []
        # Keep track of the full number of files
        fullNbFiles = 0
        print('    - Importing dataset %s.\n' % classPath)
        # Parse through all the classes
        for c in range(nbClasses):
            classData["name"].append(classesPaths[c])
            # Files for each class
            classFiles = [];
            for item in os.listdir(classPath + '/' + classesPaths[c]):
                if (os.path.splitext(item)[1][1:] in audioExt):
                    classFiles.append(item)
            classData["nb_files"][c] = len(classFiles)
            curFiles = []
            for f in range(len(classFiles)):
                curFiles.append(classPath + '/' + classesPaths[c] + '/' + classFiles[f])
            classData["filenames"].append(curFiles)
            fullNbFiles = fullNbFiles + classData["nb_files"][c]
        # Linearize into one flat structure
        filenames = []
        classes = [];
        curStart = 1;
        for c in range(nbClasses):
            nbFiles = classData["nb_files"][c];
            curFiles = classData["filenames"][c];
            filenames = filenames + curFiles;
            classes = classes + np.ndarray.tolist(np.repeat(c, nbFiles));
            curStart = curStart + nbFiles;
        dataStruct = {};
        dataStruct["filenames"] = filenames;
        dataStruct["classes"] = np.array(classes);
        dataStruct["class_names"] = classData["name"];
    elif typeD == 'music-speech':
        # Keep track of the full number of files
        fullNbFiles = 0;
        print('    - Importing dataset %s.\n' % classPath);
        classFiles = [];
        labFiles = [];
        # Parse through the audio files
        for item in os.listdir(classPath + '/music/'):
            if (os.path.splitext(item)[1][1:] in audioExt):
                classFiles.append(classPath + '/music/' + item);
                fPath = os.path.splitext(item)[0]
                labFiles.append(classPath + '/labels/' + fPath + '.lab');
        dataStruct = {};
        dataStruct["filenames"] = classFiles;
        dataStruct["labfiles"] = labFiles;
    else:
        raise Error('Unknown dataset type ' + typeD);
    return dataStruct

#
# Main transforms computation function
#
def compute_transforms(dataStruct, verbose = False):
    # Overall settings
    fSize = 1024
    wSize = fSize
    hSize = fSize//4
    refSr = 44100
    # Constant-Q settings
    fMin = librosa.note_to_hz('C1')
    nBins = 60 * 2
    # Number of files
    fullNbFiles = len(dataStruct["filenames"])
    # Create field for each transform
    dataStruct["signal"] = []
    dataStruct["srate"] = []
    dataStruct["spectrum_power"] = []
    dataStruct["spectrum_mel"] = []
    dataStruct["spectrum_chroma"] = []
    dataStruct["spectrum_CQT"] = []
    print('    - Performing transforms.');
    # Perform an analysis of spectral transform for each
    for f in range(fullNbFiles):
        if (verbose):
            print('      * %s.' % dataStruct["filenames"][f]);
        sig, sr = librosa.load(dataStruct["filenames"][f], mono=True, offset=0)
        if (sr != refSr):
            sig = librosa.resample(sig, sr, (sr/2))
        dataStruct["signal"].append(sig)
        dataStruct["srate"].append(sr)
        # Compute the FFT 
        psc = librosa.stft(sig, n_fft=fSize, win_length=wSize, hop_length=hSize, window='blackman')
        powerspec, phasespec = librosa.magphase(psc);
        dataStruct["spectrum_power"].append(powerspec[:(fSize//2), :])
        # Compute the mel spectrogram        
        wMel = librosa.feature.melspectrogram(sig, sr=sr, n_fft=fSize, hop_length=hSize)
        dataStruct["spectrum_mel"].append(wMel);
        # Compute the chromagram
        wChroma = librosa.feature.chroma_stft(S=powerspec**2, sr=sr)
        dataStruct["spectrum_chroma"].append(wChroma);
        # Compute the Constant-Q transform
        Xcq = librosa.cqt(sig, sr=refSr, n_bins=nBins, fmin=fMin, bins_per_octave=12 * 2)
        dataStruct["spectrum_CQT"].append(np.abs(Xcq));
    return dataStruct

#
# Main features computation function
#
def compute_features(dataStruct, verbose = False):
    # Window sizes
    wSize = 1024
    hSize = wSize // 4
    # Number of files
    nbFiles = len(dataStruct["filenames"])
    # Set of spectral features we will compute
    featuresYAAFE = ['SpectralVariation','SpectralFlux',
        'SpectralDecrease','SpectralFlatness','PerceptualSharpness',
        'SpectralRolloff','SpectralSlope', 'MFCC']
    featuresLibrosa = ['loudness', 'spectral_centroid', "spectral_bandwidth", 'spectral_contrast', 'spectral_flatness', 'spectral_rolloff'] 
    dataStruct["features_spectral"] = featuresLibrosa #+ featuresYAAFE
    # Initialize structure for spectral features
    for f in dataStruct["features_spectral"]:
        dataStruct[f] = []
        dataStruct[f + '_mean'] = []
        dataStruct[f + '_std'] = []
    print('    - Performing features.');
    # Computing the set of features
    for curFile in range(nbFiles):
        if (verbose):
            print('      * %s' % dataStruct["filenames"][curFile])
        curSignal = dataStruct["signal"][curFile]
        curSRate = dataStruct["srate"][curFile]
        # Add the specific features from Librosa
        dataStruct["loudness"].append(librosa.feature.rms(curSignal))
        # Compute the spectral centroid. [y, sr, S, n_fft, ...]
        dataStruct["spectral_centroid"].append(librosa.feature.spectral_centroid(curSignal))
        # Compute the spectral bandwidth. [y, sr, S, n_fft, ...]
        dataStruct["spectral_bandwidth"].append(librosa.feature.spectral_bandwidth(curSignal))
        # Compute spectral contrast [R16] , sr, S, n_fft, ...])	
        dataStruct["spectral_contrast"].append(librosa.feature.spectral_contrast(curSignal))
        # Compute the spectral flatness. [y, sr, S, n_fft, ...]
        dataStruct["spectral_flatness"].append(librosa.feature.spectral_flatness(curSignal))
        # Compute roll-off frequency
        dataStruct["spectral_rolloff"].append(librosa.feature.spectral_rolloff(curSignal))
        for f in featuresLibrosa:
            val = dataStruct[f][-1]
            dataStruct[f + '_mean'].append(np.mean(val))
            dataStruct[f + '_std'].append(np.std(val))
    return dataStruct

