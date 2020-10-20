import librosa as librosa
import numpy as np
import os as os

#
# Main function for importing a dataset
#
def importDataset(classPath, typeD):
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
        classData["nbFiles"] = np.zeros((len(classesPaths)))
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
            classData["nbFiles"][c] = len(classFiles)
            curFiles = []
            for f in range(len(classFiles)):
                curFiles.append(classPath + '/' + classesPaths[c] + '/' + classFiles[f])
            classData["filenames"].append(curFiles)
            fullNbFiles = fullNbFiles + classData["nbFiles"][c]
        # Linearize into one flat structure
        filenames = []
        classes = [];
        curStart = 1;
        for c in range(nbClasses):
            nbFiles = classData["nbFiles"][c];
            curFiles = classData["filenames"][c];
            filenames = filenames + curFiles;
            classes = classes + np.ndarray.tolist(np.repeat(c, nbFiles));
            curStart = curStart + nbFiles;
        dataStruct = {};
        dataStruct["filenames"] = filenames;
        dataStruct["classes"] = np.array(classes);
        dataStruct["classNames"] = classData["name"];
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
def computeTransforms(dataStruct):
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
    dataStruct["sRate"] = []
    dataStruct["spectrumPower"] = []
    dataStruct["spectrumMel"] = []
    dataStruct["spectrumChroma"] = []
    dataStruct["spectrumCQT"] = []
    print('    - Performing transforms.');
    # Perform an analysis of spectral transform for each
    for f in range(fullNbFiles):
        print('      * %s.' % dataStruct["filenames"][f]);
        sig, sr = librosa.load(dataStruct["filenames"][f], mono=True, offset=0)
        if (sr != refSr):
            sig = librosa.resample(sig, sr, (sr/2))
        dataStruct["signal"].append(sig)
        dataStruct["sRate"].append(sr)
        # Compute the FFT 
        psc = librosa.stft(sig, n_fft=fSize, win_length=wSize, hop_length=hSize, window='blackman')
        powerspec, phasespec = librosa.magphase(psc);
        dataStruct["spectrumPower"].append(powerspec[:(fSize//2), :])
        # Compute the mel spectrogram        
        wMel = librosa.feature.melspectrogram(sig, sr=sr, n_fft=fSize, hop_length=hSize)
        dataStruct["spectrumMel"].append(wMel);
        # Compute the chromagram
        wChroma = librosa.feature.chroma_stft(S=powerspec**2, sr=sr)
        dataStruct["spectrumChroma"].append(wChroma);
        # Compute the Constant-Q transform
        Xcq = librosa.cqt(sig, sr=refSr, n_bins=nBins, fmin=fMin, bins_per_octave=12 * 2)
        dataStruct["spectrumCQT"].append(np.abs(Xcq));
    return dataStruct

import numpy as np
import librosa as librosa

#
# Main features computation function
#
def computeFeatures(dataStruct):
    # Window sizes
    wSize = 1024
    hSize = wSize // 4
    # Number of files
    nbFiles = len(dataStruct["filenames"])
    # Set of spectral features we will compute
    featuresYAAFE = ['SpectralVariation','SpectralFlux',
        'SpectralDecrease','SpectralFlatness','PerceptualSharpness',
        'SpectralRolloff','SpectralSlope', 'MFCC']
    featuresLibrosa = ['Loudness', 'SpectralCentroid', 'SpectralContrast', 'SpectralRolloff'] 
    dataStruct["featuresSpectral"] = featuresLibrosa #+ featuresYAAFE
    # Initialize structure for spectral features
    for f in dataStruct["featuresSpectral"]:
        dataStruct[f] = []
        dataStruct[f + 'Mean'] = []
        dataStruct[f + 'Std'] = []
    print('    - Performing features.');
    # Computing the set of features
    for curFile in range(nbFiles):
        print('      * %s' % dataStruct["filenames"][curFile])
        curSignal = dataStruct["signal"][curFile]
        curSRate = dataStruct["sRate"][curFile]
        # Create YAAFE extraction engine
        #fp = y.FeaturePlan(sample_rate=curSRate)
        #for f in featuresYAAFE:
        #    fp.addFeature(f+': '+f+' blockSize='+str(wSize)+' stepSize='+str(hSize))
        #engine = y.Engine()
        #engine.load(fp.getDataFlow())
        #features = engine.processAudio(curSignal.astype('float64').reshape((1, curSignal.shape[0])))
        #for key, val in sorted(features.items()):
        #    dataStruct[key].append(val)
        #    dataStruct[key + 'Mean'].append(np.mean(val))
        #    dataStruct[key + 'Std'].append(np.std(val))
        # Add the specific features from Librosa
        dataStruct["Loudness"].append(librosa.feature.rmse(curSignal))
        # Compute the spectral centroid. [y, sr, S, n_fft, ...]
        dataStruct["SpectralCentroid"].append(librosa.feature.spectral_centroid(curSignal))
        # Compute spectral contrast [R16] , sr, S, n_fft, ...])	
        dataStruct["SpectralContrast"].append(librosa.feature.spectral_contrast(curSignal))
        # Compute roll-off frequency
        dataStruct["SpectralRolloff"].append(librosa.feature.spectral_rolloff(curSignal))
        for f in featuresLibrosa:
            val = dataStruct[f][-1]
            dataStruct[f + 'Mean'].append(np.mean(val))
            dataStruct[f + 'Std'].append(np.std(val))
    return dataStruct

