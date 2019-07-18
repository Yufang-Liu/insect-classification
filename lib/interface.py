import subprocess
import logging
import sys
sys.path.append("..")
import numpy as np
import matplotlib.mlab as mlab
import scipy.io.wavfile as wf
import lib.wavfile as gistwf
import pydub


"""
sox and ffmpeg in need
"""


logging.basicConfig(format='%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.DEBUG)
np.random.seed(1337)
LOWER_BOUND = 1e-32
WAV_LENGTH = int(5 * np.floor(16000 / (512 - 256)))


def expand_ones(x, directions=([-1, 0], [1, 0], [0, -1], [0, 1])):
    """
    if 'x' array contains 1, this expands it in the given directions used for the mask applied to the spectrogram
    """
    expand = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 1:
                for direction in directions:
                    cx = i + direction[0]
                    cy = j + direction[1]
                    if 0 <= cx < x.shape[0] and 0 <= cy < x.shape[1]:
                        expand[cx, cy] = 1
    return x + expand


def audio2spectrogram(filename, expand_by_one=True, drop_zero_columns_percent=0.95):
    """
    function to filter the spectrogram based on the energy of the signal
    :param filename: data file name
    :param expand_by_one: if it is True, than the mask of the spectrogram will be expanded in every direction
    :param drop_zero_columns_percent: determines the ratio of 0 values along the
    frequency axis when a time slice is dropped
    :return: spectrogram, filtered spectrogram
    """
    # calculate the spectrogram
    try:
        rate, data = wf.read(filename)
        # print("STANDARD MODE")
    except ValueError:
        rate, data, _, _, _ = gistwf.read(filename)
        # print("GIST MODE")

    if len(data.shape) == 2:
        # data = data[:, 0] / 2 + data[:, 1] / 2
        data = data[:, 1]
    temp_spec = np.log10(mlab.specgram(data, NFFT=512, noverlap=256, Fs=rate)[0] + LOWER_BOUND)

    # drop higher frequencies
    temp_spec = temp_spec[0:200, :]
    temp_spec_filtered = np.copy(temp_spec)

    # we analyze the spectrogram by 20x30 sized cells
    # to achieve better accuracy the size of this cell should be fine-tuned
    row_borders = np.ceil(np.linspace(0, temp_spec.shape[0], 20))
    column_borders = np.hstack((np.ceil(np.arange(0, temp_spec.shape[1], 30)), temp_spec.shape[1]))
    row_borders = [int(x) for x in row_borders]
    column_borders = [int(x) for x in column_borders]
    keep_cells = np.ones((len(row_borders) - 1, len(column_borders) - 1))

    # we create a mask for the spectrogram: we scan the spectrogram with the 20x30 sized
    # cell and create 0 mask based on the mean and std of the spectrogram calculated for the cells and rows
    for i in range(len(row_borders) - 1):
        row_mean = np.mean(temp_spec[row_borders[i]:row_borders[i + 1], :])
        row_std = np.std(temp_spec[row_borders[i]:row_borders[i + 1], :])

        for j in range(len(column_borders) - 1):
            cell_mean = np.mean(temp_spec[row_borders[i]:row_borders[i + 1], column_borders[j]:column_borders[j + 1]])
            cell_max_top10_mean = np.mean(
                np.sort(temp_spec[row_borders[i]:row_borders[i + 1], column_borders[j]:column_borders[j + 1]],
                        axis=None)[-10:])
            # if cell_mean < 0 or (cell_max_top10_mean < (row_mean + row_std) * 1.5):
            # if cell_mean < row_mean or cell_max_top10_mean < row_mean + (row_std * 0.5):
            if cell_mean < row_mean or (cell_max_top10_mean < (row_mean + row_std) * 1.5):
                keep_cells[i, j] = 0

    # expand by ones (see above)
    if expand_by_one:
        keep_cells = expand_ones(keep_cells)

    # apply the mask to the spectrogram
    for i in range(keep_cells.shape[0]):
        for j in range(keep_cells.shape[1]):
            if not keep_cells[i, j]:
                temp_spec_filtered[row_borders[i]:row_borders[i + 1], column_borders[j]:column_borders[j + 1]] = 0

    # drop zero columns
    # the amount of zero values along axis 0 (frequency) is calculated for every column (time slice)
    # and it is dropped, if the number of zero values is higher than the dropZeroColumnsPercent
    # eg. drop_zero_columns_percent=0.95, than a column (time slice) is dropped,
    # if more than 95% of the values (frequencies) is 0
    temp_spec_filtered_backup = np.copy(temp_spec_filtered)
    temp_spec_filtered = np.delete(temp_spec_filtered, np.nonzero(
        (temp_spec_filtered == 0).sum(axis=0) > temp_spec_filtered.shape[0] * drop_zero_columns_percent), axis=1)

    # if every row was 0 than use the backed up spectrogram
    if temp_spec_filtered.shape[1] == 0:
        temp_spec_filtered = temp_spec_filtered_backup

    return temp_spec, temp_spec_filtered


def spec2slices(spec, wav_length):
    spec_length = spec.shape[1]
    if spec_length < wav_length:
        return None
    data = []
    i = 0
    while ((i + 1) * wav_length) < spec_length:
        data.append(spec[:, i * wav_length: (i + 1) * wav_length])
        i += 1
    data = np.array(data)
    spec = data
    return spec


def sound2numpy(filename):
    # assert " " not in filename
    tmp_filename = "{name}_tmp.wav".format(name=filename[:-4])
    if filename.endswith(".wav"):
        # subprocess.run(["sox", filename, "-r", "16000", tmp_filename])
        tmp_filename = filename
    elif filename.endswith(".mp3"):
        # subprocess.run(["ffmpeg", "-i", filename, "-acodec", "pcm_s16le", "-ar", "16000", tmp_filename])
        # subprocess.run(["ffmpeg", "-i", filename, tmp_filename])
        sound = pydub.AudioSegment.from_mp3(filename)
        sound.export(tmp_filename, format="wav")
        # subprocess.run(["sox", tmp_filename, "-r", "16000", tmp_filename])

    elif filename.endswith(".m4a"):
        # subprocess.run(["ffmpeg", "-i", filename, "-ar", "16000", tmp_filename])
        subprocess.run(["ffmpeg", "-i", filename, tmp_filename])
    else:
        logging.info("UNKNOWN FILE TYPE")
        return None
    try:
        _, filtered_spectrogram = audio2spectrogram(tmp_filename)
        filtered_spectrogram = spec2slices(filtered_spectrogram, WAV_LENGTH).astype(np.float32)
    except AttributeError:
        logging.info(filename)
        return None
    if not filename.endswith("wav"):
        subprocess.run(["rm", tmp_filename])
    try:
        assert filtered_spectrogram.shape[1:] == (200, 310)
    except AssertionError:
        logging.info("NO ENOUGH DATA")
        return None
    return filtered_spectrogram
