import logging
from typing import Any
import numpy as np
from bcipy.tasks.exceptions import InsufficientDataException
log = logging.getLogger(__name__)

def process_data_for_decision(
        sequence_timing,
        daq,
        window,
        parameters,
        first_session_stim_time,
        static_offset=None,
        buf_length=None):
    """Process Data for Decision.

    Processes the raw data (triggers and EEG) into a form that can be passed to
    signal processing and classifiers.

    Parameters
    ----------
        sequence_timing(array): array of tuples containing stimulus timing and
            text
        daq (object): data acquisition object
        window: window to reactivate if deactivated by windows
        parameters: parameters dictionary
        first_session_stim_time (float): time that the first stimuli was presented
            for the session. Used to calculate offsets.

    Returns
    -------
        (raw_data, triggers, target_info) tuple
    """

    # Get timing of the first and last stimuli
    _, first_stim_time = sequence_timing[0]
    _, last_stim_time = sequence_timing[-1]

    static_offset = static_offset or parameters['static_trigger_offset']
    # if there is an argument supplied for buffer length use that
    if buf_length:
        buffer_length = buf_length
    else:
        buffer_length = parameters['len_data_sequence_buffer']

    # get any offset calculated from the daq
    daq_offset = daq.offset

    if daq_offset:
        offset = daq_offset - first_session_stim_time + static_offset
        time1 = (first_stim_time + offset) * daq.device_info.fs
        time2 = (last_stim_time + offset + buffer_length) * daq.device_info.fs
    else:
        time1 = (first_stim_time + static_offset) * daq.device_info.fs
        time2 = (last_stim_time + static_offset + buffer_length) * daq.device_info.fs

    # Construct triggers to send off for processing
    triggers = [(text, ((timing) - first_stim_time))
                for text, timing in sequence_timing]

    # Assign labels for triggers
    # TODO: This doesn't seem useful and is misleading
    target_info = ['nontarget'] * len(triggers)

    # Define the amount of data required for any processing to occur.
    data_limit = (last_stim_time - first_stim_time + buffer_length) * daq.device_info.fs

    # Query for raw data
    try:
        # Call get_data method on daq with start/end
        raw_data = daq.get_data(start=time1, end=time2, win=window)

        # If not enough raw_data returned in the first query, let's try again
        #  using only the start param. This is known issue on Windows.
        #  #windowsbug
        if len(raw_data) < data_limit:

            # Call get_data method on daq with just start
            raw_data = daq.get_data(start=time1, win=window)

            # If there is still insufficient data returned, throw an error
            if len(raw_data) < data_limit:
                message = f'Process Data Error: Not enough data received to process. ' \
                          f'Data Limit = {data_limit}. Data received = {len(raw_data)}'
                log.error(message)
                raise InsufficientDataException(message)

        # Take only the sensor data from raw data and transpose it
        raw_data = np.array([np.array([_float_val(col) for col in record.data])
                             for record in raw_data],
                            dtype=np.float64).transpose()

    except Exception as e:
        log.error(f'Uncaught Error in Process Data for Decision: {e}')
        raise e

    return raw_data, triggers, target_info

def _float_val(col: Any) -> float:
    """Convert marker data to float values so we can put them in a
    typed np.array. The marker column has type float if it has a 0.0
    value, and would only have type str for a marker value."""
    if isinstance(col, str):
        return 1.0
    return float(col)
