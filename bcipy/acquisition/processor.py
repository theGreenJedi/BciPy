"""DAQ Item Processors"""

import csv
import sys
from bcipy.acquisition.device_info import DeviceInfo


class Processor():
    """Abstract base class for an object that can be used to process data
    acquisition data.
    """

    def __init__(self):
        super(Processor, self).__init__()
        self._device_info = None

    def set_device_info(self, device_info: DeviceInfo):
        """
        Sets the device info, which may be used by the process method.

        Parameters
        ----------
            device_info : DeviceInfo
                Metadata with the parameters used to collect data.
        """
        self._device_info = device_info

    def _check_device_info(self):
        """
        Checks that the sample_metadata has been set.

        Exceptions
        ----------
            throws AssertionError unless set.
        """
        assert self._device_info is not None, \
            "device_info is not set; Initialize with set_device_info."

    def __enter__(self):
        # Device info should be set before using as a context.
        self._check_device_info()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        pass

    def process(self, record, timestamp=None):
        """Process the given record.

        Parameters
        ----------
            record: list<float>
                A single record returned by from reading a data packet with a
                float value for each channel.
            timestamp: float, optional
                time at which the record was obtained
        """
        raise NotImplementedError('Subclass must define the process method')


class NullProcessor(Processor):
    """A DAQ item Processor that doesn't do anything."""

    def process(self, record, timestamp=None):
        pass


class FileWriter(Processor):
    """A DAQ item Processor that writes items to a file.

    Parameters
    ----------
        filename : str
            Filename to write to.
    """

    def __init__(self, filename):
        super(FileWriter, self).__init__()
        self._filename = filename
        self._file = None
        self._writer = None

    # @override ; context manager
    def __enter__(self):
        self._check_device_info()

        # For python 2, writer needs the 'wb' option in order to work on
        # Windows. If using #Python3 'w' is needed.
        if sys.version_info >= (3, 0, 0):
            self._file = open(self._filename, 'w', newline='')
        else:
            self._file = open(self._filename, 'wb')

        self._writer = csv.writer(self._file, delimiter=',')
        self._writer.writerow(['daq_type', self._device_info.name])
        self._writer.writerow(['sample_rate', self._device_info.fs])
        self._writer.writerow(['timestamp'] + self._device_info.channels)
        return self

    # @override ; context manager
    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._file.close()

    def process(self, record, timestamp=None):
        if self._writer:
            self._writer.writerow([timestamp] + record)


class LslProcessor(Processor):
    """A DAQ item processor that writes to an LSL data stream."""

    def __init__(self):
        super(LslProcessor, self).__init__()
        self._outlet = None

    # @override
    def set_device_info(self, device_info):
        import pylsl
        import uuid

        super(LslProcessor, self).set_device_info(device_info)
        channels = self._device_info.channels
        info = pylsl.StreamInfo(self._device_info.name, 'EEG',
                                len(channels),
                                self._device_info.fs,
                                'float32', str(uuid.uuid4()))
        meta_channels = info.desc().append_child('channels')
        for channel in channels:
            meta_channels.append_child('channel') \
                .append_child_value('label', str(channel)) \
                .append_child_value('unit', 'microvolts') \
                .append_child_value('type', 'EEG')

        self._outlet = pylsl.StreamOutlet(info)

    def process(self, record, timestamp=None):
        if self._outlet:
            self._outlet.push_sample(record, timestamp)


class DispatchProcessor(Processor):
    """Processor that delegates to one or more other processors. Processors
    may be passed in through the constructor or after creating an instance and
    calling it's add method."""

    def __init__(self, *args):
        super(DispatchProcessor, self).__init__()
        self.processors = []
        for proc in args:
            self.add(proc)

    # @override
    def set_device_info(self, device_info):
        super(DispatchProcessor, self).set_device_info(device_info)
        for proc in self.processors:
            proc.set_device_info(device_info)

    def add(self, proc: Processor):
        """Add a processor."""
        self.processors.append(proc)
        if self._device_info:
            proc.set_device_info(self._device_info)

    def remove(self, proc: Processor):
        """Remove a processor."""
        self.processors.remove(proc)

    # @override
    def process(self, record, timestamp=None):
        for proc in self.processors:
            proc.process(record, timestamp)

    # @override ; context manager
    def __enter__(self):
        for proc in self.processors:
            proc.__enter__()
        return self

    # @override ; context manager
    def __exit__(self, _exc_type, _exc_value, _traceback):
        for proc in self.processors:
            proc.__exit__(_exc_type, _exc_value, _traceback)
