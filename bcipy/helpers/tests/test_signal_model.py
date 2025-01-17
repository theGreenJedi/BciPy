import unittest
import numpy as np

from bcipy.helpers.signal_model import CopyPhraseWrapper
from bcipy.helpers.task import alphabet


class TestSignalModelRelated(unittest.TestCase):
    """Test CopyPhraseWrapper"""

    def test_valid_letters(self):
        alp = alphabet()
        cp = CopyPhraseWrapper(
            min_num_seq=1,
            max_num_seq=50,
            signal_model=None,
            fs=25,
            k=2,
            alp=alp,
            task_list=[('HELLO_WORLD', 'HE')],
            is_txt_stim=True,
            device_name='LSL',
            evidence_names=['LM', 'ERP'],
            device_channels=[
                'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9',
                'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'TRG'
            ],
            stimuli_timing=[0.5, 0.25])

        triggers = [('+', 0.0), ('H', 0.5670222830376588),
                    ('D', 0.8171830819919705), ('J', 1.0843321380089037),
                    ('B', 1.3329724550130777), ('C', 1.5825864360085689),
                    ('A', 1.833380013005808), ('F', 2.083211077027954),
                    ('G', 2.333359022042714), ('I', 2.583265081048012),
                    ('E', 2.833274284028448)]
        target_info = [
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]

        letters, timings, labels = cp.letter_info(triggers, target_info)
        expected_letters = ['H', 'D', 'J', 'B', 'C', 'A', 'F', 'G', 'I', 'E']
        expected_time = [
            0.5670222830376588, 0.8171830819919705, 1.0843321380089037,
            1.3329724550130777, 1.5825864360085689, 1.833380013005808,
            2.083211077027954, 2.333359022042714, 2.583265081048012,
            2.833274284028448
        ]
        self.assertEqual(expected_letters, letters)
        self.assertEqual(expected_time, timings)
        self.assertEqual(len(letters), len(labels))

        triggers = [('calibration_trigger', 0.0), ('+', 0.1), ('H', 0.5670222830376588),
                    ('D', 0.8171830819919705), ('J', 1.0843321380089037),
                    ('B', 1.3329724550130777), ('C', 1.5825864360085689),
                    ('A', 1.833380013005808), ('F', 2.083211077027954),
                    ('G', 2.333359022042714), ('I', 2.583265081048012),
                    ('E', 2.833274284028448)]
        target_info = [
            'calib', 'fixation', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget', 'nontarget', 'nontarget', 'nontarget', 'nontarget',
            'nontarget'
        ]
        letters, timings, labels = cp.letter_info(triggers, target_info)
        self.assertEqual(expected_letters, letters)
        self.assertEqual(expected_time, timings)
        self.assertEqual(['nontarget'] * (len(letters)), labels)

        # Test it throws an exception when letter is outside alphabet
        with self.assertRaises(Exception):
            cp.letter_info([('A', 0.0), ('*', 1.0)], ['nontarget', 'nontarget'])