# coding: utf-8
"""
Standard gamepad mappings.

Pulled in to gamepad.py directly.
"""
import copy

from .gamepad import Gamepad

__all__ = ['PS4', 'XboxONE', 'Xbox']
all_controllers = copy.copy(__all__)
__all__.append(all_controllers)


class PS4(Gamepad):
    fullName = 'PlayStation 4 controller'

    def __init__(self, joystickNumber=0):
        Gamepad.__init__(self, joystickNumber)
        self.axisNames = {
            0: 'LAS -X',
            1: 'LAS -Y',
            2: 'LT',
            3: 'RAS -X',
            4: 'RAS -Y',
            5: 'RT',
            6: 'DPAD-X',
            7: 'DPAD-Y'
        }
        self.buttonNames = {
            0: 'A',
            1: 'B',
            2: 'X',
            3: 'Y',
            4: 'L1',
            5: 'R1',
            6: 'L2',
            7: 'R2',
            8: 'SHARE',
            9: 'LAS',
            10: 'RAS',
            11: 'L3',
            12: 'R3'
        }
        self._setupReverseMaps()


class XboxONE(Gamepad):
    fullName = 'Xbox ONE controller'

    def __init__(self, joystickNumber=0):
        Gamepad.__init__(self, joystickNumber)
        self.axisNames = {
            0: 'LAS -X',  # Left Analog Stick Left/Right
            1: 'LAS -Y',  # Left Analog Stick Up/Down
            2: 'RAS -X',  # Right Analog Stick Left/Right
            3: 'RAS -Y',  # Right Analog Stick Up/Down
            4: 'RT',  # Right Trigger
            5: 'LT',  # Left Trigger
            6: 'DPAD -X',  # D-Pad Left/Right
            7: 'DPAD -Y'  # D-Pad Up/Down
        }
        self.buttonNames = {
            0: 'A',  # A Button
            1: 'B',  # B Button
            3: 'X',  # X Button
            4: 'Y',  # Y Button
            6: 'LB',  # Left Bumper
            7: 'RB',  # Right Bumper
            11: 'START',  # Hamburger Button
            12: 'HOME',  # XBOX Button
            13: 'LAS',  # Left Analog Stick button
            14: 'RAS'  # Right Analog Stick button

        }
        self._setupReverseMaps()


Xbox = XboxONE
