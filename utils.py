from pathlib import Path
import random
import string
import sys


class SocketConcatenator(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """
    Pipes standard input to standard input and to a file.
    """
    print("Standard output piped to file:")
    f = open(Path(file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)
    sys.stderr = SocketConcatenator(sys.stderr, f)


nato_alphabet = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliett",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
}


def random_codeword():
    """
    Return e.g.:
        Alpha_13, Zulu_39, X-ray_95
    """
    random_letter = random.choice(string.ascii_uppercase)
    return "{}_{}".format(nato_alphabet[random_letter], random.randint(10, 99))
