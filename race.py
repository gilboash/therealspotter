import numpy as np

class RaceFSM:
    def __init__(self, track):
        self.track = track
        self.expected = 0

    def on_gate(self, detection, entry_vector):
        if self.expected >= len(self.track.gates):
            return "DONE"
        expected_gate = self.track.gates[self.expected]
        score = np.dot(entry_vector, np.array(expected_gate["entry_vector"]))
        if score > 0.5:
            self.expected += 1
            return "OK"
        return "WRONG"