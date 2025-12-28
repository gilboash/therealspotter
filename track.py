import json

class Track:
    def __init__(self, expected_gates):
        self.expected_gates = expected_gates
        self.gates = []

    def add_gate(self, detection, entry_vector):
        self.gates.append({
            "embedding": detection["embedding"].tolist(),
            "entry_vector": entry_vector.tolist()
        })

    def complete(self):
        return len(self.gates) >= self.expected_gates

    def save(self, path="track.json"):
        with open(path, "w") as f:
            json.dump({
                "expected_gates": self.expected_gates,
                "gates": self.gates
            }, f, indent=2)

    @staticmethod
    def load(path="track.json"):
        with open(path) as f:
            data = json.load(f)
        t = Track(data["expected_gates"])
        t.gates = data["gates"]
        return t