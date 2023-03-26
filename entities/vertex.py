class Vertex:
    def __init__(self, id, bonus):
        self.id = int(id)
        self.bonus = float(bonus)
        self.penalty = 99999999 if id == "0" else 50