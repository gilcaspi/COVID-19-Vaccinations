import enum


class EColor(str,  enum.Enum):
    GREEN = 'rgb(0,155,0)'
    ORANGE = 'rgb(255,137,0)'
    RED = 'rgb(255,0,0)'

    def __str__(self):
        return self.value
