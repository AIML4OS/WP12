from nicegui import ui
from math import floor

class SliderControler:
    def __init__(self, label_str:str, system = None, attribute:str = "" ,  max = 10, min=1):
        self.value = floor((max + min)/2) 
        self.min = min
        self.max = max
        self.label_str = label_str
        self.system = system
        self.attribute = attribute
        self.gui()

    def gui(self):
            ui.label(self.label_str)
            slider = ui.slider(min=self.min, 
                            max=self.max, 
                            value=self.value,
                            on_change=self.update)
            self.label = ui.label(f"Current: {self.value}")
    

    def update(self, e):
        self.value = e.value
        self.label.text = f"Current: {self.value}"
        if self.system and self.attribute:
            if hasattr(self.system, self.attribute):
                setattr(self.system, self.attribute, self.value)
                # print(getattr(self.system, self.attribute))
            else:
                raise AttributeError(
                     f"RAG system {self.system} does not have attribute {self.attribute}!"
                )



if __name__ in {"__main__", "__mp_main__"}:
    controller = SliderControler("Max Keywords")
    ui.run()


