import matplotlib.pyplot as plt

class Quadra():
    def __init__(self) -> None:
        self.especificações = {} # lista de especificações
    def quadra(self,prop):
        fig, ax = plt.subplots(figsize=(5, 5))
        rect_1 = plt.Rectangle((-54.85*prop,-118.85*prop),109.7*prop, 237.7*prop, 
            color = (0.85, 0.35, 0.19),
            fill=False)
        ax.add_patch(rect_1)
        rect_2 = plt.Rectangle((-41.1*prop,-118.85*prop),82.2*prop,237.7*prop, 
            color = (0.85, 0.35, 0.19),
            fill = False)
        ax.add_patch(rect_2)
        rect_3 = plt.Rectangle((-54.85*prop,118.9*prop), 109.7*prop, -118.9*prop,
            color = (0.85, 0.35, 0.19),
            fill = False)
        ax.add_patch(rect_3)
        rect_4 = plt.Rectangle((-41.1*prop,64.0*prop), 82.2*prop, -128.0*prop,
            color = (0.85, 0.35, 0.19),
            fill = False)
        ax.add_patch(rect_4)
        rect_5 = plt.Rectangle((41.1*prop,64.0*prop), -41.1*prop, -128.0*prop,
            color = (0.85, 0.35, 0.19),
            fill = False)
        ax.add_patch(rect_5)
        serveline_1 = plt.Rectangle((-0.5*prop,118.9*prop),1*prop,-5*prop,
            color = (0.85, 0.35, 0.19),
            fill = True)
        ax.add_patch(serveline_1)
        serveline_2 = plt.Rectangle((-0.5*prop,-118.9*prop),1*prop,5*prop,
            color = (0.85, 0.35, 0.19),
            fill = True)
        ax.add_patch(serveline_2)