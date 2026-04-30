from numpy import size
from Q1_Representacao import Graph
import queue


with open("GraphPI.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()

with open("GraphPI.txt", "a", encoding="utf-8") as h:
    linha_1 = linhas[-2]
    linha_2 = linhas[-1]
    i = "b d 2"
    if linhas[-1] != i:
        h.write("\n"+i)
        k = 0
    print(linhas[-1])
    


if __name__ == "__main__":
    g = Graph()
    g.Read('GraphPI.txt')
    print(g.GetVerticesQuantity())
    print(g.GetEdgesQuantity())
    print('GraphPI.txt')

   