from Q1_Representacao import Graph

def CicloEuleriano(graph: Graph):
    C = [] # Vetor de arestas visitadas
    i = False

    for _ in range(graph.GetEdgesQuantity()):
        C.append(False)

    flag_vertex = False
    for i in range(graph.GetVerticesQuantity()+1):      #for que verifica se o grafo é um ciclo euleriano (todos os vértices com grau par) e já busca um vértice que contenha ligação (no caso é pego o primeiro vértice que contenha ligação)
        if i > 0: 
            if graph.GetDegree(graph.GetLabel(i)) % 2 != 0:
                return [False, None]
            if graph.GetNeighborhood(graph.GetLabel(i)) != [] and flag_vertex == False:
                v = i
                flag_vertex = True

    cycle = buscarSubcicloEuleriano(graph, v, C)    #chama a função buscarsubciclo

    return [True, cycle]

def buscarSubcicloEuleriano(graph: Graph, v: int, C: list) -> list:
    cycleS = [v] # Ciclo começando em v
    cycleT = [] # pilha que retornará o ciclo
    Cindex = 0

    u = v
    j = True

    while j:
        neighbors = graph.GetNeighborhood(graph.GetLabel(u))

        for i in range(len(neighbors)):
            neighbor = graph.GetIndex(neighbors[i])
            coord = [u, neighbor]
            coordinv = [neighbor, u]    # é comparado se tanto o caminho de ida de um vertice a outro como o caminho de volta
            if (coord not in C) and (coordinv not in C):  # se não estiver na lista de arestas (C) é posto e adicionado o próximo vértice a pilha S
                C[Cindex] = (coord)   #representação das aresta é uma lista com dois vértices
                cycleS.append(neighbor)     #adiciona a pilha S
                u = neighbor
                Cindex += 1
                break

            elif i == len(neighbors)-1:
                x = cycleS.pop()
                cycleT.append(x)     #remove pilha S e adiciona na pilha T
                if cycleS != []:
                    u = cycleS[len(cycleS)-1]       #o vértice atual se torna a última posição acessada antes do ultim o vertice retirado, ou seja, vira o topo da pilha.

        if False in C or cycleS != []:                     #se não tiver nenhum False no C e a lista S estiver vazia acaba
            j = True
        else: 
            j = False
    
    return cycleT

if __name__ == '__main__':
    g = Graph()
    g.Read('GraphTest2.txt')
    cycle, cycleList = CicloEuleriano(g)

    if cycle: 
        print("1")
        cycleListStr = [str(a) for a in cycleList]
        print(",".join(cycleListStr))
    else:
        print("0")
